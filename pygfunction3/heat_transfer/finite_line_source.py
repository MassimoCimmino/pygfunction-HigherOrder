# -*- coding: utf-8 -*-
from collections.abc import Callable
import warnings

import numpy as np
import numpy.typing as npt
from scipy.integrate import quad_vec
from scipy.special import erfc, erf

from ..utils import mirror_point


def finite_line_source(
        time: npt.ArrayLike,
        alpha: float,
        p,
        borehole,
        basis,
        real_source: bool = True,
        mirror_source: bool = True
        ) -> np.ndarray:

    if real_source or mirror_source:
        integrand = _finite_line_source_integrand(p, borehole, basis, real_source=real_source, mirror_source=mirror_source)
        h = quad_vec(
            integrand,
            -1., 1.,
            args=(time, alpha),
            epsabs=1.0e-6,
            epsrel=1.0e-6
        )[0]
    else:
        # TODO: return zeros of correct shape
        shape = np.shape(time) + tuple(np.ones(np.ndim(p), dtype=int))
        h = np.zeros(shape)
    return h


def _finite_line_source_integrand(
        p,
        borehole,
        basis,
        real_source: bool = True,
        mirror_source: bool = True
        ) -> Callable:

    # Mirror points
    if mirror_source:
        p_mirror = mirror_point(p)

    # Integration function
    if real_source and mirror_source:
        def integrand(xi, time, alpha):
            p_self = borehole._position(xi, borehole.L, borehole.p_middle, borehole.unit_vector)
            dis = np.linalg.norm(p - p_self, axis=-1)
            dis_mirror = np.linalg.norm(p_mirror - p_self, axis=-1)
            psi = basis.basis_function(xi)
            two_sqrt_alpha_time = 2 * np.sqrt(alpha * time)
            dh = 0.25 * borehole.L * np.multiply.outer(
                erfc(np.divide.outer(dis, two_sqrt_alpha_time).T) / dis
                - erfc(np.divide.outer(dis_mirror, two_sqrt_alpha_time).T) / dis_mirror,
                psi
            )
            return dh
    elif real_source:
        def integrand(xi, time, alpha):
            p_self = borehole._position(xi, borehole.L, borehole.p_middle, borehole.unit_vector)
            dis = np.linalg.norm(p - p_self, axis=-1)
            psi = basis.basis_function(xi)
            two_sqrt_alpha_time = 2 * np.sqrt(alpha * time)
            dh = 0.25 * borehole.L * np.multiply.outer(
                erfc(np.divide.outer(dis, two_sqrt_alpha_time).T) / dis,
                psi
            )
            return dh
    elif mirror_source:
        def integrand(xi, time, alpha):
            p_self = borehole._position(xi, borehole.L, borehole.p_middle, borehole.unit_vector)
            dis_mirror = np.linalg.norm(p_mirror - p_self, axis=-1)
            psi = basis.basis_function(xi)
            two_sqrt_alpha_time = 2 * np.sqrt(alpha * time)
            dh = -0.25 * borehole.L * np.multiply.outer(
                erfc(np.divide.outer(dis_mirror, two_sqrt_alpha_time).T) / dis_mirror,
                psi
            )
            return dh
    else:
        # TODO: return zeros of correct shape
        def integrand(xi, time, alpha):
            shape = np.shape(time) + tuple(np.ones(np.ndim(p), dtype=int))
            dh = np.zeros(shape)

    return integrand


def _elementary_function(u, r, s):
    arg = s**2 + r**2
    sqrt_arg = np.sqrt(arg)
    term1 = np.exp(-u**2 * arg) / u
    term2 = sqrt_arg * np.sqrt(np.pi) * erfc(u * sqrt_arg)
    return term1 - term2
