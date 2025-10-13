# -*- coding: utf-8 -*-
from collections.abc import Callable
import warnings

import numpy as np
import numpy.typing as npt
from scipy.integrate import quad_vec
from scipy.special import erf

from ..utils.error_function import _error_function_coefficients
from ..utils import exp1, mirror_point


def zeroth_order_finite_line_source(
        time: npt.ArrayLike,
        alpha: float,
        p: npt.ArrayLike,
        borehole,
        r_min: float = 0.,
        deg: int | None = None,
        real_source: bool = True,
        mirror_source: bool = True
        ) -> np.ndarray:
    # Real points
    if real_source:
        r, s = borehole.radial_axial_position(p, r_min=r_min)
    else:
        s = np.zeros_like(p)
        r = np.zeros_like(p)
    # Mirror points
    if mirror_source:
        p_mirror = mirror_point(p)
        r_mirror, s_mirror = borehole.radial_axial_position(p_mirror, r_min=r_min)
    else:
        s_mirror = np.zeros_like(p)
        r_mirror = np.zeros_like(p)
    # Lower bound of integration
    u = 1 / np.sqrt(4 * alpha * time)

    h = _zeroth_order_finite_line_source(u, borehole.L, r, s, r_mirror, s_mirror, deg=deg, real_source=real_source, mirror_source=mirror_source)

    return h


def X_0(u, L, s_proj):
    term1 = erf(u * (L - s_proj))
    term2 = erf(u * s_proj)
    result = 0.5 / u * (term1 + term2)
    return result


def _zeroth_order_finite_line_source(u, L, r, s, r_mirror, s_mirror, deg=None, real_source=True, mirror_source=True):
    if deg is None:
        h = __zeroth_order_finite_line_source(u, L, r, s, r_mirror, s_mirror, real_source=real_source, mirror_source=mirror_source)
    else:
        h = __zeroth_order_finite_line_source_approximation(u, L, r, s, r_mirror, s_mirror, deg=deg, real_source=real_source, mirror_source=mirror_source)
    return h


def __zeroth_order_finite_line_source(u, L, r, s, r_mirror, s_mirror, real_source=True, mirror_source=True):
    if real_source and mirror_source:
        def integrand(u):
            return np.exp(-(u * r)**2) * X_0(u, L, s) - np.exp(-(u * r_mirror)**2) * X_0(u, L, s_mirror)
    elif real_source:
        def integrand(u):
            return np.exp(-(u * r)**2) * X_0(u, L, s)
    elif mirror_source:
        def integrand(u):
            return -np.exp(-(u * r_mirror)**2) * X_0(u, L, s_mirror)
    else:
        shape = np.shape(u) + np.shape(s)
        return np.zeros(shape)
    # Lower bound of integration
    a = u
    # Upper bound of integration
    b = np.concatenate(([np.inf], a[:-1]))
    h = np.cumsum(np.stack(
        [quad_vec(integrand, a_i, b_i, epsabs=1e-4, epsrel=1e-6)[0]
            for a_i, b_i in zip(a, b)],
        axis=0), axis=0)
    return h

def __zeroth_order_finite_line_source_approximation(u, L, r, s, r_mirror, s_mirror, deg=11, real_source=True, mirror_source=True):
    _a_erf, _b_erf = _error_function_coefficients(deg)
    # All points
    if real_source and mirror_source:
        etas = np.stack((s, s - L, s_mirror, s_mirror - L), axis=-1)
        sigmas = np.stack((r, r, r_mirror, r_mirror), axis=-1)
        q = np.array([1., -1., -1., 1.])
    elif real_source:
        etas = np.stack((s, s - L), axis=-1)
        sigmas = np.stack((r, r), axis=-1)
        q = np.array([1., -1.])
    elif mirror_source:
        etas = np.stack((s_mirror, s_mirror - L), axis=-1)
        sigmas = np.stack((r_mirror, r_mirror), axis=-1)
        q = np.array([-1., 1.])
    else:
        shape = np.shape(u) + np.shape(s)
        return np.zeros(shape)
    # Approximation of the integral
    h = 0.25 * np.sign(etas) * (
        exp1(
            np.multiply.outer(
                u**2,
                np.expand_dims(sigmas**2, axis=-1) + np.multiply.outer(etas**2, _b_erf)
                )
        ) @ _a_erf
    ) @ q
    return h
