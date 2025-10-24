# -*- coding: utf-8 -*-
import warnings

import numpy as np
import numpy.typing as npt
from scipy.integrate import quad_vec

from ._inner_integral import _zeroth_order_inner_integral
from ...utils.error_function import _error_function_coefficients
from ...utils import exp1, mirror_point


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
    """Zeroth order finite line source solution."""
    shape = p.shape[:-1]
    # Real points
    if real_source:
        r, s = borehole.radial_axial_position(p, r_min=r_min)
    else:
        r = np.empty(shape)
        s = np.empty(shape)
    # Mirror points
    if mirror_source:
        p_mirror = mirror_point(p)
        r_mirror, s_mirror = borehole.radial_axial_position(p_mirror, r_min=r_min)
    else:
        r_mirror = np.empty(shape)
        s_mirror = np.empty(shape)
    # Lower bound of integration
    u = 1 / np.sqrt(4 * alpha * time)

    I_0 = _zeroth_order_finite_line_source(u, borehole.L, r, s, r_mirror, s_mirror, deg=deg, real_source=real_source, mirror_source=mirror_source)

    return I_0


def _zeroth_order_finite_line_source(u, L, r, s, r_mirror, s_mirror, deg=None, real_source=True, mirror_source=True):
    """Zeroth order finite line source solution."""
    if deg is None:
        I_0 = __zeroth_order_finite_line_source_integral(u, L, r, s, r_mirror, s_mirror, real_source=real_source, mirror_source=mirror_source)
    else:
        I_0 = __zeroth_order_finite_line_source_approximation(u, L, r, s, r_mirror, s_mirror, deg=deg, real_source=real_source, mirror_source=mirror_source)
    return I_0


def __zeroth_order_finite_line_source_approximation(u, L, r, s, r_mirror, s_mirror, deg=11, real_source=True, mirror_source=True):
    """Zeroth order finite line source solution using approximation."""
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
    I_0 = 0.25 * np.sign(etas) * (
        exp1(
            np.multiply.outer(
                u**2,
                np.expand_dims(sigmas**2, axis=-1) + np.multiply.outer(etas**2, _b_erf)
                )
        ) @ _a_erf
    ) @ q
    return I_0


def __zeroth_order_finite_line_source_integral(u, L, r, s, r_mirror, s_mirror, real_source=True, mirror_source=True):
    """Zeroth order finite line source solution using numerical integration."""
    if real_source and mirror_source:
        def integrand(u):
            X_0 = _zeroth_order_inner_integral(u, L, s)
            X_0_mirror = _zeroth_order_inner_integral(u, L, s_mirror)
            return np.exp(-(u * r)**2) * X_0 - np.exp(-(u * r_mirror)**2) * X_0_mirror
    elif real_source:
        def integrand(u):
            X_0 = _zeroth_order_inner_integral(u, L, s)
            return np.exp(-(u * r)**2) * X_0
    elif mirror_source:
        def integrand(u):
            X_0_mirror = _zeroth_order_inner_integral(u, L, s_mirror)
            return -np.exp(-(u * r_mirror)**2) * X_0_mirror
    else:
        shape = np.shape(u) + np.shape(s)
        return np.zeros(shape)
    # Lower bound of integration
    a = u
    # Upper bound of integration
    b = np.concatenate(([np.inf], a[:-1]))
    I_0 = np.cumsum(np.stack(
        [quad_vec(integrand, a_i, b_i, epsabs=1e-4, epsrel=1e-6)[0]
            for a_i, b_i in zip(a, b)],
        axis=0), axis=0)
    return I_0
