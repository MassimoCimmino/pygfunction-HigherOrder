# -*- coding: utf-8 -*-
import warnings

import numpy as np
import numpy.typing as npt
from scipy.special import erfc

from .zeroth_order_finite_line_source import (
    _zeroth_order_finite_line_source,
)
from ...utils import mirror_point


def first_order_finite_line_source(
        time: npt.ArrayLike,
        alpha: float,
        p: npt.ArrayLike,
        borehole,
        r_min: float = 0.,
        deg: int | None = None,
        real_source: bool = True,
        mirror_source: bool = True
        ) -> np.ndarray:
    """First order finite line source solution."""
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

    # Thermal response factor
    I_1 = _first_order_finite_line_source(u, borehole.L, r, s, r_mirror, s_mirror, deg=deg, real_source=real_source, mirror_source=mirror_source)

    return I_1


def _first_order_finite_line_source(u, L, r, s, r_mirror, s_mirror, J_0_real=None, J_0_mirror=None, deg=None, real_source=True, mirror_source=True):
    """First order finite line source solution."""
    shape = np.shape(u) + np.shape(s)
    if real_source:
        if J_0_real is None:
            J_0_real = _zeroth_order_finite_line_source(u, L, r, s, r_mirror, s_mirror, deg=deg, mirror_source=False)
        J_1_real = _first_order_integral(u, L, r, s)
        I_1_real = (2 * s / L - 1) * J_0_real + J_1_real
    else:
        # TODO: return zeros of correct shape
        I_1_real = np.zeros(shape)

    if mirror_source:
        if J_0_mirror is None:
            J_0_mirror = _zeroth_order_finite_line_source(u, L, r, s, r_mirror, s_mirror, deg=deg, real_source=False)
        J_1_mirror = -_first_order_integral(u, L, r_mirror, s_mirror)
        I_1_mirror = (2 * s_mirror / L - 1) * J_0_mirror + J_1_mirror
    else:
        # TODO: return zeros of correct shape
        I_1_mirror = np.zeros(shape)

    return I_1_real + I_1_mirror


def _first_order_integral(
        u: npt.ArrayLike,
        L: float,
        r: npt.ArrayLike,
        s: npt.ArrayLike):
    """First order integral."""
    # shape = np.shape(u) + tuple(np.ones(np.ndim(s), dtype=int))
    if np.ndim(r) == 1 or np.ndim(s) == 1:
        u = np.expand_dims(u, axis=-1)
    # u_reshaped = np.reshape(u, shape)
    # Integral of exp(-u**2 * r**2) * X_1(u, L, s) du
    arg1 = (L - s)**2 + r**2
    sqrt_arg1 = np.sqrt(arg1)
    arg2 = s**2 + r**2
    sqrt_arg2 = np.sqrt(arg2)
    term1 = np.exp(-u**2 * arg1) / (u * np.sqrt(np.pi))
    term2 = np.exp(-u**2 * arg2) / (u * np.sqrt(np.pi))
    term3 = sqrt_arg1 * erfc(u * sqrt_arg1)
    term4 = sqrt_arg2 * erfc(u * sqrt_arg2)
    J_1 = (term2 - term1 + term3 - term4) / L

    return J_1
