# -*- coding: utf-8 -*-
from collections.abc import Callable
import warnings

import numpy as np
import numpy.typing as npt
from scipy.special import erfc

from .zeroth_order_finite_line_source import (
    X_0,
    _zeroth_order_finite_line_source,
)
from ..utils import mirror_point


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

    # Thermal response factor
    h = _first_order_finite_line_source(u, borehole.L, r, s, r_mirror, s_mirror, deg=deg, real_source=real_source, mirror_source=mirror_source)

    return h


def X_1(u, L, s, X_0_val=None):
    if X_0_val is None:
        X_0_val = X_0(u, L, s)
    Y_1_val = Y_1(u, L, s)
    result = (2 * s / L - 1) * X_0_val + Y_1_val
    return result


def Y_1(u, L, s):
    term1 = np.exp(-(u * (L - s))**2)
    term2 = np.exp(-(u * s)**2)
    result = 1 / (u**2 * L * np.sqrt(np.pi)) * (term2 - term1)
    return result


def _first_order_finite_line_source(u, L, r, s, r_mirror, s_mirror, h_0_real=None, h_0_mirror=None, deg=None, real_source=True, mirror_source=True):
    shape = np.shape(u) + tuple(np.ones(np.ndim(s), dtype=int))
    u_reshaped = np.reshape(u, shape)
    if real_source:
        if h_0_real is None:
            h_0_real = _zeroth_order_finite_line_source(u, L, r, s, r_mirror, s_mirror, deg=deg, mirror_source=False)
        # Real points
        arg1 = (L - s)**2 + r**2
        sqrt_arg1 = np.sqrt(arg1)
        arg2 = s**2 + r**2
        sqrt_arg2 = np.sqrt(arg2)
        # Integral of exp(-u**2 * r**2) * Y_1(u, L, s) du
        term1_real = np.exp(-u_reshaped**2 * arg1) / (u_reshaped * np.sqrt(np.pi))
        term2_real = np.exp(-u_reshaped**2 * arg2) / (u_reshaped * np.sqrt(np.pi))
        term3_real = sqrt_arg1 * erfc(u_reshaped * sqrt_arg1)
        term4_real = sqrt_arg2 * erfc(u_reshaped * sqrt_arg2)
        h_1_real = (term2_real - term1_real + term3_real - term4_real) / L + (2 * s / L - 1) * h_0_real
    else:
        # TODO: return zeros of correct shape
        h_1_real = np.zeros(shape)

    if mirror_source:
        if h_0_mirror is None:
            h_0_mirror = _zeroth_order_finite_line_source(u, L, r, s, r_mirror, s_mirror, deg=deg, real_source=False)
        # Mirror points
        arg1_mirror = (L - s_mirror)**2 + r_mirror**2
        sqrt_arg1_mirror = np.sqrt(arg1_mirror)
        arg2_mirror = s_mirror**2 + r_mirror**2
        sqrt_arg2_mirror = np.sqrt(arg2_mirror)
        # Integral of exp(-u**2 * r_mirror**2) * Y_1(u, L, s_mirror) du
        term1_mirror = np.exp(-u_reshaped**2 * arg1_mirror) / (u_reshaped * np.sqrt(np.pi))
        term2_mirror = np.exp(-u_reshaped**2 * arg2_mirror) / (u_reshaped * np.sqrt(np.pi))
        term3_mirror = sqrt_arg1_mirror * erfc(u_reshaped * sqrt_arg1_mirror)
        term4_mirror = sqrt_arg2_mirror * erfc(u_reshaped * sqrt_arg2_mirror)
        h_1_mirror = -(term2_mirror - term1_mirror + term3_mirror - term4_mirror) / L + (2 * s_mirror / L - 1) * h_0_mirror
    else:
        # TODO: return zeros of correct shape
        h_1_mirror = np.zeros(shape)

    return h_1_real + h_1_mirror
    # return (2 * s_proj / L - 1) * h_0_real + (2 * s_proj_mirror / L - 1) * h_0_mirror + h_1_real - h_1_mirror
