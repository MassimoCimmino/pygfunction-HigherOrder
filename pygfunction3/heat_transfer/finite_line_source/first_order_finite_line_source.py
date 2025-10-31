# -*- coding: utf-8 -*-
import warnings

import numpy as np
import numpy.typing as npt
from scipy.special import erfc

from .zeroth_order_finite_line_source import (
    _zeroth_order_finite_line_source,
)
from ...borehole import Borehole
from ...borefield import Borefield
from ...utils import mirror_point


def first_order_finite_line_source(
        time: npt.ArrayLike,
        alpha: float,
        p: npt.ArrayLike,
        source: Borefield | Borehole,
        r_min: float = 0.,
        deg: int | None = None,
        real_source: bool = True,
        mirror_source: bool = True
        ) -> np.ndarray:
    """First order finite line source solution."""
    if isinstance(time, list):
        time = np.asarray(time)
    # Lower bound of integration
    u = 1 / np.sqrt(4 * alpha * time)

    # Real points
    if real_source:
        r, s = source.radial_axial_position(p, r_min=r_min)
    else:
        r = np.empty(p.shape[:-1])
        s = np.empty(p.shape[:-1])
    # Mirror points
    if mirror_source:
        p_mirror = mirror_point(p)
        r_mirror, s_mirror = source.radial_axial_position(p_mirror, r_min=r_min)
    else:
        r_mirror = np.empty(p.shape[:-1])
        s_mirror = np.empty(p.shape[:-1])

    # Reshape inputs for broadcasting
    shape_u = np.shape(u)
    shape_s = np.shape(s)
    u = np.reshape(u, shape_u + (1,) * len(shape_s))

    # Thermal response factor
    I_1 = _first_order_finite_line_source(u, L, r, s, r_mirror, s_mirror, deg=deg, real_source=real_source, mirror_source=mirror_source)

    return I_1


def _first_order_finite_line_source(
        u: np.ndarray,
        L:  float | np.ndarray,
        r: np.ndarray,
        s: np.ndarray,
        r_mirror: np.ndarray,
        s_mirror: np.ndarray,
        J_0_real: np.ndarray | None = None,
        J_0_mirror: np.ndarray | None = None,
        deg: int | None = None,
        real_source: bool = True,
        mirror_source: bool = True
        ) -> np.ndarray:
    """First order finite line source solution."""
    shape = np.broadcast_shapes(np.shape(u), np.shape(s))
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
        u: np.ndarray,
        L: float | np.ndarray,
        r: np.ndarray,
        s: np.ndarray) -> np.ndarray:
    """First order integral."""
    arg1 = (L - s)**2 + r**2
    sqrt_arg1 = np.sqrt(arg1)
    arg2 = s**2 + r**2
    sqrt_arg2 = np.sqrt(arg2)
    uu = u * u
    term1 = np.exp(-uu * arg1) / (u * np.sqrt(np.pi))
    term2 = np.exp(-uu * arg2) / (u * np.sqrt(np.pi))
    term3 = sqrt_arg1 * erfc(u * sqrt_arg1)
    term4 = sqrt_arg2 * erfc(u * sqrt_arg2)
    J_1 = (term2 - term1 + term3 - term4) / L

    return J_1
