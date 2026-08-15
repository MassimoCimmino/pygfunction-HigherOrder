# -*- coding: utf-8 -*-
from typing import Tuple
import warnings

import numpy as np
import numpy.typing as npt

from .zeroth_order_steady_state_finite_line_source import (
    _zeroth_order_steady_state_finite_line_source,
)
from ...borehole import Borehole
from ...borefield import Borefield
from ...utils import mirror_point


def first_order_steady_state_finite_line_source(
        p: npt.ArrayLike,
        source: Borefield | Borehole,
        r_min: npt.ArrayLike | float | None = None,
        r_min_indices: Tuple[npt.ArrayLike, npt.ArrayLike] | npt.ArrayLike | None = None,
        real_source: bool = True,
        mirror_source: bool = True
        ) -> np.ndarray:
    """First order steady-state finite line source solution."""

    # Real points
    if real_source:
        r, s = source.radial_axial_position(p, r_min=r_min, r_min_indices=r_min_indices)
    else:
        r = np.empty(p.shape[:-1])
        s = np.empty(p.shape[:-1])
    # Mirror points
    if mirror_source:
        p_mirror = mirror_point(p)
        r_mirror, s_mirror = source.radial_axial_position(p_mirror, r_min=r_min, r_min_indices=r_min_indices)
    else:
        r_mirror = np.empty(p.shape[:-1])
        s_mirror = np.empty(p.shape[:-1])

    # Thermal response factor
    I_1 = _first_order_steady_state_finite_line_source(source.L, r, s, r_mirror, s_mirror, real_source=real_source, mirror_source=mirror_source)

    return I_1


def _first_order_steady_state_finite_line_source(
        L:  float | np.ndarray,
        r: np.ndarray,
        s: np.ndarray,
        r_mirror: np.ndarray,
        s_mirror: np.ndarray,
        J_0_real: np.ndarray | None = None,
        J_0_mirror: np.ndarray | None = None,
        real_source: bool = True,
        mirror_source: bool = True
        ) -> np.ndarray:
    """First order steady-state finite line source solution."""
    shape = np.shape(s)
    if real_source:
        if J_0_real is None:
            J_0_real = _zeroth_order_steady_state_finite_line_source(L, r, s, r_mirror, s_mirror, mirror_source=False)
        J_1_real = _first_order_steady_state_integral(L, r, s)
        I_1_real = (2 * s / L - 1) * J_0_real + J_1_real
    else:
        # TODO: return zeros of correct shape
        I_1_real = np.zeros(shape)

    if mirror_source:
        if J_0_mirror is None:
            J_0_mirror = _zeroth_order_steady_state_finite_line_source(L, r, s, r_mirror, s_mirror, real_source=False)
        J_1_mirror = -_first_order_steady_state_integral(L, r_mirror, s_mirror)
        I_1_mirror = (2 * s_mirror / L - 1) * J_0_mirror + J_1_mirror
    else:
        # TODO: return zeros of correct shape
        I_1_mirror = np.zeros(shape)

    return I_1_real + I_1_mirror


def _first_order_steady_state_integral(
        L: float | np.ndarray,
        r: np.ndarray,
        s: np.ndarray) -> np.ndarray:
    """First order integral at steady-state."""
    arg1 = (L - s)**2 + r**2
    sqrt_arg1 = np.sqrt(arg1)
    arg2 = s**2 + r**2
    sqrt_arg2 = np.sqrt(arg2)
    J_1 = (sqrt_arg1 - sqrt_arg2) / L

    return J_1
