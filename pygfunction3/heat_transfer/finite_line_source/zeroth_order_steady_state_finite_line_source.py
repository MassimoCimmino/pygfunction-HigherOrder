# -*- coding: utf-8 -*-
from typing import Tuple
import warnings

import numpy as np
import numpy.typing as npt

from ...borehole import Borehole
from ...borefield import Borefield
from ...utils import mirror_point


def zeroth_order_steady_state_finite_line_source(
        p: npt.ArrayLike,
        source: Borefield | Borehole,
        r_min: npt.ArrayLike | float | None = None,
        r_min_indices: Tuple[npt.ArrayLike, npt.ArrayLike] | npt.ArrayLike | None = None,
        real_source: bool = True,
        mirror_source: bool = True
        ) -> np.ndarray:
    """Zeroth order steady-state finite line source solution."""
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

    I_0 = _zeroth_order_steady_state_finite_line_source(source.L, r, s, r_mirror, s_mirror, real_source=real_source, mirror_source=mirror_source)

    return I_0


def _zeroth_order_steady_state_finite_line_source(
        L: float | np.ndarray,
        r: np.ndarray,
        s: np.ndarray,
        r_mirror: np.ndarray,
        s_mirror: np.ndarray,
        real_source: bool = True,
        mirror_source: bool = True
        ) -> np.ndarray:
    """Zeroth order steady-state finite line source solution."""
    # All points
    if real_source and mirror_source:
        etas = np.stack((L - s, -s, L - s_mirror, -s_mirror), axis=-1)
        sigmas = np.stack((r, r, r_mirror, r_mirror), axis=-1)
        q = np.array([1., -1., -1., 1.])
    elif real_source:
        etas = np.stack((L - s, -s), axis=-1)
        sigmas = np.stack((r, r), axis=-1)
        q = np.array([1., -1.])
    elif mirror_source:
        etas = np.stack((L - s_mirror, -s_mirror), axis=-1)
        sigmas = np.stack((r_mirror, r_mirror), axis=-1)
        q = np.array([-1., 1.])
    else:
        shape = np.shape(s)
        return np.zeros(shape)
    I_0 = 0.5 * np.arcsinh(etas / sigmas) @ q
    return I_0
