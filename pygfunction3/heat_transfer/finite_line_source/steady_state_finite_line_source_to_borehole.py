# -*- coding: utf-8 -*-
from typing import Tuple
import warnings

import numpy as np
import numpy.typing as npt

from .steady_state_finite_line_source import steady_state_finite_line_source
from ...basis import Basis
from ...borehole import Borehole
from ...borefield import Borefield


def steady_state_finite_line_source_to_borehole(
        target: Borefield | Borehole,
        source: Borefield | Borehole,
        target_basis: Basis,
        n: int,
        r_min: npt.ArrayLike | float | None = None,
        r_min_indices: Tuple[npt.ArrayLike, npt.ArrayLike] | npt.ArrayLike | None = None,
        backend: str = 'recurrence',
        real_source: bool = True,
        mirror_source: bool = True
        ) -> np.ndarray:
    """
    Steady-state finite line source solution to nodes on a borehole up to order n.

    Parameters
    ----------
    target : Borefield or Borehole
        Target borefield or borehole.
    source : Borefield or Borehole
        Source borefield or borehole.
    target_basis : Basis
        Polynomial basis functions for the target borefield or borehole.
    n : int
        Order of the solution.
    r_min : float or array_like, optional
        Minimum radial distance for the finite line source solution.
        If a single float is provided, it is applied to all source boreholes.
        If an array is provided, it must have the same length as the number of
        source boreholes and each value is applied to the corresponding borehole.
        If None, no minimum radius is applied. Default is None.
    r_min_indices : tuple of array_like or array_like, optional
        Indices for applying the minimum radial distance. If the source is a Borefield,
        this can be a tuple of two array_like, where the first array contains
        the indices of the target points and the second array contains the
        corresponding indices of the source boreholes. If the source is a Borehole,
        this can be a single array_like containing the indices of the target points
        for which the minimum radial distance should be applied. If None, the minimum radial
        distance is applied to all target-source pairs. Default is None.
    backend : str, optional
        Backend for the finite line source solution. Must be either 'recurrence' or 'integral'. Default is 'recurrence'.
    real_source : bool, optional
        Whether to include the contribution from the real source. Default is True.
    mirror_source : bool, optional
        Whether to include the contribution from the mirror source. Default is True.

    Returns
    -------
    I : ndarray
        Steady-state finite line source solution up to order n the nodes of the target borehole.

    """
    p_target = target.position(target_basis.xi)
    if isinstance(target, Borefield):
        p_target = p_target.swapaxes(0, 1)
    I = steady_state_finite_line_source(p_target, source, n, r_min=r_min, r_min_indices=r_min_indices, backend=backend, real_source=real_source, mirror_source=mirror_source)
    return I
