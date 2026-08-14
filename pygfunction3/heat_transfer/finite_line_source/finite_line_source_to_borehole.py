# -*- coding: utf-8 -*-
from typing import Tuple
import warnings

import numpy as np
import numpy.typing as npt
from .finite_line_source import finite_line_source
from ...basis import Basis
from ...borehole import Borehole
from ...borefield import Borefield


def finite_line_source_to_borehole(
        time: npt.ArrayLike,
        alpha: float,
        target: Borefield | Borehole,
        source: Borefield | Borehole,
        target_basis: Basis,
        n: int,
        r_min: npt.ArrayLike | float | None = None,
        r_min_indices: Tuple[npt.ArrayLike, npt.ArrayLike] | npt.ArrayLike | None = None,
        backend: str = 'recurrence',
        deg: int | None = None,
        real_source: bool = True,
        mirror_source: bool = True
        ) -> np.ndarray:
    """
    Finite line source solution to nodes on a borehole up to order n.

    Parameters
    ----------
    time : array_like
        Time(s) at which to evaluate the solution.
    alpha : float
        Thermal diffusivity of the ground.
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
    deg : int or None, optional
        Degree of the approximation for the zeroth-order finite line source solution.
        Only used if backend is 'recurrence'. Default is None, which means that no approximation is used.
    real_source : bool, optional
        Whether to include the contribution from the real source. Default is True.
    mirror_source : bool, optional
        Whether to include the contribution from the mirror source. Default is True.

    Returns
    -------
    I : ndarray
        Finite line source solution up to order n at the nodes of the target borehole and times.

    """
    p_target = target.position(target_basis.xi)
    if isinstance(target, Borefield):
        p_target = p_target.swapaxes(0, 1)
    I = finite_line_source(time, alpha, p_target, source, n, r_min=r_min, r_min_indices=r_min_indices, backend=backend, deg=deg, real_source=real_source, mirror_source=mirror_source)
    return I
