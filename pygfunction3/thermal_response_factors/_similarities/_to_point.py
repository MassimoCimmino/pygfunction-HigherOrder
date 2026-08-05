# -*- coding: utf-8 -*-
from time import perf_counter
from typing import Tuple
import warnings

import numpy.typing as npt

from ...basis import Basis
from ...borefield import Borefield
from ...heat_transfer.finite_line_source import finite_line_source
from ._similarities_to_point import _similarities_to_point

def _to_point(
    time: npt.ArrayLike,
    alpha: float,
    p: npt.ArrayLike,
    source: Borefield,
    basis: Basis,
    r_min: float | None = None,
    r_min_indices: Tuple[npt.ArrayLike, npt.ArrayLike] | None = None,
    fls_backend: str = 'recurrence',
    deg: int | None = None,
    real_source: bool = True,
    mirror_source: bool = True,
    rTol: float = 1e-2,
    sTol: float = 1e-2,
    assume_identical: bool = False
    ):
    """
    Thermal response factors between two vertical borefields using the similarities method.

    Parameters
    ----------
    time : array_like
        Time(s) at which to evaluate the response factors.
    alpha : float
        Thermal diffusivity of the ground.
    p : array_like
        Target points.
    source : Borefield or Borehole
        Source borefield or borehole.
    basis : Basis
        Polynomial basis for the response factors.
    r_min : float or array_like, optional
        Minimum radial distance for the finite line source solution.
        If a single float is provided, it is applied to all source boreholes.
        If an array is provided, it must have the same length as the number of
        source boreholes and each value is applied to the corresponding borehole.
        If None, no minimum radius is applied. Default is None.
    r_min_indices : tuple of array_like, optional
        Indices for applying the minimum radial distance. The first array contains
        the indices of the target points and the second array contains the
        corresponding indices of the source boreholes. If None, the minimum radial
        distance is applied to all target-source pairs. Default is None.
    rTol : float, optional
        Relative tolerance for grouping target-source pairs based on their radial
        distance. Default is 1e-2.
    sTol : float, optional
        Relative tolerance for grouping target-source pairs based on their axial
        distance. Default is 1e-2.
    fls_backend : str, optional
        Backend for the finite line source solution. Default is 'recurrence'.
    deg : int or None, optional
        Degree of the polynomial approximation for the finite line source solution.
        If None, the finite line source solution is computed without polynomial approximation.
        Default is None.
    real_source : bool, optional
        Whether to use the real source for the finite line source solution.
        Default is True.
    mirror_source : bool, optional
        Whether to use the mirror source for the finite line source solution.
        Default is True.
    assume_identical : bool, optional
        Whether to assume that all source boreholes are identical. Default is False.

    Returns
    -------
    h : ndarray
        Thermal response factors with shape (n_time, n_target_boreholes, n_basis_nodes, n_source_boreholes, n_basis_nodes).

    """

    # Similarities for the real source
    p_real, reference_source, indices = _similarities_to_point(
        p,
        source,
        basis,
        rTol=rTol,
        sTol=sTol,
        r_min=r_min,
        r_min_indices=r_min_indices,
        which='real',
        assume_identical=assume_identical
        )
    # Finite line source solution for the real source
    # TODO : (r, s) is evaluated in the _similarities_to_point function,
    #        so it would be more efficient to return them and pass them to the finite line
    #        source solution instead of evaluating them again in the finite line source solution.
    I = finite_line_source(
        time,
        alpha,
        p_real,
        reference_source,
        basis.n_nodes,
        r_min=None,
        backend=fls_backend,
        deg=deg,
        real_source=real_source,
        mirror_source=False
        )
    # Thermal response factors for the real source
    h_real = I @ basis._coefficients
    # Broadcast the thermal response factors for the real source to the full target points
    h = h_real[..., indices, :].reshape((-1, *p.shape[:-1], source.n_boreholes, basis.n_nodes))
    
    # Similarities for the mirror source
    p_mirror, reference_source, indices = _similarities_to_point(
        p,
        source,
        basis,
        rTol=rTol,
        sTol=sTol,
        r_min=r_min,
        r_min_indices=r_min_indices,
        which='mirror',
        assume_identical=assume_identical
    )
    # Finite line source solution for the mirror source
    I = finite_line_source(
        time,
        alpha,
        p_mirror,
        reference_source,
        basis.n_nodes,
        r_min=None,
        backend=fls_backend,
        deg=deg,
        real_source=False,
        mirror_source=mirror_source
    )
    # Thermal response factors for the mirror source
    h_mirror = I @ basis._coefficients
    # Broadcast the thermal response factors for the mirror source to the full target points and add to the total response factors
    h += h_mirror[..., indices, :].reshape((-1, *p.shape[:-1], source.n_boreholes, basis.n_nodes))

    return h
