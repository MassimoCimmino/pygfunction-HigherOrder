# -*- coding: utf-8 -*-
from time import perf_counter
from typing import Tuple
import warnings

import numpy as np
import numpy.typing as npt

from ...basis import Basis
from ...borehole import Borehole
from ...borefield import Borefield
from ...heat_transfer.finite_line_source import finite_line_source
from ._similarities_vertical_to_vertical import _similarities_vertical_to_vertical


def _vertical_to_vertical(
    time: npt.ArrayLike,
    alpha: float,
    target: Borefield,
    source: Borefield,
    basis: Basis,
    r_min: float | np.ndarray | None = None,
    r_min_indices: Tuple[npt.ArrayLike, npt.ArrayLike] | None = None,
    rTol: float = 1e-2,
    sTol: float = 1e-2,
    fls_backend: str = 'recurrence',
    deg: int | None = None,
    real_source: bool = True,
    mirror_source: bool = True,
    assume_vertical: bool = False,
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
    target : Borefield or Borehole
        Target borefield or borehole.
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
        the indices of the target boreholes and the second array contains the
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
    assume_vertical : bool, optional
        Whether to assume that all source and target boreholes are vertical. Default is False.
    assume_identical : bool, optional
        Whether to assume that all source boreholes are identical, and that target boreholes
        are identical. Default is False.

    Returns
    -------
    h : ndarray
        Thermal response factors with shape (n_time, n_target_boreholes, n_basis_nodes, n_source_boreholes, n_basis_nodes).

    """

    # Check verticality
    if not assume_vertical:
        assert source.all_vertical, "Source borefield must consist of vertical boreholes."
        assert target.all_vertical, "Target borefield must consist of vertical boreholes."

    # Check identicality
    if not assume_identical:
        assert np.allclose(source.L, source.L[0]), "Source borefield must consist of boreholes with identical lengths."
        assert np.allclose(source.D, source.D[0]), "Source borefield must consist of boreholes with identical burial depths."
        assert np.allclose(target.L, target.L[0]), "Target borefield must consist of boreholes with identical lengths."
        assert np.allclose(target.D, target.D[0]), "Target borefield must consist of boreholes with identical burial depths."

    # Convert time to numpy array
    time = np.asarray(time)

    # Evaluate similarities
    p, reference_source, indices = _similarities_vertical_to_vertical(
        target=target,
        source=source,
        basis=basis,
        r_min=r_min,
        r_min_indices=r_min_indices,
        rTol=rTol,
        sTol=sTol,
        assume_vertical=True,
        assume_identical=True
    )

    # Evaluate finite line source solution
    # TODO : (r, s) is evaluated in the _similarities_vertical_to_vertical function,
    #        so it would be more efficient to return them and pass them to the finite line
    #        source solution instead of evaluating them again in the finite line source solution.
    I = finite_line_source(time, alpha, p, reference_source, basis.n_nodes, r_min=0., backend=fls_backend, deg=deg, real_source=real_source, mirror_source=mirror_source)
    # Evaluate thermal response factors
    h = I @ basis._psi_coefs
    # Broadcast to the full target and source borefields
    h = h[..., indices, :].reshape((-1, target.n_boreholes, basis.n_nodes, source.n_boreholes, basis.n_nodes))

    return h
