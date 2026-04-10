# -*- coding: utf-8 -*-
from time import perf_counter
from typing import Tuple
import warnings

import numpy as np
import numpy.typing as npt

from ...basis import Basis
from ...borehole import Borehole
from ...borefield import Borefield

def _similarities_vertical_to_vertical(
    target: Borefield,
    source: Borefield,
    basis: Basis,
    r_min: float | np.ndarray | None = None,
    r_min_indices: Tuple[npt.ArrayLike, npt.ArrayLike] | None = None,
    rTol: float = 1e-2,
    sTol: float = 1e-2,
    aTol: float = 1e-8,
    which: str = 'real',
    assume_vertical: bool = False,
    assume_identical: bool = False
    ):
    """
    Similarities for thermal response factors between two vertical borefields.

    Parameters
    ----------
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
    aTol : float, optional
        Absolute tolerance for grouping target-source pairs. Default is 1e-8.
    which : str, optional
        Whether to use the real source or the mirror source for the evaluation of the
        positions. Must be either 'real' or 'mirror'. Default is 'real'.
    assume_vertical : bool, optional
        Whether to assume that all source and target boreholes are vertical. Default is False.
    assume_identical : bool, optional
        Whether to assume that all source boreholes are identical, and that target boreholes
        are identical. Default is False.

    Returns
    -------
    p : ndarray
        Radial-axial positions of the points for the finite line source solution.
    reference_source : Borehole
        Reference source borehole for the finite line source solution.
    indices : ndarray
        Indices for mapping the finite line source solution to the full target and source borefields.

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

    # Get reference source and target
    reference_source = Borehole(
        L=source.L[0],
        D=source.D[0],
        r_b=source.r_b[0],
        p_head=np.array([0., 0.])
    )
    reference_target = Borehole(
        L=target.L[0],
        D=target.D[0],
        r_b=target.r_b[0],
        p_head=np.array([0., 0.])
    )

    # Mid-length radial positions of the target boreholes
    if which.lower() == 'real':
        r = source.radial_position(target.p_middle, r_min=r_min, r_min_indices=r_min_indices)
    elif which.lower() == 'mirror':
        # r_min is not applied to the mirror source
        r = source.radial_position(target.p_middle, r_min=None, r_min_indices=None)
    else:
        raise ValueError("Invalid value for 'which'. Must be either 'real' or 'mirror'.")
    # Axial positions of the target boreholes with respect to the reference source borehole
    s = reference_source.axial_position(reference_target.position(basis.xi))

    # Lengths of the arrays
    n_sources = source.n_boreholes
    n_targets = target.n_boreholes
    n_nodes = basis.n_nodes
    # Indices for mapping the finite line source solution to the full target and source borefields
    indices = np.zeros((n_targets * n_sources, n_nodes), dtype=int)

    # Sort the radial positions
    r_sort = np.argsort(r.flatten())
    x = r.flatten()[r_sort]

    # Cluster the radial positions and get the unique positions
    n_points = 0
    j0 = 0
    j1 = 1
    p = np.zeros((0, 3))
    while j0 < len(x) and j1 > 0:
        # Find next unique radial position
        # aTol is added to avoid issues with very small (or zero) radial positions
        j1 = np.argmax(
            x >= (1 + rTol) * x[j0] + aTol)
        if j1 <= j0: j1 = len(x)
        # Mean radial position for the current cluster
        r_mean = np.mean(x[j0:j1])

        # Find unique axial positions for the current radial position
        y = s / (r_mean + aTol)
        i0 = 0
        i1 = 1
        while i0 < len(y) and i1 > 0:
            # Find next unique axial position
            i1 = np.argmax(
                y >= sTol + y[i0]
                )
            if i1 <= i0: i1 = len(y)
            # Mean axial position for the current cluster
            s_mean = np.mean(s[i0:i1])

            # Append the mean radial and axial positions to the list of points
            p = np.append(p, [[r_mean, 0., -reference_source.D - s_mean]], axis=0)
            indices[r_sort[j0:j1], i0:i1] = n_points
            n_points += 1
            i0 = i1
        j0 = j1

    # Reshape indices to map the finite line source solution to the full target and source borefields
    indices = indices.reshape((n_targets, n_sources, n_nodes)).swapaxes(1, 2).flatten()

    return p, reference_source, indices
