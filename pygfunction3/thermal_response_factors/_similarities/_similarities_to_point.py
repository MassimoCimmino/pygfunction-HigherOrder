# -*- coding: utf-8 -*-
from time import perf_counter
from typing import Tuple
import warnings

import numpy as np
import numpy.typing as npt

from ...basis import Basis
from ...borehole import Borehole
from ...borefield import Borefield
from ...utils.mirror_point import mirror_point

def _similarities_to_point(
    p_target: npt.ArrayLike,
    source: Borefield,
    basis: Basis,
    r_min: float | None = None,
    r_min_indices: Tuple[npt.ArrayLike, npt.ArrayLike] | None = None,
    rTol: float = 1e-2,
    sTol: float = 1e-2,
    aTol: float = 1e-8,
    which: str = 'real',
    assume_identical: bool = False
    ):
    """
    Similarities for thermal response factors between a borefield and points.

    Parameters
    ----------
    p_target : array_like
        Target points.
    source : Borefield
        Source borefield.
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
    aTol : float, optional
        Absolute tolerance for grouping target-source pairs. Default is 1e-8.
    which : str, optional
        Whether to use the real source or the mirror source for the evaluation of the
        positions. Must be either 'real' or 'mirror'. Default is 'real'.
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

    # Check identicality
    if not assume_identical:
        assert np.allclose(source.L, source.L[0]), "Source borefield must consist of boreholes with identical lengths."
        assert np.allclose(source.D, source.D[0]), "Source borefield must consist of boreholes with identical burial depths."
        assert np.allclose(source.tilt, source.tilt[0]), "Source borefield must consist of boreholes with identical tilt angles."

    # Convert p_target to numpy array
    p_target = np.asarray(p_target)

    # Get reference source and target
    reference_source = Borehole(
        L=source.L[0],
        D=source.D[0],
        r_b=source.r_b[0],
        p_head=np.array([0., 0.]),
        tilt=source.tilt[0],
        orientation=0.
    )

    # Radial-axial positions of the points with respect to the source boreholes
    if which.lower() == 'real':
        r, s = source.radial_axial_position(p_target, r_min=r_min, r_min_indices=r_min_indices)
    elif which.lower() == 'mirror':
        p_target = mirror_point(p_target)
        # r_min is not applied to the mirror source
        r, s = source.radial_axial_position(p_target, r_min=None, r_min_indices=None)
    else:
        raise ValueError("Invalid value for 'which'. Must be either 'real' or 'mirror'.")
    r = r.flatten()
    s = s.flatten()

    # Lengths of the arrays
    n_targets = int(np.prod(p_target.shape[:-1]) if p_target.ndim > 1 else 1)
    n_sources = source.n_boreholes
    # Indices for mapping the finite line source solution to the full target points
    indices = np.zeros(n_targets * n_sources, dtype=int)

    # Sort the radial positions
    r_sort = np.argsort(r.flatten())
    x = r.flatten()[r_sort]
    s = s.flatten()[r_sort]

    # Cluster the radial positions and get the unique positions
    n_points = 0
    j0 = 0
    j1 = 1
    p = np.zeros((0, 3))
    while j0 < len(x) and j1 > 0:
        # Find next unique radial position
        j1 = np.argmax(
            x >= (1 + rTol) * x[j0] + aTol)
        if j1 <= j0: j1 = len(x)
        r_indices = r_sort[j0:j1]
        # Mean radial position for the current cluster
        r_mean = np.mean(x[j0:j1])

        # Sort the axial positions for the current radial cluster
        s_sort = np.argsort(s[j0:j1])
        y = s[j0:j1][s_sort] / (r_mean + aTol)

        # Find unique axial positions for the current radial position
        i0 = 0
        i1 = 1
        while i0 < len(y) and i1 > 0:
            i1 = np.argmax(
                y >= sTol + y[i0]
                )
            if i1 <= i0: i1 = len(y)
            s_indices = r_indices[s_sort[i0:i1]]
            # Mean axial position for the current cluster
            s_mean = np.mean(s[j0:j1][s_sort[i0:i1]])

            # Append the mean radial and axial positions to the list of points
            p = np.append(
                p,
                [
                    [
                        s_mean * np.sin(reference_source.tilt),
                        r_mean,
                        -reference_source.D - s_mean * np.cos(reference_source.tilt)]
                    ],
                    axis=0)
            indices[s_indices] = n_points
            n_points += 1
            i0 = i1
        j0 = j1

    if which.lower() == 'mirror':
        p = mirror_point(p)

    return p, reference_source, indices
