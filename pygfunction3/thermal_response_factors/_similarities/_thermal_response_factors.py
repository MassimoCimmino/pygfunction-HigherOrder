# -*- coding: utf-8 -*-
from typing import Tuple
import warnings

import numpy as np
import numpy.typing as npt

from ...basis import Basis
from ...borehole import Borehole
from ...borefield import Borefield
from ._to_point import _to_point
from ._vertical_to_vertical import _vertical_to_vertical


def _thermal_response_factors_to_borehole(
        time: npt.ArrayLike,
        alpha: float,
        target: Borefield | Borehole,
        source: Borefield | Borehole,
        basis: Basis,
        r_min: float | npt.ArrayLike | None = None,
        r_min_indices: Tuple[npt.ArrayLike, npt.ArrayLike] | None = None,
        rTol: float = 1e-2,
        sTol: float = 1e-2,
        fls_backend: str = 'recurrence',
        deg: int | None = None,
        real_source: bool = True,
        mirror_source: bool = True,
        assume_identical_source: bool = False,
        assume_identical_target: bool = False,
        **kwargs
        ) -> np.ndarray:
    """
    Thermal response factors.

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
    assume_identical_source : bool, optional
        Whether to assume that all source boreholes are identical. Default is False.
    assume_identical_target : bool, optional
        Whether to assume that all target boreholes are identical. Default is False.

    Returns
    -------
    h : ndarray
        Thermal response factors with shape (n_time, n_target_boreholes, n_basis_nodes, n_source_boreholes, n_basis_nodes).

    """
    # Convert time to numpy array
    scalar_time = np.isscalar(time)
    time = np.asarray(time)

    # Convert scalar boreholes to single-borehole fields for uniform processing
    scalar_source = isinstance(source, Borehole)
    if scalar_source:
        source = Borefield(
            L=np.array([source.L]),
            D=np.array([source.D]),
            r_b=np.array([source.r_b]),
            p_head=np.array([source.p_head])
         )
    scalar_target = isinstance(target, Borehole)
    if scalar_target:
        target = Borefield(
            L=np.array([target.L]),
            D=np.array([target.D]),
            r_b=np.array([target.r_b]),
            p_head=np.array([target.p_head])
         )

    # Broadcast r_min if it is a single value
    if r_min is not None:
        r_min = np.broadcast_to(r_min, (source.n_boreholes,))

    # If the source borefield consists of non-identical boreholes, split it into subfields
    # of identical boreholes and compute the response factors for each subfield separately
    if not assume_identical_source and not source.all_identical:

        # Initialize response factors array
        h = np.zeros((time.size, target.n_boreholes, basis.n_nodes, source.n_boreholes, basis.n_nodes))

        # Initialize remaining source indices and subfield
        remaining_source_indices = np.arange(source.n_boreholes, dtype=int)
        remaining_source = source
        remaining_r_min = r_min
        remaining_r_min_indices = r_min_indices

        # Loop until all source boreholes have been processed
        while remaining_source_indices.size > 0:

            # Find which boreholes are identical to the first remaining borehole
            i = remaining_source_indices[0]
            reference_source = Borehole(
                L=source.L[i],
                D=source.D[i],
                r_b=source.r_b[i],
                p_head=source.p_head[i],
                tilt=source.tilt[i],
                orientation=source.orientation[i]
            )
            is_identical = reference_source.is_identical(remaining_source)

            # Create source subfield of identical boreholes
            source_subfield_indices = remaining_source_indices[is_identical]
            source_subfield = Borefield(
                L=source.L[source_subfield_indices],
                D=source.D[source_subfield_indices],
                r_b=source.r_b[source_subfield_indices],
                p_head=source.p_head[source_subfield_indices],
                tilt=source.tilt[source_subfield_indices],
                orientation=source.orientation[source_subfield_indices]
            )
            # Get r_min for the source subfield and corresponding local
            # subfield indices if r_min_indices is provided
            r_min_subfield = remaining_r_min[is_identical] if remaining_r_min is not None else None
            if r_min_indices is not None:
                # Get the r_min values for the source subfield
                # Get the indices of the source subfield boreholes in the original r_min_indices
                mask = np.isin(remaining_r_min_indices[1], source_subfield_indices)
                r_min_subfield_indices =  (
                    remaining_r_min_indices[0][mask],
                    remaining_r_min_indices[1][mask]
                )
                # Map the source subfield indices to local indices for the subfield
                local_subfield_indices = np.arange(len(source_subfield_indices))
                r_min_subfield_indices[1][:] = np.fromiter(
                    (dict(zip(source_subfield_indices, local_subfield_indices)).get(x, x) for x in r_min_subfield_indices[1]),
                    dtype=r_min_subfield_indices[1].dtype,
                    count=r_min_subfield_indices[1].size
                    )
            else:
                r_min_subfield_indices = None

            # Update remaining source indices and subfield for the next iteration
            remaining_source_indices = remaining_source_indices[np.logical_not(is_identical)]
            remaining_source = Borefield(
                L=source.L[remaining_source_indices],
                D=source.D[remaining_source_indices],
                r_b=source.r_b[remaining_source_indices],
                p_head=source.p_head[remaining_source_indices],
                tilt=source.tilt[remaining_source_indices],
                orientation=source.orientation[remaining_source_indices],
            )
            remaining_r_min = remaining_r_min[np.logical_not(is_identical)] if remaining_r_min is not None else None
            remaining_r_min_indices = (
                remaining_r_min_indices[0][np.logical_not(mask)],
                remaining_r_min_indices[1][np.logical_not(mask)]
                ) if remaining_r_min_indices is not None else None

            # Evaluate response factors for the current source subfield and assign
            # to the corresponding indices in the full response factors array
            h[:, :, :, source_subfield_indices, :] = _thermal_response_factors_to_borehole(
                time=time,
                alpha=alpha,
                target=target,
                source=source_subfield,
                basis=basis,
                r_min=r_min_subfield,
                r_min_indices=r_min_subfield_indices,
                rTol=rTol,
                sTol=sTol,
                fls_backend=fls_backend,
                deg=deg,
                real_source=real_source,
                mirror_source=mirror_source,
                assume_identical_source=True,
                assume_identical_target=assume_identical_target
            )
    else:
        # Check verticality and identicality of the source and target borefields and call the appropriate similarity function
        if source.all_vertical and target.all_vertical and (assume_identical_source or source.all_identical) and (assume_identical_target or target.all_identical):
            h = _vertical_to_vertical(
                time,
                alpha,
                target,
                source,
                basis,
                r_min=r_min,
                r_min_indices=r_min_indices,
                rTol=rTol,
                sTol=sTol,
                fls_backend=fls_backend,
                deg=deg,
                real_source=real_source,
                mirror_source=mirror_source,
                assume_vertical=True)
        else:
            p = target.position(basis.xi).swapaxes(0, 1)
            h = _to_point(
                time,
                alpha,
                p,
                source,
                basis,
                r_min=r_min,
                r_min_indices=r_min_indices,
                rTol=rTol,
                sTol=sTol,
                fls_backend=fls_backend,
                deg=deg,
                real_source=real_source,
                mirror_source=mirror_source
                ).reshape((-1, target.n_boreholes, basis.n_nodes, source.n_boreholes, basis.n_nodes))

    # Flatten dimensions if the source or target was scalar
    if scalar_source and scalar_target:
        h = h[:, 0, :, 0, :]
    elif scalar_source:
        h = h[:, :, :, 0, :]
    elif scalar_target:
        h = h[:, 0, :, :, :]
    # Flatten time dimension if the original input was scalar
    if scalar_time:
        h = h[0, ...]

    return h
