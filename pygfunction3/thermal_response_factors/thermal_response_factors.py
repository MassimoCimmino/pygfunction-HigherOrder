# -*- coding: utf-8 -*-
from time import perf_counter
from typing import Tuple
import warnings

import numpy as np
import numpy.typing as npt

from ..heat_transfer.finite_line_source import finite_line_source
from ..basis import Basis
from ..borehole import Borehole
from ..borefield import Borefield
from . import _similarities


def thermal_response_factors_to_point(
        time: npt.ArrayLike,
        alpha: float,
        p: npt.ArrayLike,
        source: Borefield | Borehole,
        basis: Basis,
        r_min: float | npt.ArrayLike | None = None,
        r_min_indices: Tuple[npt.ArrayLike, npt.ArrayLike] | None = None,
        backend: str = 'similarities',
        fls_backend: str = 'recurrence',
        deg: int | None = None,
        real_source: bool = True,
        mirror_source: bool = True
        ) -> np.ndarray:
    """
    Calculate thermal response factors to a point.

    Parameters
    ----------
    time : array-like
        Time steps at which to evaluate the thermal response factors.
    alpha : float
        Thermal diffusivity of the ground.
    p : array-like
        Points at which to evaluate the thermal response factors.
    source : Borefield or Borehole
        Source for the thermal response factors.
    basis : Basis
        Basis for the thermal response factors.
    r_min : float or array-like, optional
        Minimum radius for the thermal response factors. Default is None.
    r_min_indices : tuple of array-like, optional
        Indices at which to apply the minimum radius. Default is None.
    backend : str, optional
        Backend to use for the thermal response factors. Options are 'similarities' and 'detailed'. Default is 'similarities'.
    fls_backend : str, optional
        Backend to use for the finite line source solution. Options are 'recurrence' and 'quadrature'. Default is 'recurrence'.
    deg : int, optional
        Degree of the polynomial approximation. Default is None.
    real_source : bool, optional
        Whether to consider real sources. Default is True.
    mirror_source : bool, optional
        Whether to consider mirror sources. Default is True.

    Returns
    -------
    h : array
        Thermal response factors to the point.

    """
    I = finite_line_source(time, alpha, p, source, basis.n_nodes, r_min=r_min, r_min_indices=r_min_indices, backend=fls_backend, deg=deg, real_source=real_source, mirror_source=mirror_source)
    h = I @ basis._coefficients
    return h


def thermal_response_factors_to_borehole(
        time: npt.ArrayLike,
        alpha: float,
        target: Borefield | Borehole,
        source: Borefield | Borehole,
        basis: Basis,
        r_min: float | None = None,
        r_min_indices: Tuple[npt.ArrayLike, npt.ArrayLike] | None = None,
        backend: str = 'similarities',
        fls_backend: str = 'recurrence',
        deg: int | None = None,
        real_source: bool = True,
        mirror_source: bool = True,
        **kwargs
        ) -> np.ndarray:
    """
    Calculate thermal response factors to a borehole.

    Parameters
    ----------
    time : array-like
        Time steps at which to evaluate the thermal response factors.
    alpha : float
        Thermal diffusivity of the ground.
    target : Borefield or Borehole
        Target for the thermal response factors.
    source : Borefield or Borehole
        Source for the thermal response factors.
    basis : Basis
        Basis for the thermal response factors.
    r_min : float or array-like, optional
        Minimum radius for the thermal response factors. Default is None.
    r_min_indices : tuple of array-like, optional
        Indices at which to apply the minimum radius. Default is None.
    backend : str, optional
        Backend to use for the thermal response factors. Options are 'similarities' and 'detailed'. Default is 'similarities'.
    fls_backend : str, optional
        Backend to use for the finite line source solution. Options are 'recurrence' and 'quadrature'. Default is 'recurrence'.
    deg : int, optional
        Degree of the polynomial approximation. Default is None.
    real_source : bool, optional
        Whether to consider real sources. Default is True.
    mirror_source : bool, optional
        Whether to consider mirror sources. Default is True.

    Returns
    -------
    h : array
        Thermal response factors to the borehole.

    """
    if backend == 'similarities':
        h = _similarities._thermal_response_factors_to_borehole(
            time,
            alpha,
            target,
            source,
            basis,
            r_min=r_min,
            r_min_indices=r_min_indices,
            fls_backend=fls_backend,
            deg=deg,
            real_source=real_source,
            mirror_source=mirror_source,
            **kwargs
        )
        return h
    elif backend == 'detailed':
        p_target = target.position(basis.xi)
        if isinstance(target, Borefield):
            p_target = p_target.swapaxes(0, 1)
        I = finite_line_source(time, alpha, p_target, source, basis.n_nodes, r_min=r_min, r_min_indices=r_min_indices, backend=fls_backend, deg=deg, real_source=real_source, mirror_source=mirror_source)
        h = I @ basis._coefficients
        return h
    else:
        warnings.warn(f"Unknown backend '{backend}' for thermal response factors. Falling back to 'detailed'.")
        h = thermal_response_factors_to_borehole(
            time,
            alpha,
            target,
            source,
            basis,
            r_min=r_min,
            r_min_indices=r_min_indices,
            backend='detailed',
            fls_backend=fls_backend,
            deg=deg,
            real_source=real_source,
            mirror_source=mirror_source,
        )
        return h


# def thermal_response_factors_to_borehole(
#         time: npt.ArrayLike,
#         alpha: float,
#         target: Borefield | Borehole,
#         source: Borefield | Borehole,
#         basis: Basis,
#         r_min: float = 0.,
#         method: str = 'recurrence',
#         deg: int | None = None,
#         real_source: bool = True,
#         mirror_source: bool = True
#         ) -> np.ndarray:
#     p_target = target.position(basis.xi)
#     if isinstance(target, Borefield):
#         p_target = p_target.swapaxes(0, 1)
#     I = finite_line_source(time, alpha, p_target, source, basis.n_nodes, r_min=r_min, method=method, deg=deg, real_source=real_source, mirror_source=mirror_source)
#     h = I @ basis._psi_coefs
#     return h


# def thermal_response_factors_to_borehole(
#         time: npt.ArrayLike,
#         alpha: float,
#         target: Borefield | Borehole,
#         source: Borefield | Borehole,
#         basis: Basis,
#         r_min: float = 0.,
#         method: str = 'recurrence',
#         deg: int | None = None,
#         real_source: bool = True,
#         mirror_source: bool = True
#         ) -> np.ndarray:

#     tic = perf_counter()

#     # Find unique (L, D) combinations
#     if isinstance(source, Borefield):
#         source_labels = np.zeros(source.n_boreholes, dtype=int)
#         reference_sources = []
#         LD, unique_indices, source_labels = np.unique(np.column_stack((source.L, source.D)), axis=0, return_index=True, return_inverse=True)
#         for i, (L, D) in enumerate(LD):
#             reference_sources.append(
#                 Borehole(
#                     L=L,
#                     D=D,
#                     r_b=source.r_b[0],
#                     p_head=np.array([0., 0.])
#                 )
#             )
#     else:
#         source_labels = np.array([0], dtype=int)
#         reference_sources = [
#             Borehole(
#                 L=source.L,
#                 D=source.D,
#                 r_b=source.r_b,
#                 p_head=np.array([0., 0.])
#             )
#         ]

#     # Get reference boreholes
#     reference_source = Borehole(
#         L=source.L[0],
#         D=source.D[0],
#         r_b=source.r_b[0],
#         p_head=np.array([0., 0.])
#     )
#     reference_target = Borehole(
#         L=target.L[0],
#         D=target.D[0],
#         r_b=target.r_b[0],
#         p_head=np.array([0., 0.])
#     )

#     # Mid-length positions of the target boreholes
#     r = source.radial_position(target.position(0.5), r_min=r_min)
#     s = reference_source.axial_position(reference_target.position(basis.xi))

#     rTol = 1e-2
#     sTol = 1e-2
#     n_sources = source.n_boreholes
#     n_targets = target.n_boreholes
#     n_nodes = basis.n_nodes
#     indices = np.zeros((n_targets * n_sources, n_nodes), dtype=int)
#     # Cluster the radial positions and get the unique positions
#     r_sort = np.argsort(r.flatten())
#     x = r.flatten()[r_sort]
#     n_points = 0
#     j0 = 0
#     j1 = 1
#     p_head = np.zeros((0, 2))
#     p = np.zeros((0, 3))
#     while j0 < len(x) and j1 > 0:
#         # Find next unique radial position
#         # j1 = np.searchsorted(x, x[j0] * (1 + rTol), side='right')
#         j1 = np.argmax(
#             x >= (1 + rTol) * x[j0])
#         if j1 <= j0: j1 = len(x)
#         r_mean = np.mean(x[j0:j1])
#         p_head = np.append(p_head, [[r_mean, 0.]], axis=0)
#         # Find unique axial positions for the current radial position
#         y = s / r_mean

#         i0 = 0
#         i1 = 1
#         while i0 < len(y) and i1 > 0:
#             # i1 = np.searchsorted(y, y[i0] + sTol, side='right')
#             i1 = np.argmax(
#                 y >= sTol + y[i0]
#                 )
#             if i1 <= i0: i1 = len(y)
#             s_mean = np.mean(s[i0:i1])
#             p = np.append(p, [[r_mean, 0., -reference_source.D - s_mean]], axis=0)
#             indices[r_sort[j0:j1], i0:i1] = n_points
#             n_points += 1
#             i0 = i1
#         j0 = j1

#     indices = indices.reshape((n_targets, n_sources, n_nodes)).swapaxes(1, 2).flatten()


#     I = finite_line_source(time, alpha, p, reference_source, basis.n_nodes, r_min=0., method=method, deg=deg, real_source=real_source, mirror_source=mirror_source)
#     h = I @ basis._psi_coefs
#     h = h[..., indices, :].reshape((-1, target.n_boreholes, basis.n_nodes, source.n_boreholes, basis.n_nodes))
#     return h


# def thermal_response_factors_to_borehole_all_equal_all_vertical(
#         time: npt.ArrayLike,
#         alpha: float,
#         target: Borefield | Borehole,
#         source: Borefield | Borehole,
#         basis: Basis,
#         r_min: float = 0.,
#         method: str = 'recurrence',
#         deg: int | None = None,
#         real_source: bool = True,
#         mirror_source: bool = True
#         ) -> np.ndarray:

#     tic = perf_counter()
#     # Get reference boreholes
#     reference_source = Borehole(
#         L=source.L[0],
#         D=source.D[0],
#         r_b=source.r_b[0],
#         p_head=np.array([0., 0.])
#     )
#     reference_target = Borehole(
#         L=target.L[0],
#         D=target.D[0],
#         r_b=target.r_b[0],
#         p_head=np.array([0., 0.])
#     )

#     # Mid-length positions of the target boreholes
#     r = source.radial_position(target.position(0.5), r_min=r_min)
#     s = reference_source.axial_position(reference_target.position(basis.xi))

#     rTol = 1e-2
#     sTol = 1e-2
#     n_sources = source.n_boreholes
#     n_targets = target.n_boreholes
#     n_nodes = basis.n_nodes
#     indices = np.zeros((n_targets * n_sources, n_nodes), dtype=int)
#     # Cluster the radial positions and get the unique positions
#     r_sort = np.argsort(r.flatten())
#     x = r.flatten()[r_sort]
#     n_points = 0
#     j0 = 0
#     j1 = 1
#     p_head = np.zeros((0, 2))
#     p = np.zeros((0, 3))
#     while j0 < len(x) and j1 > 0:
#         # Find next unique radial position
#         # j1 = np.searchsorted(x, x[j0] * (1 + rTol), side='right')
#         j1 = np.argmax(
#             x >= (1 + rTol) * x[j0])
#         if j1 <= j0: j1 = len(x)
#         r_mean = np.mean(x[j0:j1])
#         p_head = np.append(p_head, [[r_mean, 0.]], axis=0)
#         # Find unique axial positions for the current radial position
#         y = s / r_mean

#         i0 = 0
#         i1 = 1
#         while i0 < len(y) and i1 > 0:
#             # i1 = np.searchsorted(y, y[i0] + sTol, side='right')
#             i1 = np.argmax(
#                 y >= sTol + y[i0]
#                 )
#             if i1 <= i0: i1 = len(y)
#             s_mean = np.mean(s[i0:i1])
#             p = np.append(p, [[r_mean, 0., -reference_source.D - s_mean]], axis=0)
#             indices[r_sort[j0:j1], i0:i1] = n_points
#             n_points += 1
#             i0 = i1
#         j0 = j1

#     indices = indices.reshape((n_targets, n_sources, n_nodes)).swapaxes(1, 2).flatten()


#     I = finite_line_source(time, alpha, p, reference_source, basis.n_nodes, r_min=0., method=method, deg=deg, real_source=real_source, mirror_source=mirror_source)
#     h = I @ basis._psi_coefs
#     h = h[..., indices, :].reshape((-1, target.n_boreholes, basis.n_nodes, source.n_boreholes, basis.n_nodes))
#     return h
