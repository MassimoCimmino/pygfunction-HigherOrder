# -*- coding: utf-8 -*-
from collections.abc import Callable
from functools import cache
from typing import Tuple
import warnings

import numpy as np
import numpy.typing as npt
from scipy.integrate import quad_vec
from scipy.linalg import pascal

from .first_order_steady_state_finite_line_source import _first_order_steady_state_integral
from .zeroth_order_steady_state_finite_line_source import _zeroth_order_steady_state_finite_line_source
from ...borehole import Borehole
from ...borefield import Borefield
from ...utils import mirror_point


def steady_state_finite_line_source(
        p: npt.ArrayLike,
        source: Borefield | Borehole,
        n: int,
        r_min: npt.ArrayLike |float | None = None,
        r_min_indices: Tuple[npt.ArrayLike, npt.ArrayLike] | npt.ArrayLike | None = None,
        backend: str = 'recurrence',
        real_source: bool = True,
        mirror_source: bool = True
        ) -> np.ndarray:
    """
    Steady-state finite line source solution up to order n.

    Parameters
    ----------
    p : array_like
        Target points.
    source : Borefield or Borehole
        Source borefield or borehole.
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
        Steady-state finite line source solution up to order n at the target points.

    """

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

    # Finite line source solution
    I = _steady_state_finite_line_source(source.L, r, s, r_mirror, s_mirror, n, backend=backend, real_source=real_source, mirror_source=mirror_source)

    return I


def _steady_state_finite_line_source(
        L: float | np.ndarray,
        r: np.ndarray,
        s: np.ndarray,
        r_mirror: np.ndarray,
        s_mirror: np.ndarray,
        n: int,
        backend: str = 'recurrence',
        real_source: bool = True,
        mirror_source: bool = True
        ) -> np.ndarray:
    """
    Steady-state finite line source solution up to order n.

    Parameters
    ----------
    L : float or array
        Length of the source borehole(s).
    r : array
        Radial position(s) of the target point(s).
    s : array
        Axial position(s) of the target point(s).
    r_mirror : array
        Radial position(s) of the mirror point(s).
    s_mirror : array
        Axial position(s) of the mirror point(s).
    n : int
        Order of the solution.
    backend : str, optional
        Backend for the finite line source solution. Must be either 'recurrence' or 'integral'. Default is 'recurrence'.
    real_source : bool, optional
        Whether to include the contribution from the real source. Default is True.
    mirror_source : bool, optional
        Whether to include the contribution from the mirror source. Default is True.

    Returns
    -------
    I : ndarray
        Steady-state finite line source solution up to order n at the target points.

    """
    # Finite line source solution
    if backend.lower() == 'recurrence':
        I = _steady_state_finite_line_source_recurrence(L, r, s, r_mirror, s_mirror, n, real_source=real_source, mirror_source=mirror_source)
    elif backend.lower() == 'integral':
        I = _steady_state_finite_line_source_integral(L, r, s, r_mirror, s_mirror, n, real_source=real_source, mirror_source=mirror_source)
    else:
        warnings.warn(f"Invalid backend '{backend}' for finite line source solution. Falling back to recurrence method.")
        I = _steady_state_finite_line_source_recurrence(L, r, s, r_mirror, s_mirror, n, real_source=real_source, mirror_source=mirror_source)
    return I


def _steady_state_finite_line_source_integrand(
        L: float | np.ndarray,
        r: np.ndarray,
        s: np.ndarray,
        r_mirror: np.ndarray,
        s_mirror: np.ndarray,
        n: int,
        real_source: bool = True,
        mirror_source: bool = True
        ) -> Callable:

    # Integration function
    if real_source and mirror_source:
        def integrand(xi):
            s_self = np.multiply.outer(L / 2, 1 + xi)
            dis = np.sqrt(r**2 + (s - s_self)**2)
            dis_mirror = np.sqrt(r_mirror**2 + (s_mirror - s_self)**2)
            psi = np.power.outer(xi, np.arange(n))
            dh = 0.25 * np.expand_dims(L, axis=-1) * np.multiply.outer(
                1 / dis
                - 1 / dis_mirror,
                psi
            )
            return dh
    elif real_source:
        def integrand(xi):
            s_self = np.multiply.outer(L / 2, 1 + xi)
            dis = np.sqrt(r**2 + (s - s_self)**2)
            psi = np.power.outer(xi, np.arange(n))
            dh = 0.25 * np.expand_dims(L, axis=-1) * np.multiply.outer(
                1 / dis,
                psi
            )
            return dh
    elif mirror_source:
        def integrand(xi):
            s_self = np.multiply.outer(L / 2, 1 + xi)
            dis_mirror = np.sqrt(r_mirror**2 + (s_mirror - s_self)**2)
            psi = np.power.outer(xi, np.arange(n))
            dh = -0.25 * np.expand_dims(L, axis=-1) * np.multiply.outer(
                1 / dis_mirror,
                psi
            )
            return dh
    else:
        # TODO: return zeros of correct shape
        def integrand(xi):
            shape = np.shape(s) + (n,)
            dh = np.zeros(shape)
            return dh

    return integrand


def _steady_state_finite_line_source_integral(
        L: float | np.ndarray,
        r: np.ndarray,
        s: np.ndarray,
        r_mirror: np.ndarray,
        s_mirror: np.ndarray,
        n: int,
        real_source: bool = True,
        mirror_source: bool = True
        ) -> np.ndarray:
    if real_source or mirror_source:
        integrand = _steady_state_finite_line_source_integrand(
            L, r, s, r_mirror, s_mirror, n, real_source=real_source, mirror_source=mirror_source)
        I = quad_vec(integrand, -1, 1)[0]
    else:
        # TODO: return zeros of correct shape
        shape = np.shape(s) + (n,)
        I = np.zeros(shape)
    return I


def _steady_state_finite_line_source_recurrence(
        L: float | np.ndarray,
        r: np.ndarray,
        s: np.ndarray,
        r_mirror: np.ndarray,
        s_mirror: np.ndarray,
        n: int,
        real_source: bool = True,
        mirror_source: bool = True
        ) -> np.ndarray:
    if (real_source or mirror_source) and n > 1:
        binom = _pascal(n)
        _n, _k = np.tril_indices(n)
        exponent = np.zeros((n, n), dtype=int)
        exponent[_n, _k] = _n - _k
    shape = np.shape(s) + (n,)
    I = np.zeros(shape)
    # Real integrals
    if real_source:
        J_real = _steady_state_integral(L, r, s, n)
        if n > 1:
            eta_real = 2 * s / L - 1
            I += np.einsum('nk,...nk,...k->...n', binom, np.power.outer(eta_real, exponent), J_real)
        else:
            I += J_real
    if mirror_source:
        J_mirror = _steady_state_integral(L, r_mirror, s_mirror, n)
        if n > 1:
            eta_mirror = 2 * s_mirror / L - 1
            I -= np.einsum('nk,...nk,...k->...n', binom, np.power.outer(eta_mirror, exponent), J_mirror)
        else:
            I -= J_mirror

    return I


def _steady_state_integral(
        L: float | np.ndarray,
        r: np.ndarray,
        s: np.ndarray,
        n: int,
        ) -> np.ndarray:
    if n == 1:
        J_0 = _zeroth_order_steady_state_finite_line_source(L, r, s, r_mirror=np.empty_like(r), s_mirror=np.empty_like(s), mirror_source=False)
        J = np.expand_dims(J_0, axis=-1)
    elif n == 2:
        J_0 = _zeroth_order_steady_state_finite_line_source(L, r, s, r_mirror=np.empty_like(r), s_mirror=np.empty_like(s), mirror_source=False)
        J_1 = _first_order_steady_state_integral(L, r, s)
        J = np.stack([J_0, J_1], axis=-1)
    else:
        shape = np.shape(s) + (n,)
        J = np.empty(shape)
        J_0 = _zeroth_order_steady_state_finite_line_source(L, r, s, r_mirror=np.empty_like(r), s_mirror=np.empty_like(s), mirror_source=False)
        J_1 = _first_order_steady_state_integral(L, r, s)
        J[..., 0] = J_0
        J[..., 1] = J_1

        LL = L * L
        rr = r * r
        rr_over_LL = rr / LL
        eta_0 = -2 * s / L
        eta_L = 2 * (L - s) / L
        sqrt_arg0 = np.sqrt(rr + s**2) / L
        sqrt_argL = np.sqrt(rr + (L - s)**2) / L
        for k in range(2, n):
            J[..., k] = (eta_L**(k - 1) * sqrt_argL - eta_0**(k - 1) * sqrt_arg0) / k
            J[..., k] -= 4 * (k - 1) / k * rr_over_LL * J[..., k - 2]

    return J


@cache
def _pascal(n: int) -> np.ndarray:
    return pascal(n, kind='lower')
