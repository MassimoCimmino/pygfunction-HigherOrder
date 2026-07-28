# -*- coding: utf-8 -*-
from collections.abc import Callable
from functools import cache
from typing import Tuple
import warnings

import numpy as np
import numpy.typing as npt
from scipy.integrate import quad_vec
from scipy.linalg import pascal
from scipy.special import erfc, erfcx

from .first_order_finite_line_source import _first_order_integral
from .zeroth_order_finite_line_source import _zeroth_order_finite_line_source
from ._inner_integral import _inner_integral
from ...borehole import Borehole
from ...borefield import Borefield
from ...utils import mirror_point


def finite_line_source(
        time: npt.ArrayLike,
        alpha: float,
        p: npt.ArrayLike,
        source: Borefield | Borehole,
        n: int,
        r_min: npt.ArrayLike |float | None = None,
        r_min_indices: Tuple[npt.ArrayLike, npt.ArrayLike] | npt.ArrayLike | None = None,
        backend: str = 'recurrence',
        deg: int | None = None,
        real_source: bool = True,
        mirror_source: bool = True
        ) -> np.ndarray:
    """
    Finite line source solution up to order n.

    Parameters
    ----------
    time : array_like
        Time(s) at which to evaluate the solution.
    alpha : float
        Thermal diffusivity of the ground.
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
        Finite line source solution up to order n at the target points and times.

    """
    if isinstance(time, list):
        time = np.asarray(time)
    # Lower bound of integration
    u = 1 / np.sqrt(4 * alpha * time)

    # Real points
    if real_source:
        r, s = source.radial_axial_position(p, r_min=r_min, r_min_indices=r_min_indices)
    else:
        r = np.empty(p.shape[:-1])
        s = np.empty(p.shape[:-1])
    # Mirror points
    if mirror_source:
        p_mirror = mirror_point(p)
        r_mirror, s_mirror = source.radial_axial_position(p_mirror, r_min=None, r_min_indices=None)
    else:
        r_mirror = np.empty(p.shape[:-1])
        s_mirror = np.empty(p.shape[:-1])

    # Reshape inputs for broadcasting
    shape_u = np.shape(u)
    shape_s = np.shape(s)
    u = np.reshape(u, shape_u + (1,) * len(shape_s))

    # Finite line source solution
    if backend.lower() == 'recurrence':
        I = _finite_line_source_recurrence(u, source.L, r, s, r_mirror, s_mirror, n, deg=deg, real_source=real_source, mirror_source=mirror_source)
    elif backend.lower() == 'integral':
        I = _finite_line_source_integral(u, source.L, r, s, r_mirror, s_mirror, n, real_source=real_source, mirror_source=mirror_source)
    else:
        warnings.warn(f"Invalid backend '{backend}' for finite line source solution. Falling back to recurrence method.")
        I = _finite_line_source_recurrence(u, source.L, r, s, r_mirror, s_mirror, n, deg=deg, real_source=real_source, mirror_source=mirror_source)

    return I


def _boundary_integral(u: np.ndarray, r: np.ndarray, s: np.ndarray) -> np.ndarray:
    arg = s * s + r * r
    sqrt_arg = np.sqrt(arg)
    return np.exp(-u * u * arg) * (1 / u - (sqrt_arg * np.sqrt(np.pi) * erfcx(u * sqrt_arg)))


def _finite_line_source_integrand(
        u: np.ndarray,
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
                erfc(u * dis) / dis
                - erfc(u * dis_mirror) / dis_mirror,
                psi
            )
            return dh
    elif real_source:
        def integrand(xi):
            s_self = np.multiply.outer(L / 2, 1 + xi)
            dis = np.sqrt(r**2 + (s - s_self)**2)
            psi = np.power.outer(xi, np.arange(n))
            dh = 0.25 * np.expand_dims(L, axis=-1) * np.multiply.outer(
                erfc(u * dis) / dis,
                psi
            )
            return dh
    elif mirror_source:
        def integrand(xi):
            s_self = np.multiply.outer(L / 2, 1 + xi)
            dis_mirror = np.sqrt(r_mirror**2 + (s_mirror - s_self)**2)
            psi = np.power.outer(xi, np.arange(n))
            dh = -0.25 * np.expand_dims(L, axis=-1) * np.multiply.outer(
                erfc(u * dis_mirror) / dis_mirror,
                psi
            )
            return dh
    else:
        # TODO: return zeros of correct shape
        def integrand(xi):
            shape = np.broadcast_shapes(np.shape(u), np.shape(s)) + (n,)
            dh = np.zeros(shape)
            return dh

    return integrand


def _finite_line_source_integral(
        u: np.ndarray,
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
        integrand = _finite_line_source_integrand(
            u, L, r, s, r_mirror, s_mirror, n, real_source=real_source, mirror_source=mirror_source)
        I = quad_vec(integrand, -1, 1)[0]
    else:
        # TODO: return zeros of correct shape
        shape = np.broadcast_shapes(np.shape(u), np.shape(s)) + (n,)
        I = np.zeros(shape)
    return I


def _finite_line_source_recurrence(
        u: np.ndarray,
        L: float | np.ndarray,
        r: np.ndarray,
        s: np.ndarray,
        r_mirror: np.ndarray,
        s_mirror: np.ndarray,
        n: int,
        deg: int | None = None,
        real_source: bool = True,
        mirror_source: bool = True
        ) -> np.ndarray:
    if real_source or mirror_source:
        binom = _pascal(n)
        _n, _k = np.tril_indices(n)
        exponent = np.zeros((n, n), dtype=int)
        exponent[_n, _k] = _n - _k
    shape = np.broadcast_shapes(np.shape(u), np.shape(s)) + (n,)
    I = np.zeros(shape)
    # Real integrals
    if real_source:
        J_real = _integral(u, L, r, s, n, deg=deg)
        eta_real = 2 * s / L - 1
        I += np.einsum('nk,...nk,...k->...n', binom, np.power.outer(eta_real, exponent), J_real)
    if mirror_source:
        J_mirror = _integral(u, L, r_mirror, s_mirror, n, deg=deg)
        eta_mirror = 2 * s_mirror / L - 1
        I -= np.einsum('nk,...nk,...k->...n', binom, np.power.outer(eta_mirror, exponent), J_mirror)

    return I


def _integral(
        u: np.ndarray,
        L: float | np.ndarray,
        r: np.ndarray,
        s: np.ndarray,
        n: int,
        deg: int | None = None
        ) -> np.ndarray:
    if n == 0:
        J_0 = _zeroth_order_finite_line_source(u, L, r, s, r_mirror=np.empty_like(r), s_mirror=np.empty_like(s), deg=deg, mirror_source=False)
        J = np.expand_dims(J_0, axis=-1)
    elif n == 1:
        J_0 = _zeroth_order_finite_line_source(u, L, r, s, r_mirror=np.empty_like(r), s_mirror=np.empty_like(s), deg=deg, mirror_source=False)
        J_1 = _first_order_integral(u, L, r, s)
        J = np.stack([J_0, J_1], axis=-1)
    else:
        shape = np.broadcast_shapes(np.shape(u), np.shape(s)) + (n,)
        J = np.empty(shape)
        J_0 = _zeroth_order_finite_line_source(u, L, r, s, r_mirror=np.empty_like(r), s_mirror=np.empty_like(s), deg=deg, mirror_source=False)
        J_1 = _first_order_integral(u, L, r, s)
        J[..., 0] = J_0
        J[..., 1] = J_1

        uu = u * u
        LL = L * L
        rr = r * r
        rr_over_LL = rr / LL
        exp_coef = 2 / (u * LL) * np.exp(-uu * rr)
        X = _inner_integral(u, L, s, n - 2)
        eta_0 = -2 * s / L
        eta_L = 2 * (L - s) / L
        B_0 = _boundary_integral(u, r, -s) / (L * np.sqrt(np.pi))
        B_L = _boundary_integral(u, r, L - s) / (L * np.sqrt(np.pi))
        for k in range(2, n):
            J[..., k] = (k - 1) / k * exp_coef * X[..., k - 2]
            J[..., k] -= 4 * (k - 1) / k * rr_over_LL * J[..., k - 2]
            J[..., k] -= 1 / k * (eta_L**(k - 1) * B_L - eta_0**(k - 1) * B_0)

    return J


@cache
def _pascal(n: int) -> np.ndarray:
    return pascal(n, kind='lower')
