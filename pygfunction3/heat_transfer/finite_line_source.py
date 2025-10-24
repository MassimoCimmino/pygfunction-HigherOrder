# -*- coding: utf-8 -*-
from collections.abc import Callable
from functools import cache
import warnings

import numpy as np
import numpy.typing as npt
from scipy.integrate import quad_vec
from scipy.linalg import pascal
from scipy.special import erfc, erfcx

from .first_order_finite_line_source import _first_order_integral
from .zeroth_order_finite_line_source import _zeroth_order_finite_line_source
from ._inner_integral import _inner_integral
from ..utils import mirror_point


def finite_line_source(
        time: npt.ArrayLike,
        alpha: float,
        p: npt.ArrayLike,
        borehole,
        basis,
        r_min: float = 0.,
        real_source: bool = True,
        mirror_source: bool = True
        ) -> np.ndarray:

    if real_source or mirror_source:
        integrand = _finite_line_source_integrand(p, borehole, basis, r_min=r_min, real_source=real_source, mirror_source=mirror_source)
        I = quad_vec(
            integrand,
            -1., 1.,
            args=(time, alpha),
            epsabs=1.0e-6,
            epsrel=1.0e-6
        )[0]
    else:
        # TODO: return zeros of correct shape
        shape = np.shape(time) + tuple(np.ones(np.ndim(p), dtype=int))
        I = np.zeros(shape)
    return I


def _boundary_integral(u, r, s):
    arg = s**2 + r**2
    sqrt_arg = np.sqrt(arg)
    return (
        np.exp(np.multiply.outer(-u**2, arg)) * (
            1 / u - (sqrt_arg * np.sqrt(np.pi) * erfcx(np.multiply.outer(u, sqrt_arg))).T).T
        )


def _finite_line_source_integrand(
        p,
        borehole,
        basis,
        r_min: float = 0.,
        real_source: bool = True,
        mirror_source: bool = True
        ) -> Callable:

    # Real points
    if real_source:
        r, s = borehole.radial_axial_position(p, r_min=r_min)
    # Mirror points
    if mirror_source:
        p_mirror = mirror_point(p)
        r_mirror, s_mirror = borehole.radial_axial_position(p_mirror, r_min=r_min)

    # Integration function
    if real_source and mirror_source:
        def integrand(xi, time, alpha):
            s_self = borehole.longitudinal_position(xi)
            dis = np.sqrt(r**2 + (s - s_self)**2)
            dis_mirror = np.sqrt(r_mirror**2 + (s_mirror - s_self)**2)
            psi = basis.basis_function(xi)
            two_sqrt_alpha_time = 2 * np.sqrt(alpha * time)
            dh = 0.25 * borehole.L * np.multiply.outer(
                erfc(np.divide.outer(dis, two_sqrt_alpha_time).T) / dis
                - erfc(np.divide.outer(dis_mirror, two_sqrt_alpha_time).T) / dis_mirror,
                psi
            )
            return dh
    elif real_source:
        def integrand(xi, time, alpha):
            s_self = borehole.longitudinal_position(xi)
            dis = np.sqrt(r**2 + (s - s_self)**2)
            psi = basis.basis_function(xi)
            two_sqrt_alpha_time = 2 * np.sqrt(alpha * time)
            dh = 0.25 * borehole.L * np.multiply.outer(
                erfc(np.divide.outer(dis, two_sqrt_alpha_time).T) / dis,
                psi
            )
            return dh
    elif mirror_source:
        def integrand(xi, time, alpha):
            s_self = borehole.longitudinal_position(xi)
            dis_mirror = np.sqrt(r_mirror**2 + (s_mirror - s_self)**2)
            psi = basis.basis_function(xi)
            two_sqrt_alpha_time = 2 * np.sqrt(alpha * time)
            dh = -0.25 * borehole.L * np.multiply.outer(
                erfc(np.divide.outer(dis_mirror, two_sqrt_alpha_time).T) / dis_mirror,
                psi
            )
            return dh
    else:
        # TODO: return zeros of correct shape
        def integrand(xi, time, alpha):
            shape = np.shape(time) + tuple(np.ones(np.ndim(p), dtype=int))
            dh = np.zeros(shape)
            return dh

    return integrand


def _finite_line_source_recurrence(u, L, r, s, r_mirror, s_mirror, n, deg=None, real_source=True, mirror_source=True):
    if real_source or mirror_source:
        binom = _pascal(n)
        _n, _k = np.tril_indices(n)
        exponent = np.zeros((n, n), dtype=int)
        exponent[_n, _k] = _n - _k
    I = np.zeros(np.shape(u) + np.shape(s) + (n,))
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


def _integral(u, L, r, s, n, deg: int | None = None,):
    if n == 0:
        J_0 = _zeroth_order_finite_line_source(u, L, r, s, r_mirror=np.empty_like(r), s_mirror=np.empty_like(s), deg=deg, mirror_source=False)
        J = np.expand_dims(I_0, axis=-1)
    elif n == 1:
        J_0 = _zeroth_order_finite_line_source(u, L, r, s, r_mirror=np.empty_like(r), s_mirror=np.empty_like(s), deg=deg, mirror_source=False)
        J_1 = _first_order_integral(u, L, r, s)
        J = np.stack([J_0, J_1], axis=-1)
    else:
        J = np.empty(np.shape(u) + np.shape(s) + (n,))
        J_0 = _zeroth_order_finite_line_source(u, L, r, s, r_mirror=np.empty_like(r), s_mirror=np.empty_like(s), deg=deg, mirror_source=False)
        J_1 = _first_order_integral(u, L, r, s)
        J[..., 0] = J_0
        J[..., 1] = J_1

        uu = u * u
        LL = L * L
        rr = r * r
        rr_over_LL = rr / LL
        exp_coef = (np.divide.outer(2 / u, LL).T * np.exp(np.multiply.outer(-uu, rr)).T).T
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
def _pascal(n):
    return pascal(n, kind='lower')
