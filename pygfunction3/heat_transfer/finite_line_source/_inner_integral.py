# -*- coding: utf-8 -*-
import warnings

import numpy as np
import numpy.typing as npt
from scipy.special import erf


def _first_order_inner_integral(
        u: np.ndarray,
        L: float | np.ndarray,
        s: np.ndarray):
    """First order inner integral of the finite line source solution."""
    uu = u * u
    term1 = np.exp(-uu * (L - s)**2)
    term2 = np.exp(-uu * s * s)
    X_1 = (term2 - term1) / (uu * L * np.sqrt(np.pi))
    return X_1


def _inner_integral(
        u: np.ndarray,
        L: float | np.ndarray,
        s: np.ndarray,
        n: int):
    """Inner integrals of the finite line source solution."""
    if n == 0:
        X_0 = _zeroth_order_inner_integral(u, L, s)
        X = np.expand_dims(X_0, axis=-1)
        return X
    elif n == 1:
        X_0 = _zeroth_order_inner_integral(u, L, s)
        X_1 = _first_order_inner_integral(u, L, s)
        X = np.stack([X_0, X_1], axis=-1)
        return X
    else:
        X = np.empty(np.broadcast_shapes(np.shape(u), np.shape(s)) + (n,))
        X[..., 0] = _zeroth_order_inner_integral(u, L, s)
        X[..., 1] = _first_order_inner_integral(u, L, s)

        uu = u * u
        arg_0 = uu * s * s
        eta_0 = - 2 * s / L
        term_0 = np.exp(-arg_0) / (uu * L * np.sqrt(np.pi))
        arg_L = uu * (L - s)**2
        eta_L = 2 * (L - s) / L
        term_L = np.exp(-arg_L) / (uu * L * np.sqrt(np.pi))

        LL = L * L
        inv_uu_LL = 1 / (uu * LL)
        p = np.arange(1, n - 1)
        X[..., 2:] = np.expand_dims(term_0, axis=-1) * np.power.outer(eta_0, p) - np.expand_dims(term_L, axis=-1) * np.power.outer(eta_L, p)
        for k in range(2, n):
            X[..., k] += 2 * (k - 1) * X[..., k - 2] * inv_uu_LL
    return X


def _zeroth_order_inner_integral(
        u: np.ndarray,
        L: float | np.ndarray,
        s: np.ndarray):
    """Zeroth order inner integral of the finite line source solution."""
    term1 = erf(u * (L - s))
    term2 = erf(u * s)
    X_0 = 0.5 / u * (term1 + term2)
    return X_0
