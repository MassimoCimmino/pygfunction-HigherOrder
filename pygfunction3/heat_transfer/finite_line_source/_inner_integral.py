# -*- coding: utf-8 -*-
import warnings

import numpy as np
import numpy.typing as npt
from scipy.special import erf


def _first_order_inner_integral(
        u: npt.ArrayLike,
        L: npt.ArrayLike,
        s: npt.ArrayLike):
    """First order inner integral of the finite line source solution."""
    term1 = np.exp(-np.multiply.outer(u, (L - s))**2)
    term2 = np.exp(-np.multiply.outer(u, s)**2)
    X_1 = (term2 - term1).T / (u**2 * L * np.sqrt(np.pi))
    return X_1.T


def _inner_integral(
        u: npt.ArrayLike,
        L: npt.ArrayLike,
        s: npt.ArrayLike,
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
        X = np.empty(np.shape(u) + np.shape(s) + (n,))
        X[..., 0] = _zeroth_order_inner_integral(u, L, s)
        X[..., 1] = _first_order_inner_integral(u, L, s)

        uu = u * u
        arg_0 = np.multiply.outer(uu, s * s)
        term_0 = np.exp(-arg_0)
        arg_L = np.multiply.outer(uu, (L - s)**2)
        term_L = np.exp(-arg_L)

        LL = L * L
        for k in range(2, n):
            X[..., k] = (2 * (k - 1) / (LL * uu) * X[..., k - 2].T).T
            X[..., k] -= ((
                (2 * (L - s) / L)**(k - 1) * term_L
                - (-2 * s / L)**(k - 1) * term_0
            ).T / (np.sqrt(np.pi) * uu * L)).T
    return X


def _zeroth_order_inner_integral(
        u: npt.ArrayLike,
        L: npt.ArrayLike,
        s: npt.ArrayLike):
    """Zeroth order inner integral of the finite line source solution."""
    term1 = erf(np.multiply.outer(u, (L - s)))
    term2 = erf(np.multiply.outer(u, s))
    X_0 = 0.5 / u * (term1 + term2).T
    return X_0.T
