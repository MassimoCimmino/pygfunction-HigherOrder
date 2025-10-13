# -*- coding: utf-8 -*-
from typing_extensions import Self     # for compatibility with Python <= 3.10
import warnings

import numpy as np
from numpy.polynomial.polynomial import polyfit, polyval
import numpy.typing as npt
from scipy.special import roots_legendre


class Basis:
    """Polynomial basis functions.

    Parameters
    ----------
    xi : array_like
        (`n_nodes`,) array of node coordinates along the interval
        ``[-1, 1]``.
    w : array_like or None, default: None
        Quadrature weights for numerical integration using `xi` as
        quadrature points. If `w` is ``None``, the weights are evaluated
        through numerical integration of order `order`.

    Attributes
    ----------
    n_nodes : int
        Number of nodes.

    """

    def __init__(self, n_nodes: int):
        # --- Class attributes ---
        self.n_nodes = n_nodes

        # --- Other attributes ---
        # Node coordinates and weights
        self.xi, self.w = roots_legendre(n_nodes)

        # --- Basis functions (psi) ---
        self._psi_coefs = polyfit(
            self.xi,
            np.eye(self.n_nodes),
            self.n_nodes - 1
            )

    def basis_function(self, xi: npt.ArrayLike) -> np.ndarray:
        """Polynomial basis functions.

        Parameters
        ----------
        xi : float or array
            Coordinate or (M,) array of coordinates.

        Returns
        -------
        array
            (`n_nodes`,) or (M, `n_nodes`,) array of polynomial basis
            functions.

        """
        return self._basis_function(xi, self._psi_coefs)

    @staticmethod
    def _basis_function(xi: npt.ArrayLike, psi_coefs: np.ndarray) -> np.ndarray:
        """Polynomial basis functions.

        Parameters
        ----------
        xi : float
            Coordinate.
        psi_coefs : array
            (`n_nodes`, `n_nodes`,) array of polynomial coefficients
            for the evaluation of the polynomial basis functions.

        Returns
        -------
        array
            (`n_nodes`,) array of polynomial basis functions.

        """
        psi = polyval(xi, psi_coefs)
        return psi.T
