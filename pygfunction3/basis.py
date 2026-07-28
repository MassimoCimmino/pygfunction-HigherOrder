# -*- coding: utf-8 -*-
from functools import cached_property, lru_cache
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

    @cached_property
    def _coefficients(self) -> np.ndarray:
        """Coefficients of the polynomial basis functions.

        Returns
        -------
        array
            (`n_nodes`, `n_nodes`,) array of polynomial coefficients for the
            evaluation of the polynomial basis functions.

        """
        psi_coefs = polyfit(
            self.xi,
            np.eye(self.n_nodes),
            self.n_nodes - 1
            )
        return psi_coefs

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
        if isinstance(xi, list):
            xi = np.asarray(xi)
        return self._basis_function(xi, self._coefficients)

    @lru_cache(maxsize=8)
    def _shifted_coefficients(self, shift: float=-1.) -> np.ndarray:
        """Coefficients of the shifted polynomial basis functions.

        Parameters
        ----------
        shift : float, default: -1
            Shift of the node coordinates for the evaluation of the shifted
            polynomial basis functions. The node coordinates are shifted as
            ``zeta = xi - shift``. Default is -1, which shifts the node
            coordinates to the interval [0, 2].

        Returns
        -------
        array
            (`n_nodes`, `n_nodes`,) array of polynomial coefficients for the
            evaluation of the shifted polynomial basis functions.

        """
        xi_shifted = self.xi - shift
        psi_coefs_shifted = polyfit(
            xi_shifted,
            np.eye(self.n_nodes),
            self.n_nodes - 1
            )
        return psi_coefs_shifted

    @classmethod
    def _basis_function(cls, xi: npt.ArrayLike, psi_coefs: np.ndarray) -> np.ndarray:
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
        if isinstance(xi, list):
            xi = np.asarray(xi)
        psi = polyval(xi, psi_coefs)
        return psi.T
