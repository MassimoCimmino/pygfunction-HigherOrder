# -*- coding: utf-8 -*-
import numpy as np
from numpy.typing import ArrayLike

from ._parallel_pipes import _ParallelPipes
from ..borehole import Borehole


class _SinglePipe(_ParallelPipes):
    """Ground heat exchanger cross-section.

    Attributes
    ----------
    fluid : Fluid
        Fluid used in the pipe.
    n_pipes : int
        Total number of pipes.
    n_pipes_down : int
        Number of downward-flowing pipes.
        Equals 1.
    n_pipes_up : int
        Number of upward-flowing pipes.
        Equals 1.

    """

    def _state_transition_matrix(
            self, borehole: Borehole, xi: float | ArrayLike, m_flow: float,
            T_f: float | None = None, xi_0: float | None = None) -> np.ndarray:
        """Evaluate the state transition matrix of the pipe.

        Parameters
        ----------
        borehole : Borehole
            Borehole containing the pipe.
        xi : float or array-like
            Dimensionless position along the borehole, or (``N``,) array of
            dimensionless positions.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.
        xi_0 : float or None, default: None
            Starting dimensionless position along the borehole. If ``None``, the
            starting position is assumed to be 0.

        Returns
        -------
        phi : array
            (`n_pipes`, `n_pipes`,) or (``N``, `n_pipes`, `n_pipes`,) array of
            state transition matrix.

        """
        s = borehole.longitudinal_position(xi)
        if xi_0 is not None:
            s_0 = borehole.longitudinal_position(xi_0)
            s = s - s_0

        A = self._ode_coefficients(m_flow, T_f=T_f)
        tau = np.trace(A) / 2
        delta = np.sqrt((A[0, 0] - A[1, 1])**2 / 4 + A[0, 1] * A[1, 0])

        phi = (
            np.exp(tau * s) * (
                np.multiply.outer(np.cosh(delta * s), np.eye(self.n_pipes)).T
                + np.multiply.outer(np.sinh(delta * s) / delta, (A - tau * np.eye(self.n_pipes))).T
            )
        ).T
        return phi
