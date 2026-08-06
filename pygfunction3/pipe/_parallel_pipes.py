# -*- coding: utf-8 -*-
from functools import cached_property, cache
from typing import Tuple

import numpy as np

from ._pipe import _Pipe


class _ParallelPipes(_Pipe):
    """Ground heat exchanger cross-section.

    Attributes
    ----------
    fluid : Fluid
        Fluid used in the pipe.
    n_pipes : int
        Total number of pipes.
    n_pipes_down : int
        Number of downward-flowing pipes.
    n_pipes_up : int
        Number of upward-flowing pipes.

    """

    @cached_property
    def _bottom_continuity_coefficients(self) -> np.ndarray:
        """Continuity condition bottom of the pipe.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{0_{u}} = \\mathbf{a_{f}} \\mathbf{T_{f}}(s)

        Returns
        -------
        a_f : array
            (`n_pipes_up`, `n_pipes`,) array of fluid temperature
            coefficients.

        """
        a_f = np.concatenate(
            (np.eye(self.n_pipes_up), -np.eye(self.n_pipes_up)),
            axis=1
            )
        return a_f

    @cached_property
    def _mass_flow_fractions(self) -> np.ndarray:
        """Mass flow rate factor of the pipe.

        Returns
        -------
        array
            (`n_pipes`,) array of mass flow rate fractions, normalized to the
            total mass flow rate.

        """
        return np.full(self.n_pipes, 1 / self.n_pipes_down)

    @cache
    def _direction_coefficients(self, reverse: bool) -> np.ndarray:
        """Direction coefficients of the pipe.

        Coefficients are positive for downward-flowing pipes and negative for
        upward-flowing pipes.

        Parameters
        ----------
        reverse : bool
            If True, the flow direction is reversed.

        Returns
        -------
        array
            (`n_pipes`,) array of direction coefficients.

        """
        if reverse:
            direction = np.concatenate(
                (-np.ones(self.n_pipes_down), np.ones(self.n_pipes_up))
                )
        else:
            direction = np.concatenate(
                (np.ones(self.n_pipes_down), -np.ones(self.n_pipes_up))
                )
        return direction

    @cache
    def _mixing_coefficients(self, reverse: bool) -> np.ndarray:
        """Mixing coefficients of the pipe.

        Returns coefficients for the relation:

            .. math::

                T_{f,out} = \\mathbf{a_{fu}} \\mathbf{T_{fu}}(s=0)

        Parameters
        ----------
        reverse : bool
            If True, the flow direction is reversed.

        Returns
        -------
        a_fu : array
            (`n_pipes_up`,) array of mixing coefficients.

        """
        a_fu = np.full(self.n_pipes_up, 1.0 / self.n_pipes_up)
        return a_fu

    @cache
    def _pipe_configuration_coefficients(
        self, reverse: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Configuration coefficients to reconstruct the temperature vector.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_{f}}(s) = \\mathbf{a_{fd}} \\mathbf{T_{fd}}(s)
                + \\mathbf{a_{fu}} \\mathbf{T_{fu}}(s)

        Parameters
        ----------
        reverse : bool
            If True, the flow direction is reversed.

        Returns
        -------
        a_fd : array
            (`n_pipes`, `n_pipes_down`,) array of inlet temperature
            coefficients.
        a_fu : array
            (`n_pipes`, `n_pipes_up`,) array of fluid temperature coefficients.

        """
        if reverse:
            a_fd = np.flip(np.eye(self.n_pipes, M=self.n_pipes_up, k=-self.n_pipes_down), axis=1)
            a_fu = np.flip(np.eye(self.n_pipes, M=self.n_pipes_down), axis=1)
        else:
            a_fd = np.eye(self.n_pipes, M=self.n_pipes_down)
            a_fu = np.eye(self.n_pipes, M=self.n_pipes_up, k=-self.n_pipes_down)
        return a_fd, a_fu

    @cache
    def _top_continuity_coefficients(
        self, reverse: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Continuity condition at the top of the pipe.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_{fd}}(s=0) = a_{in} T_{f,in}
                + \\mathbf{a_{fu}} \\mathbf{T_{fu}}(s=0)

        Parameters
        ----------
        reverse : bool
            If True, the flow direction is reversed.

        Returns
        -------
        a_in : array
            (`n_pipes_down`,) array of inlet temperature coefficients.
        a_fu : array
            (`n_pipes_down`, `n_pipes_up`,) array of fluid temperature
            coefficients.

        """
        a_in = np.ones(self.n_pipes_down)
        a_fu = np.zeros((self.n_pipes_down, self.n_pipes_up))
        return a_in, a_fu
