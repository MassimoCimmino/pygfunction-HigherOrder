# -*- coding: utf-8 -*-
from functools import cached_property
from typing import Tuple

import numpy as np

from ._parallel_pipes import _ParallelPipes


class _SeriesPipes(_ParallelPipes):
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
    def _mass_flow_fractions(self) -> np.ndarray:
        """Mass flow rate factor of the pipe.

        Returns
        -------
        array
            (`n_pipes`,) array of mass flow rate fractions, normalized to the
            total mass flow rate.

        """
        return np.ones(self.n_pipes)

    @cached_property
    def _mixing_coefficients(self) -> np.ndarray:
        """Mixing coefficients of the pipe.

        Returns coefficients for the relation:

            .. math::

                T_{f,out} = \\mathbf{a_{fu}} \\mathbf{T_{fu}}(s=0)

        Returns
        -------
        a_fu : array
            (`n_pipes_up`,) array of mixing coefficients.

        """
        a_fu = np.concatenate((np.zeros(self.n_pipes_up - 1), np.ones(1)))
        return a_fu

    @cached_property
    def _top_continuity_coefficients(
        self) -> Tuple[np.ndarray, np.ndarray]:
        """Continuity condition at the top of the pipe.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_{fd}}(s=0) = a_{in} T_{f,in}
                + \\mathbf{a_{fu}} \\mathbf{T_{fu}}(s=0)

        Returns
        -------
        a_in : array
            (`n_pipes_down`,) array of inlet temperature coefficients.
        a_fu : array
            (`n_pipes_down`, `n_pipes_up`,) array of fluid temperature
            coefficients.

        """
        a_in = np.concatenate((np.ones(1), np.zeros(self.n_pipes_down - 1)))
        a_fu = np.eye(self.n_pipes_down, M=self.n_pipes_up, k=-1)
        return a_in, a_fu
