# -*- coding: utf-8 -*-
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike

from ._single_pipe import _SinglePipe
from ..fluid import Fluid
from ..thermal_resistance import (
conduction_thermal_resistance_circular_pipe,
convective_heat_transfer_coefficient_circular_pipe,
Multipole
)


class SingleUTube(_SinglePipe):
    """U-tube heat exchanger cross-section.

    Parameters
    ----------
    p : array_like
        (`n_pipes`, 2) array of pipe positions (in meters). The position
        ``(0, 0)`` corresponds to the axis of the borehole.
    r_p_in : array_like
        Inner radius of the pipes (in meters).
    r_p_out : array_like
        Outer radius of the pipes (in meters).
    r_b : float
        Borehole radius (in meters).
    fluid : fluid
        The fluid.
    k_s : float
        Ground thermal conductivity (in W/m-K).
    k_b : float
        Grout thermal conductivity (in W/m-K).
    k_p : array_like
        Thermal conductivity of the pipes (in W/m-K).
    epsilon : float
        Pipe surface roughness (in meters).
    parallel : bool, default: True
        True if pipes are in parallel. False if pipes are in series.
    J : int, default: 3
        Order of the multipole solution.

    Attributes
    ----------
    n_pipes : int
        Total number of pipes.
    n_pipes_down : int
        Number of downward-flowing pipes.
        Equals 1.
    n_pipes_up : int
        Number of upward-flowing pipes.
        Equals 1.
    R_p : array
        Conduction thermal resistance (in m-K/W).

    """

    def __init__(
            self, p: ArrayLike, r_p_in: ArrayLike, r_p_out: ArrayLike,
            r_b: float, fluid: Fluid, k_s: float, k_b: float, k_p: ArrayLike,
            epsilon: float, parallel: bool = True, J: int = 3):
        # Convert input to numpy.Array
        p = np.atleast_2d(p)
        r_p_in = np.broadcast_to(r_p_in, p.shape[0])
        r_p_out = np.broadcast_to(r_p_out, p.shape[0])
        k_p = np.broadcast_to(k_p, p.shape[0])

        # --- Class atributes ---
        # Parameters
        self.p = p
        self.r_p_in = r_p_in
        self.r_p_out = r_p_out
        self.r_b = r_b
        self.fluid = fluid
        self.k_s = k_s
        self.k_b = k_b
        self.k_p = k_p
        self.epsilon = epsilon
        self.parallel = parallel
        self.J = J
        # Additional attributes
        self.n_pipes = p.shape[0]
        self.n_pipes_down = int(self.n_pipes / 2)
        self.n_pipes_up = int(self.n_pipes / 2)
        self.R_p = conduction_thermal_resistance_circular_pipe(
            self.r_p_in, self.r_p_out, self.k_p
            )

        # --- Multipole model ---
        self.multipole = Multipole(
            self.r_b, self.r_p_out, self.p, J=self.J)

    @lru_cache(maxsize=8, typed=False)
    def thermal_resistances(
        self, m_flow: float, T_f: float | None = None) -> np.ndarray:
        """Evaluate delta-circuit thermal resistances.

        Parameters
        ----------
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.

        Returns
        -------
        array
            (`n_pipes`, `n_pipes`,) array of delta-circuit thermal
            resistances (in m-K/W).

        """
        m_flow_pipe = self._mass_flow_fractions * m_flow
        # Fluid physical properties
        rho_f = self.fluid.density(T_f)
        mu_f = self.fluid.viscosity(T_f)
        k_f = self.fluid.conductivity(T_f)
        Pr = self.fluid.prandtl(T_f)
        # Convection thermal resistance in circular pipes
        h_fluid = convective_heat_transfer_coefficient_circular_pipe(
            m_flow_pipe,
            self.r_p_in,
            mu_f,
            rho_f,
            k_f,
            Pr,
            self.epsilon
            )
        R_f = 1 / (2 * np.pi * self.r_p_in * h_fluid)
        # Delta-circuit thermal resistances through grout
        R_fp = R_f + self.R_p
        R_d = self.multipole.thermal_resistances(self.k_s, self.k_b, R_fp)
        return R_d
