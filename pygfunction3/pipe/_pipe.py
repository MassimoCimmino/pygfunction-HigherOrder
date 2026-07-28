# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from functools import cached_property, lru_cache
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad_vec
from scipy.linalg import expm

from ..basis import Basis
from ..borehole import Borehole


class _Pipe(ABC):
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

    @abstractmethod
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
        R_d : array
            (`n_pipes`, `n_pipes`,) array of delta-circuit thermal
            resistances (in m-K/W).

        """
        ...

    @cached_property
    @abstractmethod
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
        ...

    @cached_property
    @abstractmethod
    def _direction_coefficients(self) -> np.ndarray:
        """Direction coefficients of the pipe.

        Coefficients are positive for downward-flowing pipes and negative for
        upward-flowing pipes.

        Returns
        -------
        array
            (`n_pipes`,) array of direction coefficients.

        """
        ...

    @cached_property
    @abstractmethod
    def _mass_flow_fractions(self) -> np.ndarray:
        """Mass flow rate factor of the pipe.

        Returns
        -------
        array
            (`n_pipes`,) array of mass flow rate fractions, normalized to the
            total mass flow rate.

        """
        ...

    @cached_property
    @abstractmethod
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
        ...

    @cached_property
    @abstractmethod
    def _pipe_configuration_coefficients(
        self) -> Tuple[np.ndarray, np.ndarray]:
        """Configuration coefficients to reconstruct the temperature vector.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_{f}}(s) = \\mathbf{a_{fd}} \\mathbf{T_{fd}}(s)
                + \\mathbf{a_{fu}} \\mathbf{T_{fu}}(s)

        Returns
        -------
        a_fd : array
            (`n_pipes`, `n_pipes_down`,) array of inlet temperature
            coefficients.
        a_fu : array
            (`n_pipes`, `n_pipes_up`,) array of fluid temperature coefficients.

        """
        ...

    @cached_property
    @abstractmethod
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
        ...

    def average_heat_extraction_rate(
            self, borehole: Borehole, basis: Basis, m_flow: float,
            T_f_in: float, T_b: ArrayLike, T_f: float | None = None) -> float:
        """Evaluate the average heat extraction rate.

        Parameters
        ----------
        borehole : Borehole
            Borehole containing the pipe.
        basis : Basis
            Basis functions to model the profiles along the borehole.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f_in : float
            Inlet fluid temperature (in degree celsius).
        T_b : array-like
            (`n_nodes`,) array of borehole wall temperatures (in degree
            celsius).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.

        Returns
        -------
        q_avg : float
            Average heat extraction rate (in W/m).

        """
        Q = self.total_heat_extraction_rate(
            borehole, basis, m_flow, T_f_in, T_b, T_f=T_f)
        return Q / borehole.L

    def effective_borehole_thermal_resistance(
            self, borehole: Borehole, m_flow: float, T_f: float | None = None
            ) -> float:
        """Evaluate the effective borehole thermal resistance.

        Parameters
        ----------
        borehole : Borehole
            Borehole containing the pipe.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.

        Returns
        -------
        R_b_eff : float
            Effective borehole thermal resistance (in m-K/W).

        """
        # Coefficients for the total heat extraction rates:
        # Q = a_in * T_{f,in} + [a_b] * [T_b]
        basis = Basis(1)
        a_in, a_b = self._total_heat_extraction_rate_coefficients(
            borehole, basis, m_flow, T_f=T_f)

        # Coefficients for the outlet fluid temperature:
        # T_{f,out} = b_in * T_{f,in} + [b_b] * [T_b]
        b_in, b_b = self._outlet_temperature_coefficients(
            borehole, basis, m_flow, T_f=T_f)

        # Average fluid temperature:
        # T_f_avg = c_in * T_{f,in} + [c_b] * [T_b]
        c_in = 0.5 * (1 + b_in)

        # Effective borehole thermal resistance:
        # R_b_eff = (T_b_eff - T_f_avg) / (Q / L)
        R_b_eff = -c_in * borehole.L / a_in
        return R_b_eff

    def effective_borehole_wall_temperature(
            self, borehole: Borehole, basis: Basis, m_flow: float,
            T_f_in: float, T_b: ArrayLike, T_f: float | None = None) -> float:
        """Evaluate the effective borehole wall temperature.

        Parameters
        ----------
        borehole : Borehole
            Borehole containing the pipe.
        basis : Basis
            Basis functions to model the profiles along the borehole.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f_in : float
            Inlet fluid temperature (in degree celsius).
        T_b : array-like
            (`n_nodes`,) array of borehole wall temperatures (in degree celsius).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.

        Returns
        -------
        T_b_eff : float
            Effective borehole wall temperature (in degree celsius).

        """
        T_f_out = self.outlet_temperature(
            borehole, basis, m_flow, T_f_in, T_b, T_f=T_f)
        # Average fluid temperature:
        T_f_avg = 0.5 * (T_f_in + T_f_out)
        q_avg = self.average_heat_extraction_rate(
            borehole, basis, m_flow, T_f_in, T_b, T_f=T_f)
        R_b_eff = self.effective_borehole_thermal_resistance(
            borehole, m_flow, T_f=T_f)
        # Effective borehole wall temperature:
        T_b_eff = T_f_avg + R_b_eff * q_avg
        return T_b_eff

    def heat_extraction_rate(
            self, borehole: Borehole, basis: Basis, xi: float | ArrayLike,
            m_flow: float, T_f_in: float, T_b: ArrayLike,
            T_f: float | None = None) -> float | np.ndarray:
        """Evaluate the heat extraction rate.

        Parameters
        ----------
        borehole : Borehole
            Borehole containing the pipe.
        basis : Basis
            Basis functions to model the profiles along the borehole.
        xi : float or array-like
            Dimensionless position along the borehole, or (``N``,) array of
            dimensionless positions.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f_in : float
            Inlet fluid temperature (in degree celsius).
        T_b : array-like
            (`n_nodes`,) array of borehole wall temperatures (in degree
            celsius).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.

        Returns
        -------
        float or array
            float  or (``N``,) array of heat extraction rates (in W/m).

        """
        # Convert lists to arrays if necessary
        if isinstance(T_b, list):
            T_b = np.asarray(T_b)

        # Coefficients for the heat extraction rates:
        # q(s) = a_in * T_{f,in} + [a_b] * [T_b]
        a_in, a_b = self._heat_extraction_rate_coefficients(
            borehole, basis, xi, m_flow, T_f=T_f)
        q = a_in * T_f_in + a_b @ T_b
        return q

    def inlet_temperature(
            self, borehole: Borehole, basis: Basis, m_flow: float, Q: float,
            T_b: ArrayLike, T_f: float | None = None) -> float:
        """Evaluate the inlet fluid temperature.

        Parameters
        ----------
        borehole : Borehole
            Borehole containing the pipe.
        basis : Basis
            Basis functions to model the profiles along the borehole.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        Q : float
            Total heat extraction rate (in W).
        T_b : array-like
            (`n_nodes`,) array of borehole wall temperatures (in degree
            celsius).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.

        Returns
        -------
        T_f_in : float
            Inlet fluid temperature (in degree celsius).

        """
        # Convert lists to arrays if necessary
        if isinstance(T_b, list):
            T_b = np.asarray(T_b)

        # Coefficients for the total heat extraction rates:
        # Q = a_in * T_{f,in} + [a_b] * [T_b]
        a_in, a_b = self._total_heat_extraction_rate_coefficients(
            borehole, basis, m_flow, T_f=T_f)

        # Inlet fluid temperature:
        T_f_in = (Q - a_b @ T_b) / a_in
        return T_f_in

    def m_flow_pipe(self, m_flow: float) -> np.ndarray:
        """Evaluate the mass flow rate in each pipe.

        Parameters
        ----------
        m_flow : float
            Fluid mass flow rate (in kg/s).

        Returns
        -------
        m_flow_pipe : array
            (`n_pipes`,) array of mass flow rates in each pipe (in kg/s).

        """
        return m_flow * self._mass_flow_fractions

    def temperature(
            self, borehole: Borehole, basis: Basis, xi: float | ArrayLike,
            m_flow: float, T_f_in: float, T_b: ArrayLike,
            T_f: float | None = None) -> np.ndarray:
        """Evaluate the fluid temperature.

        Parameters
        ----------
        borehole : Borehole
            Borehole containing the pipe.
        basis : Basis
            Basis functions to model the profiles along the borehole.
        xi : float or array-like
            Dimensionless position along the borehole, or (``N``,) array of
            dimensionless positions.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f_in : float
            Inlet fluid temperature (in degree celsius).
        T_b : array-like
            (`n_nodes`,) array of borehole wall temperatures (in degree
            celsius).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.

        Returns
        -------
        array
            (`n_pipes`, `n_nodes`,) or (``N``, `n_pipes`, `n_nodes`,) array of
            fluid temperatures (in degree celsius).

        """
        # Convert lists to arrays if necessary
        if isinstance(T_b, list):
            T_b = np.asarray(T_b)

        # Coefficients for the fluid temperatures:
        # [T_f](s) = [b_in] * T_{f,in} + [b_b] * [T_b]
        b_in, b_b = self._temperature_coefficients(
            borehole, basis, xi, m_flow, T_f=T_f)
        T_f = b_in * T_f_in + b_b @ T_b
        return T_f

    def total_heat_extraction_rate(
            self, borehole: Borehole, basis: Basis, m_flow: float,
            T_f_in: float, T_b: ArrayLike, T_f: float | None = None
            ) -> float:
        """Evaluate the total heat extraction rate.

        Parameters
        ----------
        borehole : Borehole
            Borehole containing the pipe.
        basis : Basis
            Basis functions to model the profiles along the borehole.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f_in : float
            Inlet fluid temperature (in degree celsius).
        T_b : array-like
            (`n_nodes`,) array of borehole wall temperatures (in degree
            celsius).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.

        Returns
        -------
        float
            Total heat extraction rate (in W).

        """
        # Coefficients for the total heat extraction rates:
        # Q = a_in * T_{f,in} + [a_b] * [T_b]
        a_in, a_b = self._total_heat_extraction_rate_coefficients(
            borehole, basis, m_flow, T_f=T_f)
        Q = a_in * T_f_in + a_b @ T_b
        return Q

    def outlet_temperature(
            self, borehole: Borehole, basis: Basis, m_flow: float,
            T_f_in: float, T_b: ArrayLike, T_f: float | None = None) -> float:
        """Evaluate the outlet fluid temperature.

        Parameters
        ----------
        borehole : Borehole
            Borehole containing the pipe.
        basis : Basis
            Basis functions to model the profiles along the borehole.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f_in : float
            Inlet fluid temperature (in degree celsius).
        T_b : array-like
            (`n_nodes`,) array of borehole wall temperatures (in degree
            celsius).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.

        Returns
        -------
        T_f_out : float
            Outlet fluid temperature (in degree celsius).

        """
        # Convert lists to arrays if necessary
        if isinstance(T_b, list):
            T_b = np.asarray(T_b)

        # Coefficients for the outlet fluid temperature:
        # T_{f,out} = b_in * T_{f,in} + [b_b] * [T_b]
        b_in, b_b = self._outlet_temperature_coefficients(
            borehole, basis, m_flow, T_f=T_f)
        T_f_out = b_in * T_f_in + b_b @ T_b
        return T_f_out

    def _total_heat_extraction_rate_coefficients(
            self, borehole: Borehole, basis: Basis, m_flow: float,
            T_f: float | None = None) -> Tuple[float, np.ndarray]:
        """Evaluate the total heat extraction rate coefficients of the pipe.

        Returns coefficients for the relation:

            .. math::

                Q = a_{in} T_{f,in} + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        borehole : Borehole
            Borehole containing the pipe.
        basis : Basis
            Basis functions to model the profiles along the borehole.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.

        Returns
        -------
        a_in : array
            Inlet temperature coefficient.
        a_b : array
            (`n_nodes`,) array of borehole wall temperature coefficients.

        """
        # Coefficients for the outlet fluid temperature:
        # T_{f,out} = b_in * T_{f,in} + [b_b] * [T_b]
        b_in, b_b = self._outlet_temperature_coefficients(
            borehole, basis, m_flow, T_f=T_f)

        # Fluid properties
        cp_f = self.fluid.specific_heat(T_f)

        # Final coefficient matrices for total heat extraction rates:
        # Q = a_in * T_{f,in} + [a_b] * [T_b]
        a_in = m_flow * cp_f * (b_in - 1)
        a_b = m_flow * cp_f * b_b
        return a_in, a_b

    def _heat_extraction_rate_coefficients(
            self, borehole: Borehole, basis: Basis, xi: float | ArrayLike,
            m_flow: float, T_f: float | None = None
            ) -> Tuple[float | np.ndarray, np.ndarray]:
        """Evaluate the heat extraction rate coefficients of the pipe.

        Returns coefficients for the relation:

            .. math::

                q(s) = a_{in} T_{f,in} + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        borehole : Borehole
            Borehole containing the pipe.
        basis : Basis
            Basis functions to model the profiles along the borehole.
        xi : float or array-like
            Dimensionless position along the borehole, or (``N``,) array of
            dimensionless positions.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.

        Returns
        -------
        a_in : float or array
            Heat extraction rate coefficient or (``N``) array of heat
            extraction rate coefficients.
        a_b : array
            (`n_nodes`,) or (``N``, `n_nodes`,) array of borehole wall
            temperature coefficients.

        """
        # Coefficients for fluid temperatures:
        # [T_f](s) = [b_in] * T_{f,in} + [b_b] * [T_b]
        b_in, b_b = self._temperature_coefficients(
            borehole, basis, xi, m_flow, T_f=T_f)

        # Thermal conductances of the pipe (in W/m.K)
        G = 1 / self.thermal_resistances(m_flow, T_f=T_f)
        G_diag = np.diag(G)
        G_trace = np.trace(G)

        # Final coefficient matrices for heat extraction rates [W/m]:
        # q(s) = a_in * T_{f,in} + [a_b] * [T_b]
        a_in = -b_in @ G_diag
        psi = basis.basis_function(xi)
        a_b = -np.einsum('i,...ij->...j', G_diag, b_b) + G_trace * psi
        return a_in, a_b

    @lru_cache(maxsize=8, typed=False)
    def _ode_coefficients(
        self, m_flow: float, T_f: float | None = None) -> np.ndarray:
        """Evaluate ODE coefficient of the pipe.

        Returns The ODE coefficient A for the relation:

            .. math::

                \\frac{d T_{f}}{d s} = \\mathbf{A} \\mathbf{T_{f}}(s)
                + \\mathbf{b} \\mathbf{T_{b}}(s)

        Parameters
        ----------
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.

        Returns
        -------
        A : array
            (`n_pipes`, `n_pipes`,) array of ODE coefficients.

        """
        # Delta-circuit thermal conductances
        R = self.thermal_resistances(m_flow, T_f=T_f)
        A = 1 / R
        # Set diagonal entries such that row sums are zero
        diag_indices = np.diag_indices_from(A)
        A[diag_indices] = -np.sum(A, axis=-1)
        # Multiply by direction coefficients
        c = self._direction_coefficients
        A = A * c[:, None]
        # Divide by heat capacity rate of the fluid
        c_p = self.fluid.specific_heat(T_f)
        m_flow_pipe = self.m_flow_pipe(m_flow)
        A = A / (m_flow_pipe * c_p)[:, None]
        return A

    def _outlet_temperature_coefficients(
            self, borehole: Borehole, basis: Basis, m_flow: float,
            T_f: float | None = None) -> Tuple[float, np.ndarray]:
        """Evaluate the outlet temperature coefficients of the pipe.

        Parameters
        ----------
        borehole : Borehole
            Borehole containing the pipe.
        basis : Basis
            Basis functions to model the profiles along the borehole.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.

        Returns
        -------
        a_in: float
            Inlet temperature coefficient.
        a_b: array
            (`n_nodes`,) array of borehole wall temperature coefficients.

        """
        # Coefficients for upward temperatures at s=0:
        # [T_fu](s=0) = [b_in] * T_{f,in} + [b_b] * [T_b]
        b_in, b_b = self._upward_top_temperature_coefficients(
            borehole, basis, m_flow, T_f=T_f)

        # Mixing coefficients:
        # T_out = [c_fu] * [T_fu](s=0)
        c_fu = self._mixing_coefficients

        # Final coefficient matrices for outlet temperatures:
        # T_{f,out} = a_in * T_{f,in} + [a_b] * [T_b]
        a_in = c_fu @ b_in
        a_b = c_fu @ b_b
        return a_in, a_b

    def _state_transition_matrix(
            self, borehole: Borehole, xi: float | ArrayLike, m_flow: float,
            T_f: float | None = None, xi_0: float | None = None) -> np.ndarray:
        """State transition matrix of the pipe.

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
            Starting dimensionless position along the borehole. If ``None``,
            the starting position is assumed to be -1.

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
        phi = expm(np.multiply.outer(s, A))
        return phi

    # TODO : Reuse the _state_transition_matrix function to avoid code duplication
    def _state_transition_matrix_b(
            self, borehole: Borehole, basis: Basis, xi: float | ArrayLike,
            m_flow: float, T_f: float | None = None, xi_0: float | None = None
            ) -> np.ndarray:
        """State transition matrix of the pipe weighted by the ODE vector.

        Parameters
        ----------
        borehole : Borehole
            Borehole containing the pipe.
        basis : Basis
            Basis functions to model the profiles along the borehole.
        xi : float or array-like
            Dimensionless position along the borehole, or (``N``,) array of
            dimensionless positions.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.
        xi_0 : float or None, default: None
            Starting dimensionless position along the borehole. If ``None``,
            the starting position is assumed to be -1.

        Returns
        -------
        phi_b : array
            (`n_pipes`,) or (``N``, `n_pipes`,) array of weighted state
            transition matrix.

        """
        phi = self._state_transition_matrix(
            borehole, xi, m_flow, T_f=T_f, xi_0=xi_0)
        b = -self._ode_coefficients(m_flow, T_f=T_f).sum(axis=-1)
        return phi @ b

    def _temperature_coefficients(
            self, borehole: Borehole, basis: Basis, xi: float | ArrayLike,
            m_flow: float, T_f: float | None = None
            ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the temperature coefficients of the pipe.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_{f}}(s) = \\mathbf{a_{in}} T_{f,in}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        borehole : Borehole
            Borehole containing the pipe.
        basis : Basis
            Basis functions to model the profiles along the borehole.
        xi : float or array-like
            Dimensionless position along the borehole, or (``N``,) array of
            dimensionless positions.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.

        Returns
        -------
        a_in : array
            (`n_pipes`,) or (``N``, `n_pipes`,) array of fluid temperature
            coefficients.
        a_b : array
            (`n_pipes`, `n_nodes`,) or (``N``, `n_pipes`, `n_nodes`,) array of
            borehole wall temperature coefficients.

        """
        # Coefficients for upward temperatures at s=0:
        # [T_fu](s=0) = [b_in] * T_{f,in} + [b_b] * [T_b]
        b_in, b_b = self._upward_top_temperature_coefficients(
            borehole, basis, m_flow, T_f=T_f)

        # Continuity condition (top of the pipe):
        # [T_fd](s=0) = [c_in] * T_{f,in} + [c_fu] * [T_fu](s=0)
        c_in, c_fu = self._top_continuity_coefficients

        # Pipe configuration:
        # [T_f](s) = [d_fd] * [T_fd](s) + [d_fu] * [T_fu](s)
        d_fd, d_fu = self._pipe_configuration_coefficients

        # General solution for temperatures:
        # [T_f](s) = [e_f0]*[T_f](s=0) + [e_b] * [T_b]
        e_f0, e_b = self._zero_response_coefficients(borehole, basis, xi, m_flow, T_f=T_f)

        # Final coefficient matrices for temperatures:
        # [T_f](s) = [a_in] * T_{f,in} + [a_b] * [T_b]
        A = d_fu + d_fd @ c_fu
        a_in = e_f0 @ (d_fd @ c_in + A @ b_in)
        a_b = e_f0 @ A @ b_b + e_b
        return a_in, a_b

    def _upward_top_temperature_coefficients(
            self, borehole: Borehole, basis: Basis, m_flow: float,
            T_f: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the upward top temperature coefficients of the pipe.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_{f,u}}(s) = \\mathbf{a_{in}} T_{f,in}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        borehole : Borehole
            Borehole containing the pipe.
        basis : Basis
            Basis functions to model the profiles along the borehole.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.

        Returns
        -------
        a_in : array
            (`n_pipes_up`,) array of inlet temperature coefficients.
        a_b : array
            (`n_pipes_up`, `n_nodes`,) array of borehole wall temperature
            coefficients.

        """
        # General solution:
        # [T_f](s=H) = [b_f0] * [T_f](s=0) + [b_b] * [T_b]
        b_f0, b_b = self._zero_response_coefficients(
            borehole, basis, 1., m_flow, T_f=T_f)

        # Continuity conditions (top of the pipe):
        # [T_fd](s=0) = [c_in]*[T_in] + [c_fu]*[T_fu](s=0)

        c_in, c_fu = self._top_continuity_coefficients
        # Continuity conditions (bottom of the pipe):
        # [0]_u = [d_f]*[T_f](s=H)
        d_f = self._bottom_continuity_coefficients

        # Pipe configuration:
        # [T_f](s) = [f_fd] * [T_fd](s) + [f_fu] * [T_fu](s)
        e_fd, e_fu = self._pipe_configuration_coefficients

        # Upward temperatures at s=0:
        # [T_fu](s=0) = [a_in] * T_{f,in} + [a_b] * [T_b]
        A = e_fu + e_fd @ c_fu
        B = np.linalg.solve(d_f @ b_f0 @ A, -d_f)
        a_in = B @ b_f0 @ e_fd @ c_in
        a_b = B @ b_b

        return a_in, a_b

    def _zero_input_response_coefficients(
            self, borehole: Borehole, xi: float | ArrayLike, m_flow: float,
            T_f: float | None = None) -> np.ndarray:
        """Evaluate the zero-input response coefficients of the pipe.

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

        Returns
        -------
        a_f0 : array
            (`n_pipes`,) or (``N``, `n_pipes`,) array of zero-input response
            coefficients.

        """
        a_f0 = self._state_transition_matrix(borehole, xi, m_flow, T_f=T_f)
        return a_f0

    def _zero_response_coefficients(
            self, borehole: Borehole, basis: Basis, xi: float | ArrayLike,
            m_flow: float, T_f: float | None = None,
            backend: str = 'augmented-expm') -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the zero-response coefficients of the pipe.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_{f}}(s) = \\mathbf{a_{f0}} \\mathbf{T_{f}}(s=0)
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        borehole : Borehole
            Borehole containing the pipe.
        basis : Basis
            Basis functions to model the profiles along the borehole.
        xi : float or array-like
            Dimensionless position along the borehole, or (``N``,) array of
            dimensionless positions.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.
        backend : str, default: 'augmented-expm'
            Backend to compute the zero-response coefficients. Options are:
            'augmented-expm' (default) or 'integral'.

        Returns
        -------
        a_f0 : array
            (`n_pipes`,) or (``N``, `n_pipes`,) array of zero-input response
            coefficients.
        a_b : array
            (`n_pipes`, `n_nodes`,) or (``N``, `n_pipes`, `n_nodes`,) array of
            zero-state response coefficients.

        """
        if backend == 'integral':
            a_f0 = self._zero_input_response_coefficients(
                borehole, xi, m_flow, T_f=T_f)
            a_b = self._zero_state_response_coefficients(
                borehole, basis, xi, m_flow, T_f=T_f)
        else:
            # ODE coefficients for the pipe:
            A = self._ode_coefficients(m_flow, T_f=T_f) * borehole.L / 2
            b = -A.sum(axis=-1)
            # Augmented matrix for the zero-response coefficients:
            J = np.eye(basis.n_nodes, k=1)
            K = np.block([
                [A, b[:, None], np.zeros((self.n_pipes, basis.n_nodes-1))],
                [np.zeros((basis.n_nodes, self.n_pipes)), J],
            ])
            # Compute the matrix exponential of the augmented matrix
            E = expm(np.multiply.outer(xi + 1, K))
            # Compute the zero-response coefficients from the augmented matrix
            # exponential
            psi_coefs = basis._shifted_coefficients(shift=-1.)
            R = np.cumulative_prod(
                np.arange(1, basis.n_nodes),
                include_initial=True
                )
            a_f0 = E[..., :self.n_pipes, :self.n_pipes]
            a_b = E[..., :self.n_pipes, -basis.n_nodes:] * R @ psi_coefs
        return a_f0, a_b

    def _zero_state_response_coefficients(
            self, borehole: Borehole, basis: Basis, xi: float | ArrayLike,
            m_flow: float, T_f: float | None = None) -> np.ndarray:
        """Evaluate the zero-state response coefficients of the pipe.

        Parameters
        ----------
        borehole : Borehole
            Borehole containing the pipe.
        basis : Basis
            Basis functions to model the profiles along the borehole.
        xi : float or array-like
            Dimensionless position along the borehole, or (``N``,) array of
            dimensionless positions.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.

        Returns
        -------
        array
            (`n_pipes`, `n_nodes`,) or (``N``, `n_pipes`, `n_nodes`,) array of
            zero-state response coefficients.

        """
        def phi_b(_xi: float) -> np.ndarray:
            return self._state_transition_matrix_b(
                borehole, basis, xi, m_flow, T_f=T_f, xi_0=_xi)
        L = borehole.L
        def integrand(_eta: float) -> np.ndarray:
            _xi = (_eta + 1) * (xi + 1) / 2 - 1
            _phi_b = phi_b(_xi)
            _psi = basis.basis_function(_xi)
            _integrand = _phi_b[..., None] * _psi[..., None, :] * L / 2
            return _integrand * (xi + 1)[..., None, None] / 2
        a_b = quad_vec(integrand, -1, 1., epsabs=1e-6, epsrel=1e-6)[0]
        return a_b
