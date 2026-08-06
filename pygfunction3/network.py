# -*- coding: utf-8 -*-
from functools import cached_property, cache
from typing import Tuple, List

import numpy as np
from numpy.typing import ArrayLike

from .basis import Basis
from .borefield import Borefield
from .borehole import Borehole
from .fluid import Fluid
from .pipe import _Pipe


class Network(object):
    """
    Network of boreholes.

    Parameters
    ----------
    borefield : Borefield
        The borefield.
    pipe : Pipe
        The pipe.
    borehole_connectivity : (`n_boreholes`,) array-like of int, optional
        Index of fluid inlet into each borehole. ``-1`` corresponds to a
        borehole connected to the borefield inlet. If None, parallel
        connections between boreholes is used.
        Default is None.
    mass_flow_fractions : (`n_inlets`,) array-like, optional
        Mass flow fractions for each inlet. If None, equal mass flow fractions
        are used. Default is None.

    Attributes
    ----------
    fluid : Fluid
        Fluid used in the network.
    n_boreholes : int
        Number of boreholes in the network.
    n_circuits: int
        Number of circuits in the network.

    """
    def __init__(
            self, borefield: Borefield, pipe: _Pipe,
            borehole_connectivity: ArrayLike | None = None,
            mass_flow_fractions: ArrayLike | None = None):

        # --- Class atributes ---
        # Parameters
        self.borefield = borefield
        self.pipe = pipe
        self.borehole_connectivity = borehole_connectivity
        self.mass_flow_fractions = mass_flow_fractions
        # Additional attributes
        self.fluid = self.pipe.fluid
        self.n_boreholes = borefield.n_boreholes
        self.n_circuits = np.count_nonzero(
            self._borehole_connectivity(False) == -1)

    @cached_property
    def _borehole_circuits(self) -> np.ndarray:
        """Returns the circuit index of each borehole.

        Returns
        -------
        circuits : array
            (`n_boreholes`,) array of circuit indices.

        """
        circuits = np.full(self.n_boreholes, -1, dtype=int)
        for i in range(self.n_circuits):
            # Assign index i to the borehole connected to the outlet i
            outlet = self._network_outlets(False)[i]
            circuits[outlet] = i
            # Assign index i to all boreholes connected to the outlet i,
            # following the borehole connectivity
            c = self._borehole_connectivity(False)[outlet]
            while c != -1:
                circuits[c] = i
                c = self._borehole_connectivity(False)[c]
        return circuits

    @cached_property
    def _borehole_mass_flow_fractions(self) -> np.ndarray:
        """Returns the mass flow fractions of each borehole.

        Returns
        -------
        mass_flow_fractions : array
            (`n_boreholes`,) array of mass flow fractions.

        """
        return self._mass_flow_fractions[self._borehole_circuits]

    @cached_property
    def _mass_flow_fractions(self) -> np.ndarray:
        """Returns the mass flow fractions of each circuit.

        Returns
        -------
        mass_flow_fractions : array
            (`n_circuits`,) array of mass flow fractions.

        """
        if self.mass_flow_fractions is None:
            return np.full(self.n_circuits, 1.0 / self.n_circuits)
        else:
            return np.asarray(self.mass_flow_fractions, dtype=float)

    @cached_property
    def _unique_borehole_mass_flow_pairs(
            self) -> Tuple[List[Borehole], np.ndarray, np.ndarray]:
        """Returns unique (L, r_b, mass_flow_fraction) groups.

        Returns
        -------
        unique_boreholes : list of Borehole
            List of unique boreholes.
        unique_mass_flow_fractions : array
            Mass flow fractions of the unique boreholes.
        unique_indices : array
            Indices of the unique boreholes in the original borefield.

        """
        data = np.column_stack([
            self.borefield.L,
            self.borefield.r_b,
            self._borehole_mass_flow_fractions
        ])

        # Fast vectorized grouping (exact float matches).
        unique_rows, inverse = np.unique(data, axis=0, return_inverse=True)
        # Create unique Borehole objects for each unique (L, r_b) pair
        unique_boreholes = [
            Borehole(
                L=row[0],
                D=0.,
                r_b=row[1],
                p_head=np.zeros(2)
            )
            for row in unique_rows
        ]
        # Get the unique mass flow fractions for the unique boreholes
        unique_mass_flow_fractions = unique_rows[:, 2].copy()
        # Get the indices of the unique boreholes in the original borefield
        unique_indices = inverse.astype(int, copy=False)

        return unique_boreholes, unique_mass_flow_fractions, unique_indices

    @cache
    def _borehole_connectivity(self, reverse: bool) -> np.ndarray:
        """Returns the borehole connectivity.

        Parameters
        ----------
        reverse : bool
            If True, the flow direction is reversed.

        Returns
        -------
        borehole_connectivity : array
            (`n_boreholes`,) array of borehole connectivity.

        """
        if self.borehole_connectivity is None:
            return np.full(self.n_boreholes, -1, dtype=int)
        else:
            if reverse:
                # Reverse the borehole connectivity
                reverse_connectivity = np.full(self.n_boreholes, -1, dtype=int)
                for i in range(self.n_boreholes):
                    c = self.borehole_connectivity[i]
                    if c != -1:
                        reverse_connectivity[c] = i
                return reverse_connectivity
            else:
                return np.asarray(self.borehole_connectivity, dtype=int)

    @cache
    def _network_inlets(self, reverse: bool) -> np.ndarray:
        """Returns the indices of the network inlets.

        Parameters
        ----------
        reverse : bool
            If True, the flow direction is reversed.

        Returns
        -------
        inlets : array
            Indices of the network inlets.

        """
        inlets = np.flatnonzero(
            self._borehole_connectivity(reverse) == -1)
        return inlets

    @cache
    def _network_outlets(self, reverse: bool) -> np.ndarray:
        """Returns the indices of the network outlets.

        Parameters
        ----------
        reverse : bool
            If True, the flow direction is reversed.

        Returns
        -------
        outlets : array
            Indices of the network outlets.

        """
        outlets = np.flatnonzero(
            ~np.isin(
                np.arange(self.n_boreholes),
                self._borehole_connectivity(reverse)
                ))
        return outlets

    @cache
    def _top_continuity_coefficients(
            self, reverse: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Continuity condition at the top of the pipe.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_{f,b,in}} =
                \\mathbf{a_{in}} T_{f,n,in}
                + \\mathbf{a_{out}} \\mathbf{T_{f,b,out}}

        Parameters
        ----------
        reverse : bool
            If True, the flow direction is reversed.

        Returns
        -------
        a_in : array
            (`n_boreholes`,) array of inlet temperature coefficients.
        a_out : array
            (`n_boreholes`, `n_boreholes`,) array of fluid temperature
            coefficients.

        """
        # Initialize coefficients
        a_in = np.zeros(self.borefield.n_boreholes, dtype=int)
        a_out = np.zeros(
            (self.borefield.n_boreholes, self.borefield.n_boreholes),
            dtype=int
            )

        borehole_connectivity = self._borehole_connectivity(reverse)
        # Boreholes connected to the inlet of the network have a coefficient
        # of 1 for the inlet temperature
        a_in[np.asarray(borehole_connectivity) == -1] = 1
        # Other boreholes have a coefficient of 1 for the outlet temperature
        # of the previous borehole
        for i in range(self.borefield.n_boreholes):
            if borehole_connectivity[i] != -1:
                a_out[i, borehole_connectivity[i]] = 1
        return a_in, a_out

    def average_heat_extraction_rate(
            self, basis: Basis, m_flow: float, T_f_in: float, T_b: ArrayLike,
            T_f: float | None = None) -> np.ndarray:
        """Evaluate the average heat extraction rate of the boreholes.

        Parameters
        ----------
        basis : Basis
            Basis functions to model the profiles along the boreholes.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f_in : float
            Inlet fluid temperature (in degree celsius).
        T_b : array-like
            (`n_boreholes`, `n_nodes`,) array of borehole wall temperatures
            (in degree celsius).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.

        Returns
        -------
        q_b_avg : array
            (`n_boreholes`,) array of heat extraction rates (in W/m).

        """
        q_b_avg = self.total_heat_extraction_rate(
            basis, m_flow, T_f_in, T_b, T_f=T_f) / self.borefield.L
        return q_b_avg

    def effective_borefield_thermal_resistance(
            self, m_flow: float, T_f: float | None = None) -> float:
        """Evaluate the effective borefield thermal resistance.

        Parameters
        ----------
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.

        Returns
        -------
        R_field : float
            Effective borefield thermal resistance (in m-K/W).

        """
        # Coefficients for the total heat extraction rates:
        # Q = a_in * T_{f,n,in} + [a_b] * [T_b]
        basis = Basis(1)
        a_in, a_b = self._network_total_heat_extraction_rate_coefficients(
            basis, m_flow, T_f=T_f)

        # Coefficients for the outlet fluid temperature:
        # T_{f,n,out} = b_in * T_{f,n,in} + [b_b] * [T_b]
        b_in, b_b = self._network_outlet_temperature_coefficients(
            basis, m_flow, T_f=T_f)

        # Average fluid temperature:
        # T_f_avg = c_in * T_{f,n,in} + [c_b] * [T_b]
        c_in = 0.5 * (1 + b_in)

        # Effective borefield thermal resistance:
        # R_field = (T_b_eff - T_f_avg) / (Q / L)
        R_field = -c_in * self.borefield.L.sum() / a_in
        return R_field

    def effective_borehole_wall_temperature(
            self, basis: Basis, m_flow: float, T_f_in: float, T_b: ArrayLike,
            T_f: float | None = None) -> float:
        """Evaluate the effective borehole wall temperature.

        Parameters
        ----------
        basis : Basis
            Basis functions to model the profiles along the boreholes.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f_in : float
            Inlet fluid temperature (in degree celsius).
        T_b : array-like
            (`n_boreholes`, `n_nodes`,) array of borehole wall temperatures
            (in degree celsius).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.

        Returns
        -------
        T_b_eff : float
            Effective borehole wall temperature (in degree celsius).

        """
        # Outlet fluid temperature
        T_f_out = self.outlet_temperature(
            basis, m_flow, T_f_in, T_b, T_f=T_f)
        # Average fluid temperature
        T_f_avg = 0.5 * (T_f_in + T_f_out)
        # Average heat extraction rate of the network
        q_avg = self.network_average_heat_extraction_rate(
            basis, m_flow, T_f_in, T_b, T_f=T_f)
        # Borefield thermal resistance
        R_field = self.effective_borefield_thermal_resistance(
            m_flow, T_f=T_f)
        # Effective borehole wall temperature:
        T_b_eff = T_f_avg + R_field * q_avg
        return T_b_eff

    def heat_extraction_rate(
            self, basis: Basis, xi: float | ArrayLike, m_flow: float,
            T_f_in: float, T_b: ArrayLike, T_f: float | None = None
            ) -> np.ndarray:
        """Evaluate the heat extraction rate along the boreholes.

        Parameters
        ----------
        basis : Basis
            Basis functions to model the profiles along the boreholes.
        xi : float or array-like
            Dimensionless position along the borehole, or (``N``,) array of
            dimensionless positions.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f_in : float
            Inlet fluid temperature (in degree celsius).
        T_b : array-like
            (`n_boreholes`, `n_nodes`,) array of borehole wall temperatures
            (in degree celsius).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.

        Returns
        -------
        q_b : array
            (`n_boreholes`,)  or (`n_boreholes`, ``N``,) array of heat
            extraction rates (in W/m).

        """
        # Convert lists to arrays if necessary
        if isinstance(T_b, list):
            T_b = np.asarray(T_b)

        # Coefficients for the heat extraction rates:
        # [q_b](s) = [a_in] * T_{f,n,in} + [a_b] * [T_b]
        a_in, a_b = self._heat_extraction_rate_coefficients(
            basis, xi, m_flow, T_f=T_f)
        q_b = a_in * T_f_in + np.tensordot(a_b, T_b)
        return q_b

    def inlet_temperature(
            self, basis: Basis, m_flow: float, T_f_in: float,
            T_b: ArrayLike, T_f: float | None = None
            ) -> np.ndarray:
        """Evaluate the inlet fluid temperatures of the boreholes.

        Parameters
        ----------
        basis : Basis
            Basis functions to model the profiles along the boreholes.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f_in : float
            Inlet fluid temperature (in degree celsius).
        T_b : array-like
            (`n_boreholes`, `n_nodes`,) array of borehole wall temperatures
            (in degree celsius).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.

        Returns
        -------
        T_fb_in : array
            (`n_boreholes`,) array of inlet fluid temperatures (in degree
            celsius).

        """
        # Convert lists to arrays if necessary
        if isinstance(T_b, list):
            T_b = np.asarray(T_b)

        # Coefficients for the inlet fluid temperatures:
        # [T_{f,b,in}] = [a_in] * T_{f,n,in} + [a_b] * [T_b]
        a_in, a_b = self._inlet_temperature_coefficients(
            basis, m_flow, T_f=T_f)
        T_fb_in = a_in * T_f_in + np.tensordot(a_b, T_b)
        return T_fb_in

    def network_average_heat_extraction_rate(
            self, basis: Basis, m_flow: float, T_f_in: float, T_b: ArrayLike,
            T_f: float | None = None
            ) -> float:
        """Evaluate the average heat extraction rate of the network.

        Parameters
        ----------
        basis : Basis
            Basis functions to model the profiles along the boreholes.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f_in : float
            Inlet fluid temperature (in degree celsius).
        T_b : array-like
            (`n_boreholes`, `n_nodes`,) array of borehole wall temperatures
            (in degree celsius).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.

        Returns
        -------
        q_avg : float
            Average heat extraction rate of the network (in W/m).

        """
        # Convert lists to arrays if necessary
        if isinstance(T_b, list):
            T_b = np.asarray(T_b)

        # Coefficients for the total heat extraction rate:
        # Q = a_in * T_{f,n,in} + [a_b] * [T_b]
        a_in, a_b = self._network_total_heat_extraction_rate_coefficients(
            basis, m_flow, T_f=T_f)
        Q = a_in * T_f_in + np.tensordot(a_b, T_b)
        # Average heat extraction rate of the network
        q_avg = Q / self.borefield.L.sum()
        return q_avg

    def network_inlet_temperature(
            self, basis: Basis, m_flow: float, Q: float, T_b: ArrayLike,
            T_f: float | None = None
            ) -> float:
        """Evaluate the inlet fluid temperature of the network.

        Parameters
        ----------
        basis : Basis
            Basis functions to model the profiles along the boreholes.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        Q : float
            Total heat extraction rate of the network (in W).
        T_b : array-like
            (`n_boreholes`, `n_nodes`,) array of borehole wall temperatures
            (in degree celsius).

        Returns
        -------
        T_f_in : float
            Inlet fluid temperature of the network (in degree celsius).

        """
        # Convert lists to arrays if necessary
        if isinstance(T_b, list):
            T_b = np.asarray(T_b)

        # Coefficients for the total heat extraction rate:
        # Q = a_in * T_{f,n,in} + [a_b] * [T_b]
        a_in, a_b = self._network_total_heat_extraction_rate_coefficients(
            basis, m_flow, T_f=T_f)

        # Inlet fluid temperature of the network
        T_f_in = (Q - np.tensordot(a_b, T_b)) / a_in
        return T_f_in

    def network_outlet_temperature(
            self, basis: Basis, m_flow: float, T_f_in: float, T_b: ArrayLike,
            T_f: float | None = None
            ) -> float:
        """Evaluate the outlet fluid temperature of the network.

        Parameters
        ----------
        basis : Basis
            Basis functions to model the profiles along the boreholes.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f_in : float
            Inlet fluid temperature (in degree celsius).
        T_b : array-like
            (`n_boreholes`, `n_nodes`,) array of borehole wall temperatures
            (in degree celsius).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.

        Returns
        -------
        T_f_out : float
            Outlet fluid temperature of the network (in degree celsius).

        """
        # Convert lists to arrays if necessary
        if isinstance(T_b, list):
            T_b = np.asarray(T_b)

        # Coefficients for the outlet fluid temperature:
        # T_{f,n,out} = a_in * T_{f,n,in} + [a_b] * [T_b]
        a_in, a_b = self._network_outlet_temperature_coefficients(
            basis, m_flow, T_f=T_f)
        T_f_out = a_in * T_f_in + np.tensordot(a_b, T_b)
        return T_f_out

    def network_total_heat_extraction_rate(
            self, basis: Basis, m_flow: float, T_f_in: float, T_b: ArrayLike,
            T_f: float | None = None
            ) -> float:
        """Evaluate the total heat extraction rate of the network.

        Parameters
        ----------
        basis : Basis
            Basis functions to model the profiles along the boreholes.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f_in : float
            Inlet fluid temperature (in degree celsius).
        T_b : array-like
            (`n_boreholes`, `n_nodes`,) array of borehole wall temperatures
            (in degree celsius).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.

        Returns
        -------
        Q : float
            Total heat extraction rate of the network (in W).

        """
        # Coefficients for the total heat extraction rate:
        # Q = a_in * T_{f,n,in} + [a_b] * [T_b]
        a_in, a_b = self._network_total_heat_extraction_rate_coefficients(
            basis, m_flow, T_f=T_f)

        # Total heat extraction rate of the network:
        Q = a_in * T_f_in + np.tensordot(a_b, T_b)
        return Q

    def outlet_temperature(
            self, basis: Basis, m_flow: float, T_f_in: float,
            T_b: ArrayLike, T_f: float | None = None
            ) -> np.ndarray:
        """Evaluate the outlet fluid temperatures of the boreholes.

        Parameters
        ----------
        basis : Basis
            Basis functions to model the profiles along the boreholes.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f_in : float
            Inlet fluid temperature (in degree celsius).
        T_b : array-like
            (`n_boreholes`, `n_nodes`,) array of borehole wall temperatures
            (in degree celsius).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.

        Returns
        -------
        T_fb_out : array
            (`n_boreholes`,) array of outlet fluid temperatures (in degree
            celsius).

        """
        # Convert lists to arrays if necessary
        if isinstance(T_b, list):
            T_b = np.asarray(T_b)

        # Coefficients for the outlet fluid temperatures:
        # [T_{f,b,out}] = [a_in] * T_{f,n,in} + [a_b] * [T_b]
        a_in, a_b = self._outlet_temperature_coefficients(
            basis, m_flow, T_f=T_f)
        T_fb_out = a_in * T_f_in + np.tensordot(a_b, T_b)
        return T_fb_out


    def temperature(
            self, basis: Basis, xi: float | ArrayLike, m_flow: float,
            T_f_in: float, T_b: ArrayLike, T_f: float | None = None
            ) -> np.ndarray:
        """Evaluate the fluid temperature.

        Parameters
        ----------
        basis : Basis
            Basis functions to model the profiles along the boreholes.
        xi : float or array-like
            Dimensionless position along the borehole, or (``N``,) array of
            dimensionless positions.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f_in : float
            Inlet fluid temperature (in degree celsius).
        T_b : array-like
            (`n_boreholes`, `n_nodes`,) array of borehole wall temperatures
            (in degree celsius).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.

        Returns
        -------
        T_fb : array
            (`n_boreholes`,)  or (`n_boreholes`, ``N``,) array of fluid
            temperatures (in degree celsius).

        """
        # Convert lists to arrays if necessary
        if isinstance(T_b, list):
            T_b = np.asarray(T_b)

        # Coefficients for the fluid temperatures:
        # [T_fb](s) = [a_in] * T_{f,n,in} + [a_b] * [T_b]
        a_in, a_b = self._temperature_coefficients(
            basis, xi, m_flow, T_f=T_f)
        T_fb = a_in * T_f_in + np.tensordot(a_b, T_b)
        return T_fb

    def total_heat_extraction_rate(
            self, basis: Basis, m_flow: float, T_f_in: float, T_b: ArrayLike,
            T_f: float | None = None
            ) -> np.ndarray:
        """Evaluate the total heat extraction rate of the boreholes.

        Parameters
        ----------
        basis : Basis
            Basis functions to model the profiles along the boreholes.`,) array
            of dimensionless positions.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f_in : float
            Inlet fluid temperature (in degree celsius).
        T_b : array-like
            (`n_boreholes`, `n_nodes`,) array of borehole wall temperatures
            (in degree celsius).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.

        Returns
        -------
        Q_b : array
            (`n_boreholes`,) array of total heat extraction rates (in W).

        """
        # Convert lists to arrays if necessary
        if isinstance(T_b, list):
            T_b = np.asarray(T_b)

        # Coefficient matrices for total heat extraction rate:
        # [Q_b] = [a_in] * T_{f,n,in} + [a_b] * [T_b]
        a_in, a_b = self._total_heat_extraction_rate_coefficients(
            basis, m_flow, T_f=T_f)
        Q_b = a_in * T_f_in + np.tensordot(a_b, T_b)
        return Q_b

    def _heat_extraction_rate_coefficients(
            self, basis: Basis, xi: float | ArrayLike, m_flow: float,
            T_f: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the heat extraction rate coefficients of the boreholes.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{q_b}(s) = \\mathbf{a_{in}} T_{f,n,in}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        basis : Basis
            Basis functions to model the profiles along the boreholes.
        xi : float or array-like
            Dimensionless position along the boreholes, or (``N``,) array of
            dimensionless positions.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.

        Returns
        -------
        a_in : array
            (`n_boreholes`,) or (`n_boreholes`, ``N``,) array of inlet
            temperature coefficients.
        a_b : array
            (`n_boreholes`, `n_boreholes`, `n_nodes`,) or
            (`n_boreholes`, ``N``, `n_boreholes`, `n_nodes`,) array of
            borehole wall temperature coefficients.

        """
        # Coefficient matrices for borehole inlet temperatures:
        # [T_{f,b,in}] = [b_in] * T_{f,n,in} + [b_b] * [T_b]
        b_in, b_b = self._inlet_temperature_coefficients(
            basis, m_flow, T_f=T_f)

        # Coefficient matrices for borehole heat extraction rates:
        # [q_b](s) = [c_{b,in}] * [T_{f,b,in}] + [c_b] * [T_b]
        boreholes, mass_flow_fractions, indices = self._unique_borehole_mass_flow_pairs
        c_b_in_unique, c_b_unique = zip(*(
            self.pipe._heat_extraction_rate_coefficients(
                _b, basis, xi, m_flow * _fraction, T_f=T_f
            ) for (_b, _fraction) in zip(boreholes, mass_flow_fractions)
        ))
        c_b_in = np.stack([c_b_in_unique[i] for i in indices], axis=0)
        c_b = np.stack([c_b_unique[i] for i in indices], axis=0)

        # Final coefficient matrices for borehole heat extraction rates:
        # [q_b](s) = [a_in] * T_{f,n,in} + [a_b] * [T_b]
        a_in = (c_b_in.T * b_in).T
        a_b = np.einsum('i...,ijk->i...jk', c_b_in, b_b)
        a_b += np.einsum(
            'ij,i...k->i...jk',
            np.eye(self.n_boreholes, dtype=c_b.dtype),
            c_b)
        return a_in, a_b

    def _inlet_temperature_coefficients(
            self, basis: Basis, m_flow: float, T_f: float | None = None
            ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the inlet temperature coefficients of the boreholes.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_{f,b,in}} = \\mathbf{a_{in}} T_{f,n,in}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        basis : Basis
            Basis functions to model the profiles along the boreholes.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.

        Returns
        -------
        a_in: array
            (`n_boreholes`,) array of inlet temperature coefficients.
        a_b: array
            (`n_boreholes`, `n_boreholes`, `n_nodes`,) array of borehole wall
            temperature coefficients.

        """
        # Coefficient matrices for borehole inlet temperatures:
        # [T_{f,b,in}] = [b_in] * T_{f,n,in} + [b_out] * [T_{f,b,out}]
        b_in, b_out = self._top_continuity_coefficients(m_flow < 0.)

        # Coefficient matrices for borehole outlet temperatures:
        # [T_{f,b,out}] = [c_{b,in}] * [T_{f,b,in}] + [c_b] * [T_{b}]
        boreholes, mass_flow_fractions, indices = self._unique_borehole_mass_flow_pairs
        c_b_in_unique, c_b_unique = zip(*(
            self.pipe._outlet_temperature_coefficients(
                _b, basis, m_flow * _fraction, T_f=T_f
            ) for (_b, _fraction) in zip(boreholes, mass_flow_fractions)
        ))
        c_b_in = np.stack([c_b_in_unique[i] for i in indices], axis=0)
        c_b = np.stack([c_b_unique[i] for i in indices], axis=0)

        # Final coefficient matrices for borehole outlet temperatures:
        # [T_{f,b,in}] = [a_in] * T_{f,n,in} + [a_b] * [T_{b}]
        A = np.eye(self.n_boreholes) - b_out * c_b_in
        a_in = np.linalg.solve(A, b_in)
        rhs = (b_out[..., np.newaxis] * c_b).reshape(self.n_boreholes, -1)
        a_b = np.linalg.solve(A, rhs).reshape(
            self.n_boreholes, self.n_boreholes, -1)
        return a_in, a_b

    def _network_outlet_temperature_coefficients(
            self, basis: Basis, m_flow: float, T_f: float | None = None
            ) -> Tuple[float, np.ndarray]:
        """Evaluate the outlet temperature coefficients of the network.

        Returns coefficients for the relation:

            .. math::

                T_{f,n,out} = a_{in} T_{f,n,in} + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
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
            (`n_boreholes`, `n_nodes`,) array of borehole wall
            temperature coefficients.

        """
        # Coefficient matrices for borehole outlet temperatures:
        # [T_{f,b,out}] = [b_in] * T_{f,n,in} + [a_b] * [T_{b}]
        b_in, b_b = self._outlet_temperature_coefficients(
            basis, m_flow, T_f=T_f)

        # Final coefficient matrices for network outlet temperature:
        # T_{f,n,out} = a_in * T_{f,n,in} + [a_b] * [T_b]
        network_outlets = self._network_outlets(m_flow < 0.)
        a_in = self._mass_flow_fractions @ (b_in[network_outlets])
        a_b =  (b_b[network_outlets].T @ self._mass_flow_fractions).T
        return a_in, a_b

    def _network_total_heat_extraction_rate_coefficients(
            self, basis: Basis, m_flow: float, T_f: float | None = None
            ) -> Tuple[float, np.ndarray]:
        """Evaluate the total heat extraction rate coefficients of the network.

        Returns coefficients for the relation:

            .. math::

                Q = a_{in} T_{f,n,in} + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        basis : Basis
            Basis functions to model the profiles along the boreholes.
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
            (`n_boreholes`, `n_nodes`,) array of borehole wall
            temperature coefficients.

        """
        # Coefficient matrices for network outlet temperature:
        # T_{f,n,out} = b_in * T_{f,n,in} + [b_b] * [T_b]
        b_in, b_b = self._network_outlet_temperature_coefficients(
            basis, m_flow, T_f=T_f)

        # Fluid properties
        cp_f = self.pipe.fluid.specific_heat(T_f)

        # Final coefficient matrices for total heat extraction rate:
        # Q = a_in * T_{f,n,in} + [a_b] * [T_b]
        a_in = np.abs(m_flow) * cp_f * (b_in - 1)
        a_b = np.abs(m_flow) * cp_f * b_b
        return a_in, a_b

    def _outlet_temperature_coefficients(
            self, basis: Basis, m_flow: float, T_f: float | None = None
            ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the outlet temperature coefficients of the boreholes.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_{f,b,out}} = \\mathbf{a_{in}} T_{f,n,in}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        basis : Basis
            Basis functions to model the profiles along the borehole.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.

        Returns
        -------
        a_in: array
            (`n_boreholes`,) array of inlet temperature coefficients.
        a_b: array
            (`n_boreholes`, `n_boreholes`, `n_nodes`,) array of borehole wall
            temperature coefficients.

        """
        # Coefficient matrices for borehole inlet temperatures:
        # [T_{f,b,in}] = [b_in] * T_{f,n,in} + [b_out] * [T_{f,b,out}]
        b_in, b_out = self._top_continuity_coefficients(m_flow < 0.)

        # Coefficient matrices for borehole outlet temperatures:
        # [T_{f,b,out}] = [c_{b,in}] * [T_{f,b,in}] + [c_b] * [T_{b}]
        boreholes, mass_flow_fractions, indices = self._unique_borehole_mass_flow_pairs
        c_b_in_unique, c_b_unique = zip(*(
            self.pipe._outlet_temperature_coefficients(
                _b, basis, m_flow * _fraction, T_f=T_f
            ) for (_b, _fraction) in zip(boreholes, mass_flow_fractions)
        ))
        c_b_in = np.stack([c_b_in_unique[i] for i in indices], axis=0)
        c_b = np.stack([c_b_unique[i] for i in indices], axis=0)

        # Final coefficient matrices for borehole outlet temperatures:
        # [T_{f,b,out}] = [a_in] * T_{f,n,in} + [a_b] * [T_{b}]
        A = np.eye(self.n_boreholes) - c_b_in[:, np.newaxis] * b_out
        a_in = np.linalg.solve(A, c_b_in * b_in)
        # Embed each borehole's wall-temperature coefficients in block form:
        # RHS has shape (n_boreholes, n_boreholes * n_nodes).
        rhs = np.einsum(
            'ij,ik->ijk',
            np.eye(self.n_boreholes, dtype=c_b.dtype),
            c_b).reshape(self.n_boreholes, -1)
        a_b = np.linalg.solve(A, rhs).reshape(
            self.n_boreholes, self.n_boreholes, -1)
        return a_in, a_b

    def _temperature_coefficients(
            self, basis: Basis, xi: float | ArrayLike, m_flow: float,
            T_f: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the temperature coefficients of the boreholes.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_{f,b}}(s) = \\mathbf{a_{in}} T_{f,in}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        basis : Basis
            Basis functions to model the profiles along the boreholes.
        xi : float or array-like
            Dimensionless position along the boreholes, or (``N``,) array of
            dimensionless positions.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.

        Returns
        -------
        a_in : array
            (`n_boreholes`, `n_pipes`,) or (`n_boreholes`, ``N``, `n_pipes`,)
            array of inlet temperature coefficients.
        a_b : array
            (`n_boreholes`, `n_pipes`, `n_boreholes`, `n_nodes`,) or
            (`n_boreholes`, ``N``, `n_pipes`, `n_boreholes`, `n_nodes`,) array
            of borehole wall temperature coefficients.

        """
        # Coefficient matrices for borehole inlet temperatures:
        # [T_{f,b,in}] = [b_in] * T_{f,n,in} + [b_b] * [T_b]
        b_in, b_b = self._inlet_temperature_coefficients(
            basis, m_flow, T_f=T_f)

        # Coefficient matrices for borehole heat extraction rates:
        # [T_{f}](s) = [c_{b,in}] * [T_{f,b,in}] + [c_b] * [T_b]
        boreholes, mass_flow_fractions, indices = self._unique_borehole_mass_flow_pairs
        c_b_in_unique, c_b_unique = zip(*(
            self.pipe._temperature_coefficients(
                _b, basis, xi, m_flow * _fraction, T_f=T_f
            ) for (_b, _fraction) in zip(boreholes, mass_flow_fractions)
        ))
        c_b_in = np.stack([c_b_in_unique[i] for i in indices], axis=0)
        c_b = np.stack([c_b_unique[i] for i in indices], axis=0)

        # Final coefficient matrices for borehole heat extraction rates:
        # [T_{f}](s) = [a_in] * T_{f,n,in} + [a_b] * [T_b]
        a_in = (c_b_in.T * b_in).T
        a_b = np.einsum('i...n,ijk->i...njk', c_b_in, b_b)
        a_b += np.einsum(
            'ij,i...nk->i...njk',
            np.eye(self.n_boreholes, dtype=c_b.dtype),
            c_b)
        return a_in, a_b

    def _total_heat_extraction_rate_coefficients(
            self, basis: Basis, m_flow: float, T_f: float | None = None
            ) -> Tuple[float, np.ndarray]:
        """Evaluate the total heat extraction rate coefficients of the boreholes.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{Q_b} = \\mathbf{a_{in}} T_{f,n,in}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        basis : Basis
            Basis functions to model the profiles along the boreholes.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f : float or None, default: None
            Fluid temperature (in degree celsius) to evaluate fluid properties.
            If ``None``, the nominal fluid temperature is used.

        Returns
        -------
        a_in: array
            (`n_boreholes`,) array of inlet temperature coefficients.
        a_b: array
            (`n_boreholes`, `n_boreholes`, `n_nodes`,) array of borehole wall
            temperature coefficients.

        """
        # Coefficient matrices for outlet temperatures:
        # [T_{f,b,out}] = [b_in] * T_{f,n,in} + [b_b] * [T_b]
        b_in, b_b = self._outlet_temperature_coefficients(
            basis, m_flow, T_f=T_f)

        # Fluid properties
        cp_f = self.fluid.specific_heat(T_f)

        # Final coefficient matrices for total heat extraction rate:
        # [Q_b] = [a_in] * T_{f,n,in} + [a_b] * [T_b]
        mass_flow_fractions = self._borehole_mass_flow_fractions
        a_in = np.abs(m_flow) * mass_flow_fractions * cp_f * (b_in - 1)
        a_b = (np.abs(m_flow) * mass_flow_fractions * cp_f * b_b.T).T
        return a_in, a_b
