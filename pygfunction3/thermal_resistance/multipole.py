# -*- coding: utf-8 -*-
from functools import cached_property, lru_cache
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import comb


class Multipole:
    """Multipole method for 2d heat transfer.

    Parameters
    ----------
    r_b : float
        Borehole radius (in meters).
    r_p : array_like
        Outer radius of the pipes (in meters).
    p : array_like
        (`n_pipes`, 2) array of pipe positions (in meters). The position
        ``(0, 0)`` corresponds to the axis of the borehole.
    J : int, default: 3
        Order of the multipole solution.

    Attributes
    ----------
    n_pipes : int
        Number of pipes.
    sigma : float
        Dimensionless parameter for the two thermal conductivities.
    z : complex array
        (`n_pipes`,) complex array of pipe positions (in meters).

    """

    def __init__(self, r_b: float, r_p: ArrayLike, p: ArrayLike, J: int = 3):
        # Convert input to numpy.Array
        r_p = np.asarray(r_p)
        p = np.atleast_2d(p)

        # --- Class atributes ---
        # Parameters
        self.r_b = r_b
        self.r_p = r_p
        self.p = p
        self.J = J
        # Additional attributes
        self.n_pipes = len(r_p)
        # self.sigma = (k_b - k_s) / (k_b + k_s)
        self.z = p[:, 0] + 1.j * p[:, 1]
        self._zz = np.multiply.outer(self.z.conj(), self.z)
        self._dis = np.subtract.outer(self.z, self.z)

    def borehole_wall_temperatures(self, theta: float | np.ndarray, q: np.ndarray, T_b: float, k_s: float, k_b: float, R_fp: np.ndarray) -> float:
        """Evaluate borehole wall temperature.

        Parameters
        ----------
        theta : float | np.ndarray
            Polar angle (in radians).
        q : array
            (`n_pipes`,) array of heat injection rates in each pipe
            (in W/m).
        T_b : float
            Average borehole wall temperature (in degree Celsius).
        k_s : float
            Ground thermal conductivity (in W/m-K).
        k_b : float
            Grout thermal conductivity (in W/m-K).
        R_fp : array
            (`n_pipes`,) array of fluid to pipe thermal resistance
            (in m-K/W).

        Returns
        -------
        float
            Borehole wall temperature (in degree Celsius).

        """
        p = self.r_b * np.stack([np.cos(theta), np.sin(theta)], axis=-1)
        T_b_wall = self.temperatures(p, q, T_b, k_s, k_b, R_fp)
        return T_b_wall

    def beta(self, k_s: float, k_b: float, R_fp: np.ndarray) -> np.ndarray:
        """Dimensionless fluid to pipe thermal resistances.

        Parameters
        ----------
        k_s : float
            Ground thermal conductivity (in W/m-K).
        k_b : float
            Grout thermal conductivity (in W/m-K).
        R_fp : array
            (`n_pipes`,) array of fluid to pipe thermal resistance
            (in m-K/W).

        Returns
        -------
        array
            Dimensionless fluid to pipe thermal resistances.

        """
        return R_fp * (2 * np.pi * k_b)

    @lru_cache(maxsize=8, typed=False)
    def factors_conjugate_multipole(self, k_s: float, k_b: float) -> np.ndarray:
        """Multipole factor coefficients for the conjugate multipoles.

        Parameters
        ----------
        k_s : float
            Ground thermal conductivity (in W/m-K).
        k_b : float
            Grout thermal conductivity (in W/m-K).

        Returns
        -------
        array
            (`n_pipes`, `J`, `n_pipes`, `J`,) array of coefficients

        """
        sigma = self.sigma(k_s, k_b)
        k = np.arange(1, self.J + 1)
        j = np.arange(1, self.J + 1)
        factors_conjugate_multipole = np.zeros((self.n_pipes, self.J, self.n_pipes, self.J), dtype=complex)
        for _k, _j in np.ndindex(self.J, self.J):
            factors_conjugate_multipole[:, _k, :, _j] = self._factors_conjugate_multipole(sigma, k[_k], j[_j])
        return factors_conjugate_multipole

    @lru_cache(maxsize=8, typed=False)
    def factors_line(self, k_s: float, k_b: float) -> np.ndarray:
        """Multipole factor coefficients for the heat injection rates.

        Parameters
        ----------
        k_s : float
            Thermal conductivity of the soil.
        k_b : float
            Thermal conductivity of the borehole wall.

        Returns
        -------
        array
             (`n_pipes`, `n_pipes`,) array of coefficients

        """
        k = np.arange(1, self.J + 1)
        factors_line = np.stack(
            [self._factors_line(k_s, k_b, _k) for _k in k],
            axis=1
        )
        return factors_line

    @cached_property
    def factors_multipole(self) -> np.ndarray:
        """Multipole factor coefficients for the multipoles.

        Returns
        -------
        array
            (`n_pipes`, `J`, `n_pipes`, `J`,) array of coefficients

        """
        k = np.arange(1, self.J + 1)
        j = np.arange(1, self.J + 1)
        factors_multipole = np.zeros((self.n_pipes, self.J, self.n_pipes, self.J), dtype=complex)
        for _k, _j in np.ndindex(self.J, self.J):
            factors_multipole[:, _k, :, _j] = self._factors_multipole(k[_k], j[_j])
        return factors_multipole

    def fluid_temperatures(self, q: np.ndarray, T_b: float, k_s: float, k_b: float, R_fp: np.ndarray) -> np.ndarray:
        """Evaluate fluid temperatures.

        Parameters
        ----------
        q : array
            (`n_pipes`,) array of heat injection rates in each pipe
            (in W/m).
        T_b : float
            Average borehole wall temperature (in degree Celsius).
        k_s : float
            Ground thermal conductivity (in W/m-K).
        k_b : float
            Grout thermal conductivity (in W/m-K).
        R_fp : array
            (`n_pipes`,) array of fluid to pipe thermal resistance
            (in m-K/W).

        Returns
        -------
        array
            (`n_pipes`,) array of Fluid temperatures (in degree Celsius).

        """
        # Multipole strengths (eqs. 34-36)
        P = self.multipoles(q, k_s, k_b, R_fp)
        # Fluid temperatures (eq. 32)
        a_q = self.thermal_resistances_line(k_s, k_b, R_fp)
        a_P = self.thermal_resistances_multipole(k_s, k_b)
        T_f = T_b + a_q @ q + np.tensordot(a_P, P).real
        return T_f

    def multipoles(self, q: np.ndarray, k_s: float, k_b: float, R_fp: np.ndarray) -> np.ndarray:
        """Evaluate multipoles.

        Parameters
        ----------
        q : array
            (`n_pipes`,) array of heat injection rates in each pipe
            (in W/m).
        k_s : float
            Ground thermal conductivity (in W/m-K).
        k_b : float
            Grout thermal conductivity (in W/m-K).
        R_fp : array
            (`n_pipes`,) array of fluid to pipe thermal resistance
            (in m-K/W).

        Returns
        -------
        array
            (`n_pipes`, `J`,) array of multipole strengths.

        """
        N = self.n_pipes * self.J
        # Solve for multipoles (eq. 36)
        A, b = self._system_of_equations(q, k_s, k_b, R_fp)
        P = np.linalg.solve(A, b)
        P = (P[:N] + 1.j * P[N:]).reshape((self.n_pipes, self.J))
        return P

    def pipe_temperatures(self, index: int, theta: float | np.ndarray, q: np.ndarray, T_b: float, k_s: float, k_b: float, R_fp: np.ndarray) -> np.ndarray:
        """Evaluate pipe temperatures.

        Parameters
        ----------
        index : int
            Index of the pipe for which to evaluate temperatures.
        theta : float or array
            Angular position(s) along the pipe circumference (in radians).
        q : array
            (`n_pipes`,) array of heat injection rates in each pipe
            (in W/m).
        T_b : float
            Average borehole wall temperature (in degree Celsius).
        k_s : float
            Ground thermal conductivity (in W/m-K).
        k_b : float
            Grout thermal conductivity (in W/m-K).
        R_fp : array
            (`n_pipes`,) array of fluid to pipe thermal resistance
            (in m-K/W).

        Returns
        -------
        array
            (`n_pipes`,) array of Pipe temperatures (in degree Celsius).

        """
        p = self.p[index] + self.r_p[index] * np.stack([np.cos(theta), np.sin(theta)], axis=-1)
        T_p = self.temperatures(p, q, T_b, k_s, k_b, R_fp)
        return T_p

    @lru_cache(maxsize=8, typed=False)
    def sigma(cls, k_s, k_b) -> float:
        """Dimensionless parameter for the two thermal conductivities.

        Parameters
        ----------
        k_s : float
            Ground thermal conductivity (in W/m-K).
        k_b : float
            Grout thermal conductivity (in W/m-K).

        Returns
        -------
        float
            Dimensionless parameter for the two thermal conductivities.

        """
        return (k_b - k_s) / (k_b + k_s)

    def thermal_resistances(self, k_s: float, k_b: float, R_fp: np.ndarray) -> np.ndarray:
        """Evaluate delta-circuit thermal resistances.

        Parameters
        ----------
        k_s : float
            Ground thermal conductivity (in W/m-K).
        k_b : float
            Grout thermal conductivity (in W/m-K).
        R_fp : array
            (`n_pipes`,) array of fluid to pipe thermal resistance
            (in m-K/W).

        Returns
        -------
        array
            (`n_pipes`, `n_pipes`,) array of delta-circuit thermal
            resistances (in m-K/W).

        """
        # Evaluate fluid temperatures for unit heat transfer rates at
        # each pipe, individually (eq. 32).
        R = np.stack(
            [
                self.fluid_temperatures(_q, 0., k_s, k_b, R_fp)
                for _q in np.eye(self.n_pipes)
                ],
            axis=0
        )
        K = -np.linalg.inv(R)
        # Delta-circuit thermal resistances (eq. 51)
        diag_indices = np.diag_indices_from(K)
        K[diag_indices] = -K.sum(axis=1)
        R_d = 1 / K
        return R_d

    def temperatures(self, p: np.ndarray, q: np.ndarray, T_b: float, k_s: float, k_b: float, R_fp: np.ndarray, tol: float = 1e-6, nan_inside_pipes: bool = False) -> np.ndarray:
        """Evaluate temperatures at given positions.

        Parameters
        ----------
        p : array
            (`n_positions`, 2) array of positions at which to evaluate
            temperatures (in meters). The position ``(0, 0)`` corresponds
            to the axis of the borehole.
        q : array
            (`n_pipes`,) array of heat injection rates in each pipe
            (in W/m).
        T_b : float
            Average borehole wall temperature (in degree Celsius).
        k_s : float
            Ground thermal conductivity (in W/m-K).
        k_b : float
            Grout thermal conductivity (in W/m-K).
        R_fp : array
            (`n_pipes`,) array of fluid to pipe thermal resistance
            (in m-K/W).
        tol : float, default: 1e-6
            Tolerance for determining whether a position is inside a
            pipe.
        nan_inside_pipes : bool, default: False
            Whether to set temperatures inside the pipes to NaN instead of
            the fluid temperatures.

        Returns
        -------
        array
            Temperatures at the given positions (in degree Celsius).

        """
        # Check if the input is a single position
        is_scalar = p.ndim == 1
        p = np.atleast_2d(p)
        # Determine which positions are inside the borehole and which are outside
        index_inside_borehole = np.linalg.norm(p, axis=1) < self.r_b
        index_outside_borehole = np.logical_not(index_inside_borehole).nonzero()[0]
        index_inside_borehole = index_inside_borehole.nonzero()[0]
        # Multipole strengths (eqs. 34-36)
        P = self.multipoles(q, k_s, k_b, R_fp)
        # Initialize temperature array
        T = np.zeros(p.shape[0])
        if np.size(index_inside_borehole) > 0:
            # Determine which positions are inside the pipes
            index_inside_pipes, index_pipes = (
                np.linalg.norm(
                    p[index_inside_borehole][:, np.newaxis] - self.p, axis=2
                    ) < self.r_p * (1 - tol)
                ).nonzero()
            if np.size(index_inside_pipes) > 0:
                # Inside the pipes, set temperatures to fluid temperatures
                if nan_inside_pipes:
                    T[index_inside_borehole[index_inside_pipes]] = np.nan
                else:
                    T_f = self.fluid_temperatures(q, T_b, k_s, k_b, R_fp)
                    T[index_inside_borehole[index_inside_pipes]] = T_f[index_pipes]
                # Remove positions inside the pipes from the positions inside the borehole
                index_inside_borehole = np.delete(
                    index_inside_borehole,
                    index_inside_pipes
                    )
            if np.size(index_inside_borehole) > 0:
                # Inside the borehole
                W_0, W_j = self._basic_solutions_inside_borehole(
                    p[index_inside_borehole], k_s, k_b
                    )
                T[index_inside_borehole] = T_b + np.real(
                    W_0 @ q / (2 * np.pi * k_b) + np.tensordot(W_j, P)
                )
        if np.size(index_outside_borehole) > 0:
            # Outside the borehole
            W_0, W_j = self._basic_solutions_outside_borehole(
                p[index_outside_borehole], k_s, k_b
                )
            T[index_outside_borehole] = T_b + np.real(
                W_0 @ q / (2 * np.pi * k_b) + np.tensordot(W_j, P)
            )
        if is_scalar:
            # Return a scalar if the input was a single position
            T = T.item()
        return T

    def thermal_resistances_line(self, k_s: float, k_b: float, R_fp: np.ndarray) -> np.ndarray:
        """Evaluate thermal resistances for heat extraction rates.

        Parameters
        ----------
        R_fp : array
            Fluid to pipe thermal resistances (in m-K/W).

        Returns
        -------
        array
            (`n_pipes`, `n_pipes`,) array of thermal resistances
            (in m-K/W).

        """
        thermal_resistances_line_zero_beta = self._thermal_resistances_line_zero_beta(k_s, k_b)
        diag_indices = np.diag_indices_from(
            thermal_resistances_line_zero_beta
            )
        R = thermal_resistances_line_zero_beta.copy()
        R[diag_indices] += R_fp
        return R

    @lru_cache(maxsize=8, typed=False)
    def thermal_resistances_multipole(self, k_s: float, k_b: float) -> np.ndarray:
        """Evaluate thermal resistances for multipoles.

        Parameters
        k_s : float
            Ground thermal conductivity (in W/m-K).
        k_b : float
            Grout thermal conductivity (in W/m-K).

        Returns
        -------
        array
             (`n_pipes`, `n_pipes`, `J`) array of thermal resistances.

        """
        thermal_resistances_multipole = np.stack(
            [
                self._thermal_resistances_multipole(k_s, k_b, j)
                for j in range(1, self.J + 1)
            ],
            axis=-1
        )
        return thermal_resistances_multipole

    def _basic_solutions_inside_borehole(self, p: np.ndarray, k_s: float, k_b: float) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate temperatures at given positions, assuming they are inside the borehole.

        Parameters
        ----------
        p : array
            (`n_positions`, 2) array of positions at which to evaluate
            temperatures (in meters). The position ``(0, 0)`` corresponds
            to the axis of the borehole.
        q : array
            (`n_pipes`,) array of heat injection rates in each pipe
            (in W/m).
        T_b : float
            Average borehole wall temperature (in degree Celsius).
        k_s : float
            Ground thermal conductivity (in W/m-K).
        k_b : float
            Grout thermal conductivity (in W/m-K).
        R_fp : array
            (`n_pipes`,) array of fluid to pipe thermal resistance
            (in m-K/W).

        Returns
        -------
        W_0 : array
            (`n_positions`, `n_pipes`,) array of basic solutions for the
            heat injection rates.
        W_j : array
            (`n_positions`, `n_pipes`,) array of basic solutions for the
            multipoles.

        """
        z = p[:, 0] + 1.j * p[:, 1]
        sigma = self.sigma(k_s, k_b)
        W_0 = (
            np.log(self.r_b / np.subtract.outer(z, self.z))
            + sigma * np.log(self.r_b**2 / (self.r_b**2 - np.multiply.outer(z.conj(), self.z)))
        )
        j = np.arange(1, self.J + 1)
        W_j = (
            np.power.outer(self.r_p / np.subtract.outer(z, self.z), j)
            + sigma * np.power.outer(np.multiply.outer(z.conj(), self.r_p) / (self.r_b**2 - np.multiply.outer(z.conj(), self.z)), j)
        )
        return W_0, W_j

    def _basic_solutions_outside_borehole(self, p: np.ndarray, k_s: float, k_b: float) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate temperatures at given positions, assuming they are outside the borehole.

        Parameters
        ----------
        p : array
            (`n_positions`, 2) array of positions at which to evaluate
            temperatures (in meters). The position ``(0, 0)`` corresponds
            to the axis of the borehole.
        q : array
            (`n_pipes`,) array of heat injection rates in each pipe
            (in W/m).
        T_b : float
            Average borehole wall temperature (in degree Celsius).
        k_s : float
            Ground thermal conductivity (in W/m-K).
        k_b : float
            Grout thermal conductivity (in W/m-K).
        R_fp : array
            (`n_pipes`,) array of fluid to pipe thermal resistance
            (in m-K/W).

        Returns
        -------
        W_0 : array
            (`n_positions`, `n_pipes`,) array of basic solutions for the
            heat injection rates.
        W_j : array
            (`n_positions`, `n_pipes`,) array of basic solutions for the
            multipoles.

        """
        z = p[:, 0] + 1.j * p[:, 1]
        sigma = self.sigma(k_s, k_b)
        W_0 = (
            (1 + sigma) * np.log(self.r_b / np.subtract.outer(z, self.z))
            + sigma * ((1 + sigma) / (1 - sigma)) * np.log(self.r_b / z[..., np.newaxis])
        )
        j = np.arange(1, self.J + 1)
        W_j = (1 + sigma) * np.power.outer(self.r_p / np.subtract.outer(z, self.z), j)
        return W_0, W_j

    def _factors_conjugate_multipole(self, sigma: float, k: int, j: int) -> np.ndarray:
        """Multipole factor coefficients for the conjugate multipoles.

        Parameters
        ----------
        sigma : float
            Dimensionless parameter for the two thermal conductivities.
        k : int
            Target multipole order.
        J : int
            Source multipole order.

        Returns
        -------
        array
            (`n_pipes`, `n_pipes`,) array of coefficients

        """
        j_p_max = np.minimum(k, j)
        z = self.z
        # Fourth term (eq. 34)
        a_P_conj = np.zeros((self.n_pipes, self.n_pipes), dtype=complex)
        for i in range(j_p_max + 1):
            numerator = np.multiply.outer(
                self.r_p**k * z**(j - i),
                self.r_p**j * z.conj()**(k - i)
                )
            denominator = (self.r_b**2 - self._zz.T)**(k + j - i)
            a_P_conj += sigma * comb(j, i) * comb(k + j - i - 1, j - 1) * numerator / denominator
        return a_P_conj

    def _factors_line(self, k_s: float, k_b: float, k: int) -> np.ndarray:
        """Multipole factor coefficients for the heat injection rates.

        Parameters
        ----------
        k_s : float
            Thermal conductivity of the soil.
        k_b : float
            Thermal conductivity of the borehole wall.
        k : int
            Target multipole order.

        Returns
        -------
        array
            (`n_pipes`, `n_pipes`,) array of coefficients

        """
        sigma = self.sigma(k_s, k_b)
        z = self.z
        zz = self._zz
        I = np.eye(self.n_pipes)
        # First term (eq. 34)
        denom_1 = ((1 - I) * -self._dis + I)**k
        numer_1 = ((1 - I) * self.r_p**k).T
        # Second term (eq. 34)
        denom_2 = (self.r_b**2 - zz.T)**k
        numer_2 = sigma * np.multiply.outer(self.r_p**k, z.conj()**k)
        # Sum of the two terms
        a_q = (numer_1 / denom_1 + numer_2 / denom_2) / k / (2 * np.pi * k_b)
        return a_q

    def _factors_multipole(self, k: int, j: int) -> np.ndarray:
        """Multipole factor coefficients for the multipoles.

        Parameters
        ----------
        k : int
            Target multipole order.
        J : int
            Source multipole order.

        Returns
        -------
        array
            (`n_pipes`, `n_pipes`,) array of coefficients

        """
        I = np.eye(self.n_pipes)
        # Third term (eq. 34)
        numerator = np.multiply.outer(
            (-self.r_p)**k,
            self.r_p**j
            )
        denominator = self._dis**(k + j) + I
        a_P = (1 - I) * comb(k + j - 1, j - 1) * numerator / denominator
        return a_P

    @lru_cache(maxsize=8, typed=False)
    def _initial_system_of_equations(self, k_s: float, k_b: float):
        """Initial system of equations for multipoles.

        Parameters
        ----------
        k_s : float
            Ground thermal conductivity (in W/m-K).
        k_b : float
            Grout thermal conductivity (in W/m-K).

        Returns
        -------
        array
            (`n_pipes`*`J`, `n_pipes`*`J`,) array of coefficients for the
            system of equations for multipoles.

        """
        N = self.n_pipes * self.J
        A = self.factors_multipole.reshape((N, N))
        C = self.factors_conjugate_multipole(k_s, k_b).reshape((N, N))
        # Split the real and image parts to allow the solution of a linear
        # system of equations (eq. 36). The coefficients `C` multiply the
        # conjugate of the multipoles.
        A = np.block(
            [[A.real + C.real, -A.imag + C.imag],
             [A.imag + C.imag, A.real - C.real]]
            )
        return A

    def _system_of_equations(self, q: np.ndarray, k_s: float, k_b: float, R_fp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the system of equations for the multipole solution.

        Parameters
        ----------
        q : array
            (`n_pipes`,) array of heat injection rates in each pipe
            (in W/m).
        k_s : float
            Ground thermal conductivity (in W/m-K).
        k_b : float
            Grout thermal conductivity (in W/m-K).
        R_fp : array
            (`n_pipes`,) array of fluid to pipe thermal resistance
            (in m-K/W).

        Returns
        -------
        A : array
            Coefficient matrix for the multipole solution.
        b : array
            Right-hand side vector for the multipole solution.

        """
        k = np.arange(1, self.J + 1)
        # Coefficients that depend on the fluid to pipe thermal
        # resistances (eq. 36)
        beta = self.beta(k_s, k_b, R_fp)
        beta_k = np.multiply.outer(beta, k)
        coeffs = (1 + beta_k) / (1 - beta_k)
        A = self._initial_system_of_equations(k_s, k_b).copy()
        diag_indices = np.diag_indices_from(A)
        A[diag_indices] += np.concatenate([coeffs.flatten(), -coeffs.flatten()])
        # Right hand side (eq. 36)
        b = (-self.factors_line(k_s, k_b) @ q).flatten()
        b = np.concatenate([b.real, b.imag])
        return A, b

    @lru_cache(maxsize=8, typed=False)
    def _thermal_resistances_line_zero_beta(self, k_s: float, k_b: float) -> np.ndarray:
        """Initialize thermal resistances for heat extraction rates.

        Parameters
        ----------
        k_s : float
            Ground thermal conductivity (in W/m-K).
        k_b : float
            Grout thermal conductivity (in W/m-K).

        Returns
        -------
        array
            (`n_pipes`, `n_pipes`,) array of thermal resistances
            (in m-K/W).

        """
        sigma = self.sigma(k_s, k_b)
        z = self.z
        zz = self._zz
        I = np.eye(self.n_pipes)
        absdis = np.abs(self._dis)
        # Thermal resistances (eq. 33), assuming `beta=0`
        R = 1 / (2 * np.pi * k_b) * (
            np.log(self.r_b / ((1 - I) * absdis + np.diag(self.r_p)))
            + sigma * np.log(
                self.r_b**2 / (
                    (1 - I) * np.abs(self.r_b**2 - zz)
                    + np.diag(self.r_b**2 - np.abs(z)**2)
                )
            )
        )
        return R

    def _thermal_resistances_multipole(self, k_s: float, k_b: float, j: int) -> np.ndarray:
        """Evaluate thermal resistances for multipoles.

        Parameters
        ----------
        k_s : float
            Ground thermal conductivity (in W/m-K).
        k_b : float
            Grout thermal conductivity (in W/m-K).
            Dimensionless parameter for the two thermal conductivities.
        j : int
            Multipole order.

        Returns
        -------
        array
            (`n_pipes`, `n_pipes`,) array of thermal resistances.

        """
        sigma = self.sigma(k_s, k_b)
        z = self.z
        zz = self._zz
        I = np.eye(self.n_pipes)
        # Thermal resistances of multipoles (eq. 32)
        numer_1 = (1 - I) * self.r_p**j
        denom_1 = self._dis**j + I
        numer_2 = np.multiply.outer(z.conj()**j, self.r_p**j)
        denom_2 = (self.r_b**2 - zz)**j
        a_P = (numer_1 / denom_1 + sigma * numer_2 / denom_2)
        return a_P
