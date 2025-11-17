# -*- coding: utf-8 -*-
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
    k_s : float
        Ground thermal conductivity (in W/m-K).
    k_b : float
        Grout thermal conductivity (in W/m-K).
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

    def __init__(self, r_b: float, r_p: ArrayLike, p: ArrayLike, k_s: float, k_b: float, J: int = 3):
        # Convert input to numpy.Array
        r_p = np.asarray(r_p)
        p = np.atleast_2d(p)

        # --- Class atributes ---
        # Parameters
        self.r_b = r_b
        self.r_p = r_p
        self.p = p
        self.k_s = k_s
        self.k_b = k_b
        self.J = J
        # Additional attributes
        self.n_pipes = len(r_p)
        self.sigma = (k_b - k_s) / (k_b + k_s)
        self.z = p[:, 0] + 1.j * p[:, 1]
        self._zz = np.multiply.outer(self.z.conj(), self.z)
        self._dis = np.subtract.outer(self.z, self.z)

        # Initialize mutipole coefficients and resistances
        self._initialize_factors()
        self._initialize_thermal_resistances()
        self._initialize_system_of_equations()

    def beta(self, R_fp: np.ndarray) -> np.ndarray:
        """Dimensionless fluid to pipe thermal resistances.

        Parameters
        ----------
        R_fp : array
            (`n_pipes`,) array of fluid to pipe thermal resistance
            (in m-K/W).

        Returns
        -------
        array
            Dimensionless fluid to pipe thermal resistances.

        """
        return R_fp * (2 * np.pi * self.k_b)

    def fluid_temperatures(self, q: np.ndarray, T_b: float, R_fp: np.ndarray) -> np.ndarray:
        """Evaluate fluid temperatures.

        Parameters
        ----------
        q : array
            (`n_pipes`,) array of heat injection rates in each pipe
            (in W/m).
        T_b : float
            Average borehole wall temperature (in degree Celsius).
        R_fp : array
            (`n_pipes`,) array of fluid to pipe thermal resistance
            (in m-K/W).

        Returns
        -------
        array
            (`n_pipes`,) array of Fluid temperatures (in degree Celsius).

        """
        # Dimensionless fluid to pipe thermal resistances
        beta = self.beta(R_fp)
        # Multipole strengths (eqs. 34-36)
        P = self.multipoles(q, beta)
        # Fluid temperatures (eq. 32)
        a_q = self._thermal_resistances_line(beta)
        a_P = self.thermal_resistances_multipole
        T_f = T_b + a_q @ q + np.tensordot(a_P, P).real
        return T_f

    def multipoles(self, q: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Evaluate multipoles.

        Parameters
        ----------
        q : array
            (`n_pipes`,) array of heat injection rates in each pipe
            (in W/m).
        beta : array
            Dimensionless fluid to pipe thermal resistances.

        Returns
        -------
        array
            (`n_pipes`, `J`,) array of multipole strengths.

        """
        N = self.n_pipes * self.J
        k = np.arange(1, self.J + 1)
        # Coefficients that depend on the fluid to pipe thermal
        # resistances (eq. 36)
        beta_k = np.multiply.outer(beta, k)
        coeffs = (1 + beta_k) / (1 - beta_k)
        diag_indices = np.diag_indices_from(self.A)
        A = self.A.copy()
        A[diag_indices] += np.concatenate([coeffs.flatten(), -coeffs.flatten()])
        # Right hand side (eq. 36)
        B = (-self.factors_line @ q).flatten()
        B = np.concatenate([B.real, B.imag])
        # Solve for multipoles (eq. 36)
        P = np.linalg.solve(A, B)
        P = (P[:N] + 1.j * P[N:]).reshape((self.n_pipes, self.J))
        return P

    def thermal_resistances(self, R_fp: np.ndarray) -> np.ndarray:
        """Evaluate delta-circuit thermal resistances.

        Parameters
        ----------
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
                self.fluid_temperatures(_q, 0., R_fp)
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

    def _factors_line(self, k: int) -> np.ndarray:
        """Multipole factor coefficients for the heat injection rates.

        Parameters
        ----------
        k : int
            Target multipole order.

        Returns
        -------
        array
            (`n_pipes`, `n_pipes`,) array of coefficients

        """
        z = self.z
        zz = self._zz
        I = np.eye(self.n_pipes)
        # First term (eq. 34)
        denom_1 = ((1 - I) * -self._dis + I)**k
        numer_1 = ((1 - I) * self.r_p**k).T
        # Second term (eq. 34)
        denom_2 = (self.r_b**2 - zz.T)**k
        numer_2 = self.sigma * np.multiply.outer(self.r_p**k, z.conj()**k)
        # Sum of the two terms
        a_q = (numer_1 / denom_1 + numer_2 / denom_2) / k / (2 * np.pi * self.k_b)
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

    def _factors_conjugate_multipole(self, k: int, j: int) -> np.ndarray:
        """Multipole factor coefficients for the conjugate multipoles.

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
            a_P_conj += self.sigma * comb(j, i) * comb(k + j - i - 1, j - 1) * numerator / denominator
        return a_P_conj

    def _initialize_factors(self):
        """Initialize multipole factor coefficients.

        """
        k = np.arange(1, self.J + 1)
        j = np.arange(1, self.J + 1)
        self.factors_line = np.stack(
            [self._factors_line(_k) for _k in k],
            axis=1
        )
        self.factors_multipole = np.zeros(
            (self.n_pipes, self.J, self.n_pipes, self.J),
            dtype=complex
        )
        self.factors_conjugate_multipole = np.zeros(
            (self.n_pipes, self.J, self.n_pipes, self.J),
            dtype=complex
        )
        for _k, _j in np.ndindex(self.J, self.J):
            self.factors_multipole[:, _k, :, _j] = self._factors_multipole(k[_k], j[_j])
            self.factors_conjugate_multipole[:, _k, :, _j] = self._factors_conjugate_multipole(k[_k], j[_j])

    def _initialize_system_of_equations(self):
        """Initialize the system of equations for multipoles.

        """
        N = self.n_pipes * self.J
        A = self.factors_multipole.reshape((N, N))
        C = self.factors_conjugate_multipole.reshape((N, N))
        # Split the real and image parts to allow the solution of a linear
        # system of equations (eq. 36). The coefficients `C` multiply the
        # conjugate of the multipoles.
        self.A = np.block(
            [[A.real + C.real, -A.imag + C.imag],
             [A.imag + C.imag, A.real - C.real]]
            )

    def _initialize_thermal_resistances(self):
        """Initialize thermal resistances.

        """
        self.thermal_resistances_line_zero_beta = (
            self._thermal_resistances_line_zero_beta()
            )
        self.thermal_resistances_multipole = np.stack(
            [
                self._thermal_resistances_multipole(j)
                for j in range(1, self.J + 1)
            ],
            axis=-1
        )

    def _thermal_resistances_line(self, beta: np.ndarray) -> np.ndarray:
        """Evaluate thermal resistances for heat extraction rates.

        Parameters
        ----------
        beta : array
            Dimensionless fluid to pipe thermal resistances.

        Returns
        -------
        array
            (`n_pipes`, `n_pipes`,) array of thermal resistances
            (in m-K/W).

        """
        diag_indices = np.diag_indices_from(
            self.thermal_resistances_line_zero_beta
            )
        R = self.thermal_resistances_line_zero_beta.copy()
        R[diag_indices] += beta / (2 * np.pi * self.k_b)
        return R

    def _thermal_resistances_line_zero_beta(self) -> np.ndarray:
        """Initialize thermal resistances for heat extraction rates.

        Returns
        -------
        array
            (`n_pipes`, `n_pipes`,) array of thermal resistances
            (in m-K/W).

        """
        z = self.z
        zz = self._zz
        I = np.eye(self.n_pipes)
        absdis = np.abs(self._dis)
        # Thermal resistances (eq. 33), assuming `beta=0`
        R = 1 / (2 * np.pi * self.k_b) * (
            np.log(self.r_b / ((1 - I) * absdis + np.diag(self.r_p)))
            + self.sigma * np.log(
                self.r_b**2 / (
                    (1 - I) * np.abs(self.r_b**2 - zz)
                    + np.diag(self.r_b**2 - np.abs(z)**2)
                )
            )
        )
        return R

    def _thermal_resistances_multipole(self, j: int) -> np.ndarray:
        """Evaluate thermal resistances for multipoles.

        Parameters
        ----------
        j : int
            Multipole order.

        Returns
        -------
        array
            (`n_pipes`, `n_pipes`,) array of thermal resistances.

        """
        z = self.z
        zz = self._zz
        I = np.eye(self.n_pipes)
        # Thermal resistances of multipoles (eq. 32)
        numer_1 = (1 - I) * self.r_p**j
        denom_1 = self._dis**j + I
        numer_2 = np.multiply.outer(z.conj()**j, self.r_p**j)
        denom_2 = (self.r_b**2 - zz)**j
        a_P = (numer_1 / denom_1 + self.sigma * numer_2 / denom_2)
        return a_P
