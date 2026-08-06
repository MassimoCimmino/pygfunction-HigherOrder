# -*- coding: utf-8 -*-
from collections.abc import Callable
from typing_extensions import Self

import numpy as np
from numpy.typing import ArrayLike

from scp import get_fluid


class Fluid:
    """Heat carrier fluid.

    Parameters
    ----------
    T_nominal : float
        Nominal fluid temperature (in degree Celsius).
    f_conductivity : callable
        Function that takes a temperature or an (``N``,) array of
        temperatures ``T`` and returns a float or an (``N``,) array of
        fluid thermal conductivities.
    f_density : callable
        Function that takes a temperature or an (``N``,) array of
        temperatures ``T`` and returns a float or an (``N``,) array of
        fluid densities.
    f_specific_heat : callable
        Function that takes a temperature or an (``N``,) array of
        temperatures ``T`` and returns a float or an (``N``,) array of
        fluid specific heat capacities.
    f_viscosity : callable
        Function that takes a temperature or an (``N``,) array of
        temperatures ``T`` and returns a float or an (``N``,) array of
        fluid dynamic viscosities.
    f_prandtl : callable, default: None
        Function that takes a temperature or an (``N``,) array of
        temperatures ``T`` and returns a float or an (``N``,) array of
        fluid Prandtl numbers. If ``None``, the Prandtl numbers are
        instead evaluated from the other physical properties.

    """

    def __init__(
        self,
        T_nominal: float,
        f_conductivity: Callable[[float| ArrayLike], float| np.ndarray],
        f_density: Callable[[float| ArrayLike], float| np.ndarray],
        f_specific_heat: Callable[[float| ArrayLike], float| np.ndarray],
        f_viscosity: Callable[[float| ArrayLike], float| np.ndarray],
        f_prandtl: Callable[[float| ArrayLike], float| np.ndarray] | None = None
    ):
        if f_prandtl is None:
            def f_prandtl(_T: float | ArrayLike) -> float | np.ndarray:
                """Prandtl number."""
                cp_f = f_specific_heat(_T)
                mu_f = f_viscosity(_T)
                k_f = f_conductivity(_T)
                return cp_f * mu_f / k_f

        # --- Class atributes ---
        self.T_nominal = T_nominal
        self._k_f = f_conductivity(T_nominal)
        self._rho_f = f_density(T_nominal)
        self._cp_f = f_specific_heat(T_nominal)
        self._mu_f = f_viscosity(T_nominal)
        self._prandtl = f_prandtl(T_nominal)

        # --- Fluid properties functions ---
        self.f_conductivity = f_conductivity
        self.f_density = f_density
        self.f_specific_heat = f_specific_heat
        self.f_viscosity = f_viscosity
        self.f_prandtl = f_prandtl

    def conductivity(
            self, T: float | ArrayLike | None = None) -> float | np.ndarray:
        """Thermal conductivity.

        Parameters
        ----------
        T : float, array or None, default: None
            Fluid temperature (in degree Celsius). If ``None``, the property
            at `T_nominal` is returned.

        Returns
        -------
        float or array
            The thermal conductivity (in W/m-K).

        """
        # Return the property at T_nominal if T is None
        if T is None:
            return self._k_f
        # Convert lists to arrays if necessary
        if isinstance(T, list):
            T = np.asarray(T)
        return self.f_conductivity(T)

    def density(
            self, T: float | ArrayLike | None = None) -> float | np.ndarray:
        """Density.

        Parameters
        ----------
        T : float, array or None, default: None
            Fluid temperature (in degree Celsius). If ``None``, the property
            at `T_nominal` is returned.

        Returns
        -------
        float or array
            The density (in kg/m3).

        """
        # Return the property at T_nominal if T is None
        if T is None:
            return self._rho_f
        # Convert lists to arrays if necessary
        if isinstance(T, list):
            T = np.asarray(T)
        return self.f_density(T)

    def prandtl(
            self, T: float | ArrayLike | None = None) -> float | np.ndarray:
        """Prandtl number.

        Parameters
        ----------
        T : float, array or None, default: None
            Fluid temperature (in degree Celsius). If ``None``, the property
            at `T_nominal` is returned.

        Returns
        -------
        float or array
            The Prandtl number.

        """
        # Return the property at T_nominal if T is None
        if T is None:
            return self._prandtl
        # Convert lists to arrays if necessary
        if isinstance(T, list):
            T = np.asarray(T)
        return self.f_prandtl(T)

    def specific_heat(
            self, T: float | ArrayLike | None = None) -> float | np.ndarray:
        """Specific isobaric heat capacity.

        Parameters
        ----------
        T : float, array or None, default: None
            Fluid temperature (in degree Celsius). If ``None``, the property
            at `T_nominal` is returned.

        Returns
        -------
        float or array
            The specific heat capacity (in J/kg-K).

        """
        # Return the property at T_nominal if T is None
        if T is None:
            return self._cp_f
        # Convert lists to arrays if necessary
        if isinstance(T, list):
            T = np.asarray(T)
        return self.f_specific_heat(T)

    def viscosity(
            self, T: float | ArrayLike | None = None) -> float | np.ndarray:
        """Dynamic viscosity.

        Parameters
        ----------
        T : float, array or None, default: None
            Fluid temperature (in degree Celsius). If ``None``, the property
            at `T_nominal` is returned.

        Returns
        -------
        float or array
            The dynamic viscosity (in Pa-s).

        """

        # Return the property at T_nominal if T is None
        if T is None:
            return self._mu_f
        # Convert lists to arrays if necessary
        if isinstance(T, list):
            T = np.asarray(T)
        return self.f_viscosity(T)

    @classmethod
    def from_scp(
        cls, fluid_name: str, T_nominal: float, x: float | None = None) -> Self:
        """Import heat carrier fluid from SecondaryCoolantProps.

        Parameters
        ----------
        fluid_name : {'Water', 'EthylAlcohol', 'EthyleneGlycol', 'MethylAlcohol', 'PropyleneGlycol'}
            Name of the fluid, as per the nomenclature of
            ``SecondaryCoolantProps``.
        T_nominal : float
            Nominal fluid temperature (in degree Celsius).
        x : float or None, default: None
            Mass fraction of the brine (from ``0.`` to ``1.``). Not
            used for fluid ``'Water'``.

        Returns
        -------
        fluid
            The fluid.

        """
        fluid_lower = fluid_name.lower()
        if fluid_lower == 'water':
            fluid = get_fluid('water')
        elif x is None:
            raise TypeError(f"Expected float input; got x = {x}")
        elif fluid_lower == 'ethylalcohol' or fluid_lower == 'ethyl_alcohol':
            fluid = get_fluid('ethyl_alcohol', concentration=x)
        elif fluid_lower == 'ethyleneglycol' or fluid_lower == 'ethylene_glycol':
            fluid = get_fluid('ethylene_glycol', concentration=x)
        elif fluid_lower == 'methylalcohol' or fluid_lower == 'methyl_alcohol':
            fluid = get_fluid('methyl_alcohol', concentration=x)
        elif fluid_lower == 'propyleneglycol' or fluid_lower == 'propylene_glycol':
            fluid = get_fluid('propylene_glycol', concentration=x)
        else:
            raise ValueError(
                f"Fluid {fluid_lower} is not a valid SecondaryCoolantProps fluid."
                )

        # Create functions for fluid properties
        f_conductivity = np.vectorize(fluid.conductivity)
        f_density = np.vectorize(fluid.density)
        f_prandtl = np.vectorize(fluid.prandtl)
        f_specific_heat = np.vectorize(fluid.specific_heat)
        f_viscosity = np.vectorize(fluid.viscosity)
        fluid = cls(
            T_nominal, f_conductivity, f_density, f_specific_heat, f_viscosity, f_prandtl=f_prandtl
            )
        return fluid
