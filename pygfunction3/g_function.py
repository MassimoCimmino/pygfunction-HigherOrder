# -*- coding: utf-8 -*-
from time import perf_counter
from typing import Tuple

import numpy as np
import numpy.typing as npt
from scipy.interpolate import make_interp_spline

from .basis import Basis
from .borefield import Borefield
from .thermal_response_factors import (
    thermal_response_factors_to_borehole,
    thermal_response_factors_to_point
)
from .temporal_superposition import LoadHistoryReconstruction


def g_function(
        borefield: Borefield,
        time: npt.ArrayLike,
        alpha: float,
        k_s: float,
        basis: Basis,
        boundary_condition: str = 'UBWT',
        fls_backend: str = 'recurrence',
        trf_backend: str = 'similarities',
        p: npt.ArrayLike | None = None,
        disp: bool = True,
        full_output: bool = False,
        options: dict = {}
        ) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the g-function of a borefield.

    Parameters
    ----------
    borefield : Borefield
        Borefield for which to calculate the g-function.
    time : array-like
        Time steps at which to evaluate the g-function.
    alpha : float
        Thermal diffusivity of the ground.
    k_s : float
        Thermal conductivity of the ground.
    basis : Basis
        Basis for the thermal response factors.
    boundary_condition : str, optional
        Boundary condition to use for the g-function calculation.
        Options are 'UBWT' (uniform borehole wall temperature) and 'UHTR'
        (uniform heat transfer rate). Default is 'UBWT'.
    fls_backend : str, optional
        Backend to use for the finite line source solution. Options are 'recurrence'
        and 'quadrature'. Default is 'recurrence'.
    trf_backend : str, optional
        Backend to use for the thermal response factors. Options are 'similarities'
        and 'detailed'. Default is 'similarities'.
    p : array-like, optional
        Points at which to evaluate the thermal response factors to point. If None, only
        thermal response factors to borehole will be evaluated. Default is None.
    disp : bool, optional
        Whether to display progress. Default is True.
    full_output : bool, optional
        Whether to return full output (g-function, borehole load history, and borehole temperature history). Default is False.
    options : dict, optional
        Additional options to pass to the thermal response factor calculation. Default is empty dict.

    """
    # Convert lists to arrays if necessary
    if isinstance(time, list):
        time = np.asarray(time)
    n_times = len(time)                     # Number of time steps
    n_boreholes = borefield.n_boreholes     # Number of boreholes
    n_nodes = basis.n_nodes                 # Number of nodes per borehole
    n_sources = n_boreholes * n_nodes       # Number of nodes in the borefield

    # Evaluate thermal response factors (TRFs)
    tic = perf_counter()
    if disp:
        print('Evaluating thermal response factors...', end=' ')
    h_to_self = thermal_response_factors_to_borehole(
        time,
        alpha,
        borefield,
        borefield,
        basis,
        r_min=borefield.r_b,
        r_min_indices=np.diag_indices(n_boreholes),
        backend=trf_backend,
        fls_backend=fls_backend,
        **options
        ).reshape((n_times, n_sources, -1))
    if disp:
        toc0 = perf_counter()
        print(f'done ({toc0 - tic:.2f} seconds)')

    # Solve for the g-function
    if disp:
        print('Solving for g-function...', end=' ')
    g_function, q_b, T_b = _solve(
        time,
        borefield,
        basis,
        h_to_self,
        h_to_point=None,
        boundary_condition=boundary_condition
    )
    if disp:
        toc1 = perf_counter()
        print(f'done ({toc1 - toc0:.2f} seconds)')
        print(f'Total time: {toc1 - tic:.2f} seconds')

    if full_output:
        return g_function, q_b, T_b
    else:
        return g_function


def _solve(
        time: np.ndarray,
        borefield: Borefield,
        basis: Basis,
        h_to_self: np.ndarray,
        h_to_point: np.ndarray | None = None,
        boundary_condition: str = 'UBWT'
        ):
    """
    Solve for the g-function given the thermal response factors.

    Parameters
    ----------
    time : array
        Time steps at which to evaluate the g-function.
    borefield : Borefield
        Borefield for which to calculate the g-function.
    basis : Basis
        Basis for the thermal response factors.
    h_to_self : array
        Thermal response factors to the borehole.
    h_to_point : array, optional
        Thermal response factors to points. If None, only thermal response factors to borehole will be evaluated. Default is None.
    boundary_condition : str, optional
        Boundary condition to use for the g-function calculation.
        Options are 'UBWT' (uniform borehole wall temperature) and 'UHTR'
        (uniform heat transfer rate). Default is 'UBWT'.

    Returns
    -------
    g_function : array
        G-function values.
    q_b : array
        Borehole load history.
    T_b : array
        Borehole temperature history.

    """
    n_times = len(time)                     # Number of time steps
    n_boreholes = borefield.n_boreholes     # Number of boreholes
    n_nodes = basis.n_nodes                 # Number of nodes per borehole
    n_sources = n_boreholes * n_nodes       # Number of nodes in the borefield

    # Initialize arrays for results
    g_function = np.empty(n_times)
    q_b = np.zeros((n_times, n_sources))
    T_b = np.zeros((n_times, n_sources))

    # Build weight array for the borefield
    w_sources = np.multiply.outer(borefield.L / 2, basis.w).flatten()
    L_tot = np.sum(borefield.L)             # Total length of the borefield

    if boundary_condition == 'UHTR':
        T_b = h_to_self.sum(axis=-1)
        q_b = np.ones_like(T_b)
        g_function = T_b @ w_sources / L_tot
    elif boundary_condition == 'UBWT':
        # Build interpolant for TRFs
        dt = np.diff(time, prepend=0.)
        h_to_self_dt = make_interp_spline(
            np.concatenate((np.zeros(1), time)),
            np.concatenate((np.zeros((1, n_sources, h_to_self.shape[-1])), h_to_self), axis=0),
            bc_type='not-a-knot',
            k=1,
            )(dt)

        # Solve for the g-function
        load_history = LoadHistoryReconstruction(time, h_to_self, h_to_point=h_to_point)
        for k in range(n_times):
            g_function[k], q_b[k], T_b[k] = _solve_time_step_UBWT(load_history, h_to_self_dt[k], w_sources, L_tot)
    return g_function, q_b, T_b


def _solve_time_step_UBWT(
        load_history: LoadHistoryReconstruction,
        h_to_self_dt: np.ndarray,
        w_sources: np.ndarray,
        L_tot: float
        ):
    """
    Solve for the g-function at a single time step given the thermal response factors and the load history.

    Parameters
    ----------
    load_history : LoadHistoryReconstruction
        Load history reconstruction object.
    h_to_self_dt : array
        Thermal response factors to the borehole at the current time step.
    w_sources : array
        Weight array for the borefield.
    L_tot : float
        Total length of the borefield.

    Returns
    -------
    g_function : float
        G-function value at the current time step.
    q_b : array
        Borehole load at the current time step.
    T_b : float
        Borehole temperature at the current time step.

    """
    # Advance to the next time step in the load history
    load_history.next_time_step()

    # Zero-load borehole wall temperature
    T_b0 = load_history.temperature_to_self()

    # Assemble system of equations for the current time step
    # The system is of the form A x = b, where x contains the borehole
    # load and temperature, and the g-function is the last element of x.
    A = np.block(
        [
            [h_to_self_dt, -np.ones((h_to_self_dt.shape[1], 1))],
            [w_sources[np.newaxis, ...], np.zeros((1, 1))]
        ]
    )
    b = np.concatenate((-T_b0, [L_tot]))

    # Solve the system of equations
    x = np.linalg.solve(A, b)

    # Extract results
    q_b = x[:-1]
    T_b = x[-1]
    g_function = x[-1]

    # Apply the current load to the load history
    load_history.set_current_load(q_b)

    return g_function, q_b, T_b
