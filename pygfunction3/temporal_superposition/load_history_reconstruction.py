# -*- coding: utf-8 -*-
import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import make_interp_spline

from ._temporal_superposition import _TemporalSuperposition


class LoadHistoryReconstruction(_TemporalSuperposition):
    """
    Load history reconstruction.

    Parameters
    ----------
    time : array-like
        Time steps at which the g-function is evaluated (in seconds).
    h_to_self : array-like
        Thermal response factors to borehole wall for each time step. The shape
        should be (`n_times`, `n_boreholes`, `n_nodes`,).
    h_to_point : array-like, optional
        Thermal response factors to point for each time step. The shape should be
        (`n_times`, `n_points`,). If None, only the borehole wall temperatures will be
        evaluated. Default is None.

    """

    def __init__(self, time: np.ndarray, h_to_self: np.ndarray, h_to_point: np.ndarray | None = None):
        if isinstance(time, list):
            time = np.asarray(time)
        self.time = time
        self.h_to_self = np.diff(h_to_self, prepend=0., axis=0)
        self.h_to_point = np.diff(h_to_point, prepend=0., axis=0) if h_to_point is not None else None
        n_axes = (np.ndim(h_to_self) - 1) // 2
        self.q = np.zeros((len(time), *h_to_self.shape[-n_axes:]))

        self.n_times = len(time)
        self._k = -1
        self._time = 0.

    def next_time_step(self) -> float:
        """Advance to next simulation time step.

        Returns
        -------
        float
            Time (in seconds) of the new time step.

        """
        self._k += 1
        self._time = self.time[self._k]
        return self._time

    def reset_history(self):
        """Reset the history to its initial condition.

        """
        self.q[...] = 0.
        self._time = 0.
        self._k = -1
        return

    def set_current_load(self, q: np.ndarray):
        """Set the current heat extraction rate.

        Parameters
        ----------
        q : array
            (`n_boreholes`, `n_nodes`,) array of heat extraction rates
            (in W/m).

        """
        self.q[self._k] = q
        return

    def temperature_to_self(self) -> np.ndarray:
        """Evaluate temperatures at nodes.

        Returns
        ----------
        T_b : array
            (`n_boreholes`, `n_nodes`,) array of borehole wall
            temperature variations (in degree Celcius).

        """
        # TODO : Do not reconstruct q twice
        q_reconstructed = self._reconstruct_load_history(self.time[:self._k+1], self.q[:self._k+1])
        T = self._spatial_temporal_superposition(self.h_to_self[:self._k+1], q_reconstructed)
        return T

    def temperature_to_point(self) -> np.ndarray:
        """Evaluate temperatures at point.

        Returns
        ----------
        T : array
            (`n_points`,) array of ground temperature variations
            (in degree Celcius).

        """
        # TODO : Do not reconstruct q twice
        q_reconstructed = self._reconstruct_load_history(self.time[:self._k+1], self.q[:self._k+1])
        T = self._spatial_temporal_superposition(self.h_to_point[:self._k+1], q_reconstructed)
        return T

    @staticmethod
    def _reconstruct_load_history(time: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Reconstruct the load history

        Parameters
        ----------
        current_time : float
            Current time (in seconds).
        time : array
            (`n_times`,) array of time steps (in seconds).
        q : array
            (`n_times`,`n_boreholes`, `n_nodes`,) array of loads (in W/m).

        Returns
        -------
        array
            (`n_times`,`n_boreholes`, `n_nodes`,) array of reconstructed
            loads (in W/m).

        """
        time = np.concatenate((np.zeros(1), time))
        dtime = np.diff(time)
        q_accumulated = np.cumulative_sum(dtime * q.T, axis=-1, include_initial=True).T
        dtime_reconstructed = dtime[::-1]
        time_reconstructed = np.cumulative_sum(dtime_reconstructed, include_initial=True)
        q_reconstructed = make_interp_spline(
            time,
            q_accumulated,
            k=1,
            bc_type='not-a-knot',
        )(time_reconstructed)
        q_reconstructed = np.diff(q_reconstructed, axis=0)
        q_reconstructed = (q_reconstructed.T / dtime_reconstructed).T[::-1]
        return q_reconstructed
