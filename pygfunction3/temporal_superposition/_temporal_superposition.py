# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike


class _TemporalSuperposition(ABC):
    """Temporal superposition of thermal response factors.

    """

    @abstractmethod
    def next_time_step(self) -> float:
        """Advance to next simulation time step.

        Returns
        -------
        float
            Time (in seconds) of the new time step.

        """
        ...

    @abstractmethod
    def reset_history(self):
        """Reset the history to its initial condition.

        """
        ...

    @abstractmethod
    def set_current_load(self, q: np.ndarray):
        """Set the current heat extraction rate.

        """
        ...

    @abstractmethod
    def temperature_to_self(self) -> np.ndarray:
        """Evaluate temperatures at nodes.

        """
        ...

    @abstractmethod
    def temperature_to_point(self) -> np.ndarray:
        """Evaluate temperatures at points.

        """
        ...

    @staticmethod
    def _spatial_temporal_superposition(h: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Spatial and temporal superposition.

        """
        axes = np.ndim(q)
        n = np.shape(q)[0]
        h = np.swapaxes(h[:n], 0, -axes)
        T = np.tensordot(h, q, axes=axes)
        return T
