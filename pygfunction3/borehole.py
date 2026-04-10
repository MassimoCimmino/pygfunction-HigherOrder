# -*- coding: utf-8 -*-
from functools import cached_property
from typing_extensions import Self     # for compatibility with Python <= 3.10
import warnings

import numpy as np
import numpy.typing as npt

from .borefield import Borefield


class Borehole(object):
    """
    Contains information regarding the dimensions and position of a borehole.

    Attributes
    ----------
    L : float
        Borehole length (in meters).
    D : float
        Borehole buried depth (in meters).
    r_b : float
        Borehole radius (in meters).
    x : float
        Position (in meters) of the head of the borehole along the x-axis.
    y : float
        Position (in meters) of the head of the borehole along the y-axis.
    tilt : float, optional
        Angle (in radians) from vertical of the axis of the borehole.
        Default is 0.
    orientation : float, optional
        Direction (in radians) of the tilt of the borehole. Defaults to zero
        if the borehole is vertical.
        Default is 0.

    """
    def __init__(self, L, D, r_b, p_head, tilt=0., orientation=0.):
        self.L = float(L)      # Borehole length
        self.D = float(D)      # Borehole buried depth
        self.r_b = float(r_b)  # Borehole radius
        # Borehole head position (x, y)
        if isinstance(p_head, list):
            p_head = np.atleast_1d(p_head)
        self.p_head = p_head
        # Borehole inclination
        self.tilt = float(tilt)
        # Borehole orientation
        if self.is_tilted:
            self.orientation = float(orientation)
        else:
            self.orientation = 0.

    def __repr__(self):
        s = (f'Borehole(H={self.H}, D={self.D}, r_b={self.r_b}, p={self.p},'
             f' tilt={self.tilt}, orientation={self.orientation})')
        return s

    @cached_property
    def is_tilted(self) -> bool:
        """
        Returns true if the borehole is inclined.

        Returns
        -------
        bool
            True if borehole is inclined.

        """
        return np.abs(self.tilt) > 1.0e-6

    @cached_property
    def is_vertical(self) -> bool:
        """
        Returns true if the borehole is vertical.

        Returns
        -------
        bool
            True if borehole is vertical.

        """
        return not self.is_tilted

    @cached_property
    def unit_vector(self) -> np.ndarray:
        """
        Unit vector along the borehole axis.

        Returns
        -------
        array
            Unit vector along the borehole axis.

        Examples
        --------
        >>> b1 = gt.borehole.Borehole(H=150., D=4., r_b=0.075, p=[0., 0.])
        >>> b1.unit_vector
        [ 0.  0. -1.]

        """
        unit_vector = np.array([
            np.sin(self.tilt) * np.cos(self.orientation),
            np.sin(self.tilt) * np.sin(self.orientation),
            -np.cos(self.tilt)
        ])
        return unit_vector

    @cached_property
    def p_top(self) -> np.ndarray:
        """
        Position of the top of the borehole.

        Returns
        -------
        array
            Position (x, y, z) (in meters) of the top of the borehole.

        Examples
        --------
        >>> b1 = gt.borehole.Borehole(H=150., D=4., r_b=0.075, p=[5., 0.])
        >>> b1.p_top
        [ 5.  0. -4.]

        """
        return self.position(-1.)

    @cached_property
    def p_middle(self) -> np.ndarray:
        """
        Position of the middle of the borehole.

        Returns
        -------
        array
            Position (x, y, z) (in meters) of the middle of the borehole.

        Examples
        --------
        >>> b1 = gt.borehole.Borehole(H=150., D=4., r_b=0.075, p=[5., 0.])
        >>> b1.p_middle
        [ 5.   0. -79.]

        """
        p_middle = np.array([
            self.p_head[0] + 0.5 * self.L * np.sin(self.tilt) * np.cos(self.orientation),
            self.p_head[1] + 0.5 * self.L * np.sin(self.tilt) * np.sin(self.orientation),
            -self.D - 0.5 * self.L * np.cos(self.tilt)
        ])
        return p_middle

    @cached_property
    def p_bottom(self) -> np.ndarray:
        """
        Position of the bottom of the borehole.

        Returns
        -------
        array
            Position (x, y, z) (in meters) of the bottom of the borehole.

        Examples
        --------
        >>> b1 = gt.borehole.Borehole(H=150., D=4., r_b=0.075, p=[5., 0.])
        >>> b1.p_bottom
        [ 5.  0. -154.]

        """
        return self.position(1.)

    def axial_position(self, p: npt.ArrayLike) -> np.ndarray:
        """
        Axial position with regard to the top of the borehole.

        Parameters
        ----------
        p : array-like
            Point(s) (x, y, z) (in meters) for which to compute the axial position.

        Returns
        -------
        s : array
            Axial position (in meters) along the borehole.

        """
        if isinstance(p, list):
            p = np.asarray(p)
        p_proj = self.projection(p)
        s = self._axial_position(p_proj, self.p_top, self.unit_vector)
        return s

    def is_identical(self, other: Self | Borefield) -> bool:
        """
        Returns true if the borehole is identical to another borehole.

        Parameters
        ----------
        other : Borehole
            Borehole to compare with.

        Returns
        -------
        bool
            True if the boreholes are identical (same L, D and tilt).

        """
        identical = (
            np.isclose(self.L, other.L)
            & np.isclose(self.D, other.D)
            & np.isclose(self.tilt, other.tilt)
        )
        return identical

    def longitudinal_position(self, xi: npt.ArrayLike) -> np.ndarray:
        """
        Longitudinal position along the borehole.

        Parameters
        ----------
        xi : array-like
            Coordinate or (M,) array of coordinates along the borehole.

        Returns
        -------
        s : float or array
            Longitudinal position (in meters) along the borehole.

        """
        return self._longitudinal_position(xi, self.L)

    def position(self, xi: npt.ArrayLike) -> np.ndarray:
        """
        Position along the borehole.

        Parameters
        ----------
        xi : array-like
            Coordinate or (M,) array of coordinates along the borehole

        Returns
        -------
        p : array
            Position (x, y, z) (in meters) along the borehole.

        Examples
        --------
        >>> b1 = gt.borehole.Borehole(H=150., D=4., r_b=0.075, p=[5., 0.])
        >>> b1.position(-1.)
        [ 5.  0. -4.]

        """
        if isinstance(xi, list):
            xi = np.asarray(xi)
        p = self._position(xi, self.L, self.p_middle, self.unit_vector)
        return p

    def projection(self, p: npt.ArrayLike) -> np.ndarray:
        """
        Projection of a point onto the borehole axis.

        Parameters
        ----------
        p : array-like
            Point(s) (x, y, z) (in meters) to be projected onto the borehole axis.

        Returns
        -------
        p_proj : array
            Projected point(s) (x, y, z) (in meters) onto the borehole axis.

        """
        if isinstance(p, list):
            p = np.asarray(p)
        p_proj = self._projection(p, self.p_middle, self.unit_vector)
        return p_proj

    def radial_axial_position(self, p: npt.ArrayLike, r_min: float | None = None, r_min_indices: npt.ArrayLike | None = None) -> tuple[float | np.ndarray, float | np.ndarray]:
        """
        Radial distance from the borehole axis and the axial position.

        Parameters
        ----------
        p : array-like
            Point(s) (x, y, z) (in meters) for which to compute the radial distance and axial position.
        r_min : float or None, optional
            Minimum radius (in meters) to avoid singularities. If not None,
            the returned radial distance is ``sqrt(r**2 + r_min**2)``.
            Default is None.
        r_min_indices : array-like of int, optional
            Indices of points for which to apply the minimum radius. If not None,
            the returned radial distance is ``sqrt(r**2 + r_min**2)`` for the
            specified indices and ``r`` for the other points. Default is None.

        Returns
        -------
        r : float or array
            Radial distance (in meters) from the borehole axis.
        s : float or array
            Axial distance (in meters) along the borehole.

        """
        if isinstance(p, list):
            p = np.asarray(p)
        p_proj = self.projection(p)
        r = self._radial_position(p_proj, p, r_min=r_min, r_min_indices=r_min_indices)
        s = self._axial_position(p_proj, self.p_top, self.unit_vector)
        return r, s

    def radial_position(self, p: npt.ArrayLike, r_min: float | npt.ArrayLike | None = None, r_min_indices: npt.ArrayLike | None = None) -> float | np.ndarray:
        """
        Radial distance from the borehole axis.

        Parameters
        ----------
        p : array-like
            Point(s) (x, y, z) (in meters) for which to compute the radial distance.
        r_min : float or None, optional
            Minimum radius (in meters) to avoid singularities. If not None,
            the returned radial distance is ``sqrt(r**2 + r_min**2)``.
            Default is None.
        r_min_indices : array-like of int, optional
            Indices of points for which to apply the minimum radius. If not None,
            the returned radial distance is ``sqrt(r**2 + r_min**2)`` for the
            specified indices and ``r`` for the other points. Default is None.

        Returns
        -------
        r : float or array
            Radial distance (in meters) from the borehole axis.

        """
        if isinstance(p, list):
            p = np.asarray(p)
        p_proj = self.projection(p)
        r = self._radial_position(p_proj, p, r_min=r_min, r_min_indices=r_min_indices)
        return r

    @classmethod
    def _axial_position(cls, p_proj: np.ndarray, p_top: np.ndarray, unit_vector: np.ndarray) -> float | np.ndarray:
        """
        Axial position along the borehole with regard to the top of the borehole.

        Parameters
        ----------
        p_proj : array
            Projected point(s) (x, y, z) (in meters) onto the borehole axis.
        p_top : array
            Position (x, y, z) (in meters) of the top of the borehole.
        unit_vector : array
            Unit vector along the borehole axis.

        Returns
        -------
        s : float or array
            Axial position (in meters) along the borehole with regard to the top of the borehole.

        """
        dp = p_proj - p_top
        s = dp @ unit_vector
        return s

    @classmethod
    def _longitudinal_position(cls, xi: float | np.ndarray, L: float) -> float | np.ndarray:
        """
        Longitudinal position along the borehole.

        Parameters
        ----------
        xi : array-like
            Coordinate or (M,) array of coordinates along the borehole.

        Returns
        -------
        s : float or array
            Longitudinal position (in meters) along the borehole.

        """
        s = 0.5 * L * (xi + 1.)
        return s

    @classmethod
    def _position(cls, xi: float | np.ndarray, L: float, p_middle: np.ndarray, unit_vector: np.ndarray) -> np.ndarray:
        """
        Returns the position along the borehole.

        Parameters
        ----------
        xi : float or array
            Coordinate or (M,) array of coordinates along the borehole.

        Returns
        -------
        p : array
            Position (x, y, z) (in meters) along the borehole.

        Examples
        --------
        >>> b1 = gt.borehole.Borehole(H=150., D=4., r_b=0.075, p=[5., 0.])
        >>> b1._position(0.)
        [5.0, 0.0, -79.0]

        """
        p = p_middle + 0.5 * L * np.multiply.outer(xi, unit_vector)
        return p

    @classmethod
    def _projection(cls, p: np.ndarray, p_middle: np.ndarray, unit_vector: np.ndarray) -> np.ndarray:
        """
        Projection of a point onto the borehole axis.

        Parameters
        ----------
        p : array
            Point(s) (x, y, z) (in meters) to be projected onto the borehole axis.
        p_middle : array
            Position (x, y, z) (in meters) of the middle of the borehole.
        unit_vector : array
            Unit vector along the borehole axis.

        Returns
        -------
        p_proj : array
            Projected point(s) (x, y, z) (in meters) onto the borehole axis.

        """
        dp = p - p_middle
        p_proj = p_middle + np.multiply.outer(
            dp @ unit_vector,
            unit_vector
            )
        return p_proj

    @classmethod
    def _radial_position(cls, p_proj: np.ndarray, p: np.ndarray, r_min: float | npt.ArrayLike | None = None, r_min_indices: npt.ArrayLike | None = None) -> float | np.ndarray:
        """
        Radial position from the borehole axis.

        Parameters
        ----------
        p_proj : array
            Projected point(s) (x, y, z) (in meters) onto the borehole axis.
        p : array
            Point(s) (x, y, z) (in meters) for which to calculate the radial position.
        r_min : float or array, optional
            Minimum radius (in meters) for the radial position. Default is None.
        r_min_indices : array-like, optional
            Indices of points for which to apply the minimum radius. If not None,
            the returned radial distance is ``sqrt(r**2 + r_min**2)`` for the
            specified indices and ``r`` for the other points. Default is None.

        Returns
        -------
        r : float or array
            Radial distance (in meters) from the borehole axis.

        """
        dp = p_proj - p
        r = np.linalg.norm(
            dp,
            axis=-1
        )
        if r_min is not None:
            if r_min_indices is None:
                r = np.sqrt(r**2 + r_min**2)
            else:
                r[r_min_indices] = np.sqrt(r[r_min_indices]**2 + r_min**2)
        return r
