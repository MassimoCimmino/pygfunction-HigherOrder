# -*- coding: utf-8 -*-
from functools import cached_property
from typing_extensions import Self     # for compatibility with Python <= 3.10
import warnings

import numpy as np
import numpy.typing as npt


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
        Returns the unit vector along the borehole axis.

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
        Returns the position of the top of the borehole.

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
        Returns the position of the middle of the borehole.

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
        Returns the position of the bottom of the borehole.

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
        Returns the axial position with regard to the top of the borehole.

        Parameters
        ----------
        p : array-like
            Point(s) (x, y, z) (in meters) for which to compute the axial position.

        Returns
        -------
        s : array
            Axial position (in meters) along the borehole.

        """
        p_proj = self.projection(p)
        s = self._axial_position(p_proj, self.p_top, self.unit_vector)
        return s

    def longitudinal_position(self, xi: npt.ArrayLike) -> np.ndarray:
        """
        Returns the longitudinal position along the borehole.

        Parameters
        ----------
        xi : array-like
            Coordinate or (M,) array of coordinates along the borehole.

        Returns
        -------
        s : array
            Longitudinal position (in meters) along the borehole.

        """
        return self._longitudinal_position(xi, self.L)

    def position(self, xi: npt.ArrayLike) -> np.ndarray:
        """
        Returns the position along the borehole.

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
        p = self._position(xi, self.L, self.p_middle, self.unit_vector)
        return p

    def projection(self, p: npt.ArrayLike) -> np.ndarray:
        """
        Returns the projection of a point onto the borehole axis.

        Parameters
        ----------
        p : array-like
            Point(s) (x, y, z) (in meters) to be projected onto the borehole axis.

        Returns
        -------
        p_proj : array
            Projected point(s) (x, y, z) (in meters) onto the borehole axis.

        """
        p_proj = self._projection(p, self.p_middle, self.unit_vector)
        return p_proj

    def radial_axial_position(self, p: npt.ArrayLike, r_min=None) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the radial distance from the borehole axis and the axial position.

        Parameters
        ----------
        p : array-like
            Point(s) (x, y, z) (in meters) for which to compute the radial distance and axial position.
        r_min : float or None, optional
            Minimum radius (in meters) to avoid singularities. If not None,
            the returned radial distance is ``sqrt(r**2 + r_min**2)``.
            Default is None.

        Returns
        -------
        r : array
            Radial distance (in meters) from the borehole axis.
        s : array
            Axial distance (in meters) along the borehole.

        """
        p_proj = self.projection(p)
        r = self._radial_position(p_proj, p, r_min=r_min)
        s = self._axial_position(p_proj, self.p_top, self.unit_vector)
        return r, s

    def radial_position(self, p: npt.ArrayLike, r_min=None) -> np.ndarray:
        """
        Returns the radial distance from the borehole axis.

        Parameters
        ----------
        p : array-like
            Point(s) (x, y, z) (in meters) for which to compute the radial distance.
        r_min : float or None, optional
            Minimum radius (in meters) to avoid singularities. If not None,
            the returned radial distance is ``sqrt(r**2 + r_min**2)``.
            Default is None.

        Returns
        -------
        r : array
            Radial distance (in meters) from the borehole axis.

        """
        p_proj = self.projection(p)
        r = self._radial_position(p_proj, p, r_min=r_min)
        return r

    @classmethod
    def _axial_position(cls, p_proj, p_top, unit_vector):
        """

        """
        dp = p_proj - p_top
        s = dp @ unit_vector
        return s

    @classmethod
    def _longitudinal_position(cls, xi, L):
        """
        Returns the longitudinal position along the borehole.

        Parameters
        ----------
        xi : array-like
            Coordinate or (M,) array of coordinates along the borehole.

        Returns
        -------
        s : float or array
            Longitudinal position (in meters) along the borehole.

        """
        if isinstance(xi, list):
            xi = np.atleast_1d(xi)
        s = 0.5 * L * (xi + 1.)
        return s

    @classmethod
    def _position(cls, xi, L, p_middle, unit_vector):
        """
        Returns the position along the borehole.

        Parameters
        ----------
        xi : float
            Coordinate along the borehole.

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
        if isinstance(xi, list):
            xi = np.atleast_1d(xi)
        p = p_middle + 0.5 * L * np.multiply.outer(xi, unit_vector)
        return p

    @classmethod
    def _projection(cls, p, p_middle, unit_vector):
        dp = p - p_middle
        p_proj = p_middle + np.multiply.outer(
            dp @ unit_vector,
            unit_vector
            )
        return p_proj

    @classmethod
    def _radial_position(cls, p_proj, p, r_min=None):
        """

        """
        dp = p_proj - p
        r = np.linalg.norm(
            dp,
            axis=-1
        )
        if r_min is not None:
            r = np.sqrt(r**2 + r_min**2)
        return r
