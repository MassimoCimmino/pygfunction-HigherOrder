# -*- coding: utf-8 -*-
from typing_extensions import Self     # for compatibility with Python <= 3.10
import warnings

import numpy as np


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

        # Check if borehole is inclined
        self._is_tilted = np.abs(self.tilt) > 1.0e-6
        # Borehole orientation
        if self._is_tilted:
            self.orientation = float(orientation)
        else:
            self.orientation = 0.
        # Unit vector along the borehole axis
        self._unit_vector = np.array([
            np.sin(self.tilt) * np.cos(self.orientation),
            np.sin(self.tilt) * np.sin(self.orientation),
            -np.cos(self.tilt)
        ])
        # Top and bottom end positions
        self.p_middle = np.array([
            self.p_head[0] + 0.5 * self.L * np.sin(self.tilt) * np.cos(self.orientation),
            self.p_head[1] + 0.5 * self.L * np.sin(self.tilt) * np.sin(self.orientation),
            -self.D - 0.5 * self.L * np.cos(self.tilt)
        ])
        self.p_top = self._position(
            -1., self.L, self.p_middle, self.unit_vector)
        self.p_bottom = self._position(
            1., self.L, self.p_middle, self.unit_vector)

    def __repr__(self):
        s = (f'Borehole(H={self.H}, D={self.D}, r_b={self.r_b}, p={self.p},'
             f' tilt={self.tilt}, orientation={self.orientation})')
        return s

    @property
    def is_tilted(self) -> bool:
        """
        Returns true if the borehole is inclined.

        Returns
        -------
        bool
            True if borehole is inclined.

        """
        return self._is_tilted

    @property
    def is_vertical(self) -> bool:
        """
        Returns true if the borehole is vertical.

        Returns
        -------
        bool
            True if borehole is vertical.

        """
        return not self._is_tilted
    
    @property
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
        return self._unit_vector

    def position(self, s):
        """
        Returns the position along the borehole.

        Parameters
        ----------
        s : float
            Longitudinal position along the borehole (in meters).

        Returns
        -------
        p : array
            Position (x, y, z) (in meters) along the borehole.

        Examples
        --------
        >>> b1 = gt.borehole.Borehole(H=150., D=4., r_b=0.075, p=[5., 0.])
        >>> b1.position(75.)
        [5.0, 0.0, -79.0]

        """
        xi = self._coordinate(s, self.L)
        p = self._position(xi, self.L, self.p_middle, self.unit_vector)
        return p

    def axial_distance(self, s, p):
        xi = self._coordinate(s, self.L)
        p_proj = self._projection(p, self.p_middle, self.unit_vector)
        p_self = self._position(xi, self.L, self.p_middle, self.unit_vector)
        dz = self._axial_distance(p_proj, p_self, self.unit_vector)
        return dz

    def radial_distance(self, p, r_min=None):
        p_proj = self._projection(p, self.p_middle, self.unit_vector)
        r = self._radial_distance(p_proj, p, self.unit_vector, r_min=r_min)
        return r
    
    @classmethod
    def _axial_distance(cls, p_proj, p_self, unit_vector):
        """

        """
        if p_proj.ndim == 2 and p_self.ndim == 2:
            dp = p_proj - p_self[:, np.newaxis, ...]
        else:
            dp = p_proj - p_self
        # dz = np.linalg.norm(dp, axis=-1)
        dz = dp @ unit_vector
        return dz

    @classmethod
    def _projection(cls, p, p_middle, unit_vector):
        dp = p - p_middle
        p_proj = p_middle + np.multiply.outer(
            dp @ unit_vector,
            unit_vector
            )
        return p_proj
    
    @classmethod
    def _radial_distance(cls, p_proj, p, unit_vector, r_min=None):
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

    @classmethod
    def _coordinate(cls, s, L):
        """
        Returns the coordinates xi along the borehole.

        Parameters
        ----------
        s : float
            Longitudinal position along the borehole (in meters).

        Returns
        -------
        xi : float
            Coordinate along the borehole.

        """
        if isinstance(s, list):
            s = np.atleast_1d(s)
        xi = -1. + 2 * s / L
        return xi

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
        # p = np.stack(
        #     [
        #         p_head[0] + 0.5 * L * np.sin(tilt) * np.cos(orientation) * (1. + xi),
        #         p_head[1] + 0.5 * L * np.sin(tilt) * np.sin(orientation) * (1. + xi),
        #         -D - 0.5 * L * np.cos(tilt) * (1. + xi)
        #     ],
        #     axis=-1
        # )
        return p
