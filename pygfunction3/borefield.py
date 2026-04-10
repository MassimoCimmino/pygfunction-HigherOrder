# -*- coding: utf-8 -*-
from functools import cached_property
from typing import Tuple
from typing_extensions import Self     # for compatibility with Python <= 3.10
import warnings

import numpy as np
import numpy.typing as npt


class Borefield(object):
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
        # Borehole head position (x, y)
        if isinstance(p_head, list):
            p_head = np.atleast_2d(p_head)
        self.n_boreholes = np.shape(p_head)[0]
        # Check if arguments are broadcastable to a valid common shape
        arg_names = ("L", "D", "r_b", "tilt", "orientation")
        args = (L, D, r_b, tilt, orientation)
        b = np.broadcast(*args)
        # Broadcast all variables to arrays of length `nBoreholes`
        for name, value in zip(arg_names, args):
            setattr(self, name, np.broadcast_to(value, self.n_boreholes))
        self.p_head = p_head
        # Vertical boreholes default to an orientation of zero
        self.orientation = np.where(
            np.broadcast_to(self.is_tilted, self.n_boreholes),
            orientation,
            0.
        )

    # def __repr__(self):
    #     s = (f'Borehole(H={self.H}, D={self.D}, r_b={self.r_b}, p={self.p},'
    #          f' tilt={self.tilt}, orientation={self.orientation})')
    #     return s

    @cached_property
    def all_identical(self) -> np.ndarray:
        """
        Returns true if all boreholes are identical.

        Returns
        -------
        bool
            True if all boreholes are identical (same L, D and tilt).

        """
        all_identical = (
            np.allclose(self.L, self.L[0]) and
            np.allclose(self.D, self.D[0]) and
            np.allclose(self.tilt, self.tilt[0])
        )
        return all_identical

    @cached_property
    def all_tilted(self) -> np.ndarray:
        """
        Returns true if all boreholes are inclined.

        Returns
        -------
        bool
            True if all boreholes are inclined.

        """
        return np.all(self.is_tilted)

    @cached_property
    def all_vertical(self) -> np.ndarray:
        """
        Returns true if all boreholes are vertical.

        Returns
        -------
        bool
            True if all boreholes are vertical.

        """
        return np.all(self.is_vertical)

    @cached_property
    def is_tilted(self) -> np.ndarray:
        """
        Returns true if the borehole is inclined.

        Returns
        -------
        bool
            True if borehole is inclined.

        """
        return np.abs(self.tilt) > 1.0e-6

    @cached_property
    def is_vertical(self) -> np.ndarray:
        """
        Returns true if the borehole is vertical.

        Returns
        -------
        bool
            True if borehole is vertical.

        """
        return np.logical_not(self.is_tilted)

    @cached_property
    def unit_vector(self) -> np.ndarray:
        """
        Unit vectors along the borehole axes.

        Returns
        -------
        array
            Unit vectors along the borehole axes.

        """
        unit_vector = np.stack(
            [
                np.sin(self.tilt) * np.cos(self.orientation),
                np.sin(self.tilt) * np.sin(self.orientation),
                -np.cos(self.tilt)
            ],
            axis=-1
            )
        return unit_vector

    @cached_property
    def p_top(self) -> np.ndarray:
        """
        Position of the top of the boreholes.

        Returns
        -------
        array
            Positions (x, y, z) (in meters) of the top of the boreholes.

        """
        return self.position(-1.)

    @cached_property
    def p_middle(self) -> np.ndarray:
        """
        Position of the middle of the boreholes.

        Returns
        -------
        array
            Positions (x, y, z) (in meters) of the middle of the boreholes.

        """
        p_middle = np.stack(
            [
                self.p_head[..., 0] + 0.5 * self.L * np.sin(self.tilt) * np.cos(self.orientation),
                self.p_head[..., 1] + 0.5 * self.L * np.sin(self.tilt) * np.sin(self.orientation),
                -self.D - 0.5 * self.L * np.cos(self.tilt)
            ],
            axis=-1
            )
        return p_middle

    @cached_property
    def p_bottom(self) -> np.ndarray:
        """
        Position of the bottom of the boreholes.

        Returns
        -------
        array
            Positions (x, y, z) (in meters) of the bottom of the boreholes.

        """
        return self.position(1.)

    def axial_position(self, p: npt.ArrayLike) -> np.ndarray:
        """
        Axial positions with regard to the top of the boreholes.

        Parameters
        ----------
        p : array-like
            Point(s) (x, y, z) (in meters) for which to compute the axial positions.

        Returns
        -------
        s : array
            Axial positions (in meters) along the boreholes.

        """
        if isinstance(p, list):
            p = np.asarray(p)
        p_proj = self.projection(p)
        s = self._axial_position(p_proj, self.p_top, self.unit_vector)
        return s

    def longitudinal_position(self, xi: npt.ArrayLike) -> np.ndarray:
        """
        Longitudinal positions along the boreholes.

        Parameters
        ----------
        xi : array-like
            Coordinate or (M,) array of coordinates along the boreholes.

        Returns
        -------
        s : array
            Longitudinal positions (in meters) along the boreholes.

        """
        if isinstance(xi, list):
            xi = np.asarray(xi)
        return self._longitudinal_position(xi, self.L)

    def position(self, xi: npt.ArrayLike) -> np.ndarray:
        """
        Positions along the boreholes.

        Parameters
        ----------
        xi : array-like
            Coordinate or (M,) array of coordinates along the boreholes.

        Returns
        -------
        p : array
            Positions (x, y, z) (in meters) along the boreholes.

        """
        if isinstance(xi, list):
            xi = np.asarray(xi)
        p = self._position(xi, self.L, self.p_middle, self.unit_vector)
        return p

    def projection(self, p: npt.ArrayLike) -> np.ndarray:
        """
        Projection of a point onto the borehole axes.

        Parameters
        ----------
        p : array-like
            Point(s) (x, y, z) (in meters) to be projected onto the borehole axes.

        Returns
        -------
        p_proj : array
            Projected point(s) (x, y, z) (in meters) onto the borehole axes.

        """
        if isinstance(p, list):
            p = np.asarray(p)
        p_proj = self._projection(p, self.p_middle, self.unit_vector)
        return p_proj

    def radial_axial_position(self, p: npt.ArrayLike, r_min: float | npt.ArrayLike | None = None, r_min_indices: Tuple[npt.ArrayLike, npt.ArrayLike] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Radial distances from the borehole axes and axial positions along the boreholes.

        Parameters
        ----------
        p : array-like
            Point(s) (x, y, z) (in meters) for which to compute the radial distance and axial position.
        r_min : float, array-like or None, optional
            Minimum radius (in meters) to avoid singularities. If not None,
            the returned radial distance is ``sqrt(r**2 + r_min**2)``.
            Default is None.
        r_min_indices : tuple of array-like or None, optional
            Tuple of indices specifying the points for which to apply the minimum radius. If not None,
            the returned radial distance for the specified points is ``sqrt(r**2 + r_min**2)``.
            Default is None.

        Returns
        -------
        r : array
            Radial distances (in meters) from the borehole axes.
        s : array
            Axial positions (in meters) along the boreholes.

        """
        if isinstance(p, list):
            p = np.asarray(p)
        p_proj = self.projection(p)
        r = self._radial_position(p_proj, p, r_min=r_min, r_min_indices=r_min_indices)
        s = self._axial_position(p_proj, self.p_top, self.unit_vector)
        return r, s

    def radial_position(self, p: npt.ArrayLike, r_min: float | npt.ArrayLike | None = None, r_min_indices: Tuple[npt.ArrayLike, npt.ArrayLike] | None = None) -> np.ndarray:
        """
        Radial distances from the borehole axes.

        Parameters
        ----------
        p : array-like
            Point(s) (x, y, z) (in meters) for which to compute the radial distance.
        r_min : float, array-like or None, optional
            Minimum radius (in meters) to avoid singularities. If not None,
            the returned radial distance is ``sqrt(r**2 + r_min**2)``.
            Default is None.
        r_min_indices : tuple of array-like or None, optional
            Tuple of indices specifying the points for which to apply the minimum radius. If not None,
            the returned radial distance for the specified points is ``sqrt(r**2 + r_min**2)``.
            Default is None.

        Returns
        -------
        r : array
            Radial distances (in meters) from the borehole axes.

        """
        if isinstance(p, list):
            p = np.asarray(p)
        p_proj = self.projection(p)
        r = self._radial_position(p_proj, p, r_min=r_min, r_min_indices=r_min_indices)
        return r

    @classmethod
    def _axial_position(cls, p_proj: np.ndarray, p_top: np.ndarray, unit_vector: np.ndarray) -> np.ndarray:
        """
        Axial positions along the boreholes with respect to the top of the boreholes.

        Parameters
        ----------
        p_proj : array
            Projected point(s) (x, y, z) (in meters) onto the borehole axes.
        p_top : array
            Positions (x, y, z) (in meters) of the top of the boreholes.
        unit_vector : array
            Unit vectors along the borehole axes.

        Returns
        -------
        s : array
            Axial positions (in meters) along the boreholes with respect to the top of the boreholes.

        """
        dp = p_proj - p_top
        s = np.einsum('...np,np->...n', dp, unit_vector)
        return s

    @classmethod
    def _longitudinal_position(cls, xi: npt.ArrayLike, L: float) -> np.ndarray:
        """
        Longitudinal positions along the boreholes.

        Parameters
        ----------
        xi : array-like
            Coordinate or (M,) array of coordinates along the boreholes.

        Returns
        -------
        s : float or array
            Longitudinal position (in meters) along the boreholes.

        """
        s = 0.5 * np.multiply.outer(xi + 1., L)
        return s

    @classmethod
    def _position(cls, xi: npt.ArrayLike, L: float, p_middle: np.ndarray, unit_vector: np.ndarray) -> np.ndarray:
        """
        Positions along the boreholes.

        Parameters
        ----------
        xi : float or array
            Coordinate or (M,) array of coordinates along the boreholes.

        Returns
        -------
        p : array
            Positions (x, y, z) (in meters) along the boreholes.

        """
        dp = np.einsum(
            'n,...,np->...np',
            0.5 * L,
            xi,
            unit_vector
        )
        p = p_middle + dp
        return p

    @classmethod
    def _projection(cls, p: np.ndarray, p_middle: np.ndarray, unit_vector: np.ndarray) -> np.ndarray:
        """
        Projection of a point onto the borehole axes.

        Parameters
        ----------
        p : array
            Point(s) (x, y, z) (in meters) to be projected onto the borehole axes.
        p_middle : array
            Positions (x, y, z) (in meters) of the middle of the boreholes.
        unit_vector : array
            Unit vectors along the borehole axes.

        Returns
        -------
        p_proj : array
            Projected point(s) (x, y, z) (in meters) onto the borehole axes.

        """
        p = p[..., np.newaxis, :]
        dp = p - p_middle
        p_proj = p_middle + np.einsum(
            '...np,np,nq->...nq',
            dp, unit_vector, unit_vector
            )
        return p_proj

    @classmethod
    def _radial_position(cls, p_proj: np.ndarray, p: np.ndarray, r_min: float | np.ndarray | None = None, r_min_indices: Tuple[npt.ArrayLike, npt.ArrayLike] | None = None) -> np.ndarray:
        """
        Radial positions from the borehole axes.

        Parameters
        ----------
        p_proj : array
            Projected point(s) (x, y, z) (in meters) onto the borehole axes.
        p : array
            Point(s) (x, y, z) (in meters) for which to calculate radial positions.
        r_min : float | np.ndarray | None
            Minimum radial distance (in meters).
        r_min_indices : Tuple[npt.ArrayLike, npt.ArrayLike] | None
            Indices of the points for which to apply the minimum radial distance.

        Returns
        -------
        r : array
            Radial positions (in meters) from the borehole axes.

        """
        p = p[..., np.newaxis, :]
        dp = p_proj - p
        r = np.linalg.norm(
            dp,
            axis=-1
        )
        if r_min is not None:
            if r_min_indices is None:
                r = np.sqrt(r**2 + r_min**2)
            elif isinstance(r_min, (float, np.floating)):
                r[r_min_indices[0], ..., r_min_indices[1]] = np.sqrt(r[r_min_indices[0], ..., r_min_indices[1]]**2 + r_min**2)
            else:
                r[r_min_indices[0], ..., r_min_indices[1]] = np.sqrt(r[r_min_indices[0], ..., r_min_indices[1]].T**2 + r_min[r_min_indices[1]]**2).T
        return r
