# -*- coding: utf-8 -*-
from .multiple_u_tube_parallel import MultipleUTubeParallel
from ._series_pipes import _SeriesPipes


class MultipleUTubeSeries(_SeriesPipes, MultipleUTubeParallel):
    """U-tube heat exchanger cross-section.

    Parameters
    ----------
    p : array_like
        (`n_pipes`, 2) array of pipe positions (in meters). The position
        ``(0, 0)`` corresponds to the axis of the borehole.
    r_p_in : array_like
        Inner radius of the pipes (in meters).
    r_p_out : array_like
        Outer radius of the pipes (in meters).
    r_b : float
        Borehole radius (in meters).
    fluid : fluid
        The fluid.
    k_s : float
        Ground thermal conductivity (in W/m-K).
    k_b : float
        Grout thermal conductivity (in W/m-K).
    k_p : array_like
        Thermal conductivity of the pipes (in W/m-K).
    epsilon : float
        Pipe surface roughness (in meters).
    parallel : bool, default: True
        True if pipes are in parallel. False if pipes are in series.
    J : int, default: 3
        Order of the multipole solution.

    Attributes
    ----------
    n_pipes : int
        Total number of pipes.
    n_pipes_down : int
        Number of downward-flowing pipes.
    n_pipes_up : int
        Number of upward-flowing pipes.
    R_p : array
        Conduction thermal resistance (in m-K/W).

    """
