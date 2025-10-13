# -*- coding: utf-8 -*-
import warnings

import numpy as np
import numpy.typing as npt


def mirror_point(p: npt.ArrayLike) -> np.ndarray:
    """Generate the mirror point of a given point across the z=0 plane.

    Parameters
    ----------
    p : npt.ArrayLike
        The original point(s) as an array-like structure with shape (..., 3).

    Returns
    -------
    np.ndarray
        The mirror point(s) with the same shape as the input.

    """
    p_mirror = p.copy()
    p_mirror[..., -1] = -p[..., -1]
    return p_mirror
