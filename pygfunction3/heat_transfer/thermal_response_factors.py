# -*- coding: utf-8 -*-
import warnings

import numpy as np
import numpy.typing as npt

from .finite_line_source import finite_line_source
from ..basis import Basis
from ..borehole import Borehole
from ..borefield import Borefield


def thermal_response_factors(
        time: npt.ArrayLike,
        alpha: float,
        p: npt.ArrayLike,
        source: Borefield | Borehole,
        basis: Basis,
        r_min: float = 0.,
        method: str = 'recurrence',
        deg: int | None = None,
        real_source: bool = True,
        mirror_source: bool = True
        ) -> np.ndarray:
    """Thermal response factors."""
    I = finite_line_source(time, alpha, p, source, basis.n_nodes, r_min=r_min, method=method, deg=deg, real_source=real_source, mirror_source=mirror_source)
    h = I @ basis._psi_coefs
    return h
