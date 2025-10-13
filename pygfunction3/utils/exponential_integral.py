# -*- coding: utf-8 -*-
import warnings

import numpy as np
import numpy.typing as npt


def exp1(x):
    q = 20 / 47 * x**np.sqrt(31 / 26)
    G = np.exp(-np.euler_gamma)
    b = np.sqrt(2 * (1 - G) / (G * (2 - G)))
    h_inf = (1 - G) * (G**2 - 6 * G + 12.) / (3 * G * (2 - G)**2 * b)
    h = 1 / (1 + x * np.sqrt(x)) + h_inf * q / (1 + q)
    E1 = np.exp(-x) * np.log(1 + G / x - (1 - G) / (h + b * x)**2) / (
        G + (1 - G) * np.exp(-x / (1 - G)))
    return E1
