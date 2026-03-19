# -*- coding: utf-8 -*-
"""
This module contains historical correction factor models for NIST calorimeters.

These models should take in an array of frequency (GHz)
and output an array of the same shape that represent the microcalorimeter
correction factor.

"""

import numpy as np


def type_N_clrm_A(f: np.ndarray) -> np.ndarray:
    """
    gc model of NIST's type N clrm A (type N label A) microcalorimeter.

    Parameters
    ----------
    f : np.ndarray
        Frquency (GHz)

    Returns
    -------
    gc : np.ndarray
        Calorimetric correction factor, same shape as f.
    """
    gc = 0.00344 * f**0.427 + 0.0003 - 1.11e-4 * f + 6.3e-6 * f**2
    return gc
