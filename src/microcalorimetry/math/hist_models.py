# -*- coding: utf-8 -*-
"""
This module contains historical Type A repeatability models.
"""


def typeA_24mm(f):
    """
    Historical repeatability of NIST's 2.4 mm Type A uncertainty.

    Accepts frequency in GHz

    Parameters
    ----------
    f : array
        Frequency in GHz.

    Returns
    -------
    unc : array
        Type A uncertainty same shape as f.

    """
    out = f.copy()
    f_hi = f[f > 29.8]
    out[f <= 29.8] = 0.001
    out[f > 29.8] = 7e-4 + 9e-6 * (f_hi - 24) ** 2
    return out
