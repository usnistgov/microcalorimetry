"""Module for trigonometry analysis functions."""

import xarray as xr
import numpy as np

__all__ = ['angle_diff']


def angle_diff(phase1: xr.DataArray, phase2: xr.DataArray, deg=False) -> xr.DataArray:
    """
    Calculate the minimum difference between two angles.

    Parameters
    ----------
    angle1 : xr.DataArray
        Phase array 1 in radians or deg.
    angle2 : xr.DataArray
        Phase array 1 in radians or deg.
    deg: bool, optional
        If true, calculates in degrees

    Returns
    -------
    out : xr.DataArray
        Phase difference between two angles.

    """
    const = np.pi
    if deg:
        const = 180

    diff = (phase1 - phase2).values
    diff[diff > const] -= 2 * const
    diff[diff < -const] += 2 * const

    # return the minimum possible phase difference
    out = phase1.copy()
    out.values = diff
    return out
