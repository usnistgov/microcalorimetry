"""
This module containes misceallaneous helper functions.

Specifically, these are functions that aren't intended for exposure
as part of the public API.
"""

import xarray as xr
import numpy as np
from rmellipse.uobjects import RMEMeas


def try_sel(thing: RMEMeas, name: str, fs: np.ndarray):
    """
    Try to down select to frequency. Make user aware if interpolating.

    Parameters
    ----------
    thing : RMEMeas
        RMEMeas object.
    name : str
        Name of object for print statments.
    fs : np.ndarray
        Frequency list to interpolate too if can't select.

    Returns
    -------
    RMEMeas
        original obect down selected or interpolated to fs.

    """
    try:
        return thing.sel(frequency=fs)
    except KeyError:
        missing = fs[np.logical_not(np.isin(fs, thing.nom.frequency.values))]
        print(name, 'interpolating these frequencies : ', missing)
        out = thing.interp(frequency=fs, kwargs=dict(fill_value='extrapolate'))
        return out


def drop_duplicate(arr: xr.DataArray, dim: str = 'frequency') -> xr.DataArray:
    """Drop duplicates across dimension

    Parameters
    ----------
    arr : xr.DataArray
        Dataarray
    dim : str
        Frequency dimension, by default 'frequency'

    Returns
    -------
    xd.DataArray
        Duplicates Dropped
    """
    return arr.drop_duplicates(dim, keep='first')


def concat(*arrs, dim: str, new_coords: iter):
    # this is set explicitly to fix a deprecation warning
    # that might cause errors in the future
    out = xr.concat(arrs, dim, join = 'outer')
    out = out.assign_coords({dim: new_coords})
    return out
