"""
Generic numerical methods.
"""

import xarray as xr

def mean_unique_values(arr: xr.DataArray, dim: str):
    """
    Average duplicate values across a dimension.
    """
    old_coords = {k: v for k, v in arr.coords.items() if k != dim}
    out = arr.groupby(dim).mean(dim)
    # copy over coordinates from the old array
    # don't include the dimension that was grouped and averaged
    out = out.assign_coords(old_coords)
    return out

def greedy_average(*arrays: xr.DataArray) -> xr.DataArray:
    """
    Average arrays by alligning coordinates.

    Missing values are ignored during the averaging.

    Parameters
    ----------
    *arrays : xr.DataArray
        Arrays to greedy average across.
    """
    alligned = xr.align(*arrays, join='outer')
    alligned = xr.concat(alligned, dim='__all__')
    mean = alligned.mean(dim='__all__', skipna=True)
    return mean


def greedy_std(
    *arrays: xr.DataArray,
    ddof: int = 1,
    min_points: int = 3,
) -> xr.DataArray:
    """
    Take the standard deviation across arrays.

    Values missing between the arrays are ignored.

    Parameters
    ----------
    ddof : int, optional
        DDOF, set to 1 for an unbiased estimator, by default 1
    min_points : int, optional
        _description_, by default 3

    Returns
    -------
    xr.DataArray
        _description_
    """
    alligned = xr.align(*arrays, join='outer')
    alligned = xr.concat(alligned, dim='__all__')
    std = alligned.std(dim='__all__', ddof=ddof, skipna=True)
    return std
