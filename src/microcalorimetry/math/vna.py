"""A collection of analysis functions commonly used with VNA measurements."""

import numpy as np
import xarray as xr
import microcalorimetry._gwex as dfm
import warnings

DataArray = xr.DataArray

__all__ = ['dbmag', 'phase', 'linmag']


def dbmag(s_params: DataArray, const: float = 20) -> DataArray:
    """
    Convert complex s parameters to magnitude in dB.

    Calculates by const*log10(linmag(s_params)). By default const = 20.

    Parameters
    ----------
    s_params : array like
        any complex format s parameters. array like.
    const : float, optional
        Multiplication constant for converting to magnitude (i.e.
        const*log10(magnitude). Default is 20


    Returns
    -------
    DataArray
        Magnitude of s parameters in dB.

    """
    return const * np.log10(np.abs(s_params))


def phase(s_params: DataArray):
    """
    Calculate the phase of complex s parameters.

    Parameters
    ----------
    s_params : DataArray
        any complex format s parameters. array like.

    Returns
    -------
    DataArray
        Phase of s_params.

    """
    new = s_params.copy()
    new.values = np.angle(new.values)
    return new


def linmag(s_params: DataArray):
    """
    Calculate linear magnitude of complex s-parameters.

    Parameters
    ----------
    s_params : DataArray
        any complex format s parameters.


    Returns
    -------
    DataArray
        Converted sparams.

    """
    return np.abs(s_params)
