"""A collection of analysis functions commonly used with VNA measurements."""

import numpy as np
import xarray as xr
import microcalorimetry._gwex as dfm
import warnings

DataArray = xr.DataArray


def calc_s1p_basis(*s1p_c):
    """
    Generate basis frequency dependent basis vectors for the RI plane.

    Returns a 2x2 matrix that when given to rotate_s1p basis will transform
    them onto the new coordinate space in the imaginary plane.

    The new basis vectors are given from SVD, and so the first one represents
    the directo of change for the given sensors.

    Parameters
    ----------
    *s1p_c : TYPE
        Sensors.

    Returns
    -------
    new_basis : TYPE
        new basis vectors.

    """
    vals = [dfm.convert(dfm.s1p_ri, s).values for s in s1p_c]

    # insert a new dimension to stack along so you have
    # ...., frequency, sensor, (sreal,simag)
    axis = vals[0].ndim - 1
    vals = [np.expand_dims(v, axis) for v in vals]

    M = np.concat(vals, axis=axis)

    # perform a stacked SVD
    U, S, Vh = np.linalg.svd(M)

    # construct nwe basis matrix
    # transpose along last 2 dimensions
    # we can right multiply this
    axis = [i for i in range(Vh.ndim)]
    axis[-1], axis[-2] = axis[-2], axis[-1]
    new_basis_vals = Vh.transpose(axis)

    # put into xarray object
    # use s2p as a template
    with warnings.catch_warnings(action='ignore'):
        new_basis = dfm.empty_like(s1p_c[0], dfm.s2p_c).astype(float)
    new_basis.values = new_basis_vals
    new_basis.attrs.pop('dataformat')
    return new_basis


def rotate_s1p_basis(
    s1p_c: xr.DataArray, basis: xr.DataArray, return_as: dfm.DataFormat = dfm.s1p_c
):
    """
    Rotate an s1p_c file to a new basis in the RlIm plane.

    Parameters
    ----------
    s1p_c : xr.DataArray
        Complex s1p file.
    basis : xr.DataArray
        s2p_c like with real values.
    return_as: optional
        What format to return as. Defaults to s1p_c.

    Returns
    -------
    s1p_c
        s1p_c data that has been rotated.

    """
    ri = dfm.convert(dfm.s1p_ri, s1p_c)
    ri_2 = np.expand_dims(ri.values, axis=ri.ndim - 1)
    rotated = (ri_2 @ basis.values)[..., 0, :]

    # out
    out = dfm.empty_like(ri, dfm.s1p_ri)
    out.values = rotated
    return dfm.convert(return_as, out)


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


def rotate_device(device: DataArray, axes: list):
    """
    Rotate the parameters of a device

    Parameters
    ----------
    device: DataArray
        The device to rotate. Supported formats are s2p.

    axes: list
        The axes to change to. The original ones are the index (1, 2, ...) and the list
        are the ones to change to, so for example (2, 1) on an s2p will flip sides 1 and 2.

    Returns
    -------
    DataArray
        Rotated device.

    """
    # I can generalize later sort of; each format seems to have its own naming convention though
    # I am assuming s3 and 4p's will be the same so I can easily generalize that part if so, they just dont exist currently

    device_copy = device.copy(deep=True)

    if device.dataformat == 's2p_c':
        S11 = device_copy.loc[..., 1, 1].values
        S22 = device_copy.loc[..., 2, 2].values
        S12 = device_copy.loc[..., 1, 2].values
        S21 = device_copy.loc[..., 2, 1].values

        device_copy.loc[..., 1, 1] = S22
        device_copy.loc[..., 2, 2] = S11
        device_copy.loc[..., 1, 2] = S21
        device_copy.loc[..., 2, 1] = S12

    return device_copy
