"""Analsysis functions for performing fitting using DataArrays."""

import xarray as xr
import numpy as np


def polyval(coord, coeffs, degree_dim: str = 'degree'):
    out = xr.polyval(coord, coeffs)
    out = out.transpose(..., *coord.dims)
    return out


def polyfit(xarr: xr.DataArray, dim: str, deg: int):
    """
    Polynomial fit a dataset.

    Parameters
    ----------
    xarr : xr.DataArray
        Data to fit.
    dim : str
        dimension to fit along.
    deg : int
        degree of fit.

    Returns
    -------
    coeffs : xr.DataArray
        Coefficients along dim
        'degree' as last dimension.

    """
    fit = xarr.polyfit(dim, 1, cov=True)
    coeffs = fit.polyfit_coefficients
    coeffs = coeffs.transpose(..., 'degree')
    return coeffs


def polyval2(
    coeffs: xr.DataArray,
    x: xr.DataArray,
):
    """
    Evaluate polynomial coefficients on x and y.

    coeffs x assumed to have dimension 'deg' with index corresponding
    to the power. I.E. deg = 2  is the coeffcient for c_2*x**2.

    Parameters
    ----------
    coeffs : xr.DataArray
        Coefficients along dimension 'deg'. Index of 'deg' corresponds
        to power of polynomial coefficient.
        coeffs should have shape (..., m) where m is the number of
        coefficients and (...) is the same shape as x.
    x : xr.DataArray.
        (..., n) shape, last dimension is the fit dimension.

    Returns
    -------
    y : xr.DataArray
        x evaluated at polynomial coefficients shape (..., n)
    """
    y = x.copy() * 0
    for d in coeffs['deg']:
        y += coeffs.sel(deg=d) * x**d
    return y


def polyfit2(
    x: xr.DataArray,
    y: xr.DataArray,
    deg: int,
    constrain_zero: bool = False,
    yunc: xr.DataArray = None,
):
    """
    Fit polynominal coefficients c_n for y = Sum(c_n * x^n) for n<=deg.

    Parameters
    ----------
    x : xr.DataArray.
        (..., n) shape, last dimension is the fit dimension.
    y : xr.DataArray.
        (..., n) shape, last dimension is the fit dimension.
        Must be same shape as x.
    deg : int
        Degree of polynomial fit.
    constrain_zero : bool
        Constrain fit to cross zero.
    yunc: xr.DataArray, optional
        (n,) 1d-array. Uncertainty on the yvalues that is diagonalized
        and used as weighting in the ordinary least squares. If not
        provided, no weighting is used.

    Returns
    -------
    coeffs : xr.DataArray
        Coefficients along dimension 'deg'. Index of 'deg' corresponds
        to power of polynomial coefficient.
    """
    # if constraining to zero, leave out the 0th term
    if not constrain_zero:
        coeff_ind = np.arange(0, deg + 1, 1)
    else:
        coeff_ind = np.arange(1, deg + 1, 1)

    # create an observation matrix X (..., n, m)
    X = x.expand_dims(dim={'coeff': coeff_ind}, axis=(len(x.shape))).copy()
    for i in coeff_ind:
        X.loc[..., i] = x**i

    # (..., n, 1) column vector of observations
    n = y.shape[-1]
    Y = y.expand_dims(
        dim='temp_fit',
        axis=len(y.shape),
    )

    # (..., n, n) matrix of weights, currently
    # a diagonal matrix of the standard uncertainties.
    # Should probably incorporate the covariance matrix, but
    # this is trivial for the time being.
    W = None
    if yunc is not None:
        W = xr.concat([Y] * n, dim='temp_fit')
        W.data[:] = np.diag(yunc)

    # do the weighted OLS (or non weighted if W is none)
    coeffs = ordinary_least_squares(X, Y, W)

    # set index to 'deg' for polynomial coefficients
    coeffs = coeffs.assign_coords({'coeff': coeff_ind})
    coeffs = coeffs.rename({'coeff': 'deg'})

    # if zero was constrained, add it
    # back in as all zeros for the simpler
    # data storage
    if constrain_zero:
        zeroth = coeffs.sel(deg=[1]).copy() * 0
        zeroth = zeroth.assign_coords(deg=[0])
        coeffs = xr.concat((zeroth, coeffs), dim='deg')

    return coeffs


def polysolve2(
    coeffs: xr.DataArray,
    y: xr.DataArray,
):
    """
    Solve's for solution to polynomial.

    Solve for x in y = Sum( c_n * x^n).

    Parameters
    ----------
    coeffs: xr.DataArray,
        Coefficients with along last dimension called deg. Index
        is integer with ineteger value corresponding to coefficient
        degree. Shape (..., m) for degree m polynomial. The 0th
        degree coefficient can optionally be omitted.
    y : xr.DataArray, optional
        Value to solve for solutions to polynomial at (if provided).
        Shape (..., n) where (...) matches the dimensions (...) in coeffs.
        Assumed to be zero if not provided.

    Returns
    -------
    result: xr.DataArray
        Shape (..., n, m-1) solutions to polynomial equation at y.
    """

    n = y.shape[-1]
    m = max(coeffs.deg)
    root_ind = np.arange(0, m, 1)

    # pre-allocate a solution array with a new dimension to store the roots
    sol = y.expand_dims(dim={'roots': root_ind}, axis=(len(y.shape))).copy() * 0

    raise Exception('Not yet implemented')


def ordinary_least_squares(
    X: xr.DataArray, Y: xr.DataArray, W: xr.DataArray = None
) -> xr.DataArray:
    """
    Perform ordinary least squares on observation M and results y.

    This function is designed to operate on stacks of arrays,where the
    Last 2 dimensions of M are treated as regressor array and last 2 dimensions
    of Y are treated as the response matrix. Formulation follows the linear
    matrix formulation outlined in:

    https://en.wikipedia.org/wiki/Ordinary_least_squares.

    All dimensions '...' in (...,n,m) are treates as copies/stacks. It is
    assmued y has the same dimensions (...).

    Weight matrix defined as follows:
    https://online.stat.psu.edu/stat501/lesson/13/13.1

    Typically a diagonal matrix of the uncertainties.

    Parameters
    ----------
    X : xr.DataArray
        With shape (...,n,m)
    Y : xr.DataArray
        With shape (...,n,1).
    W : xr.DataArray, optional
        Weight matrix with shape (..., n,m). If none is provide, identity
        matrix is used (i.e. no weighting).

    Returns
    -------
    coeff : xr.DataArray
        Array of coefficiencts with dimensions (...,m) corresponding
        to the columns along dimension m in X.

    """
    M = X.values
    y = Y.values

    # this transposes M along last 2 dimensions
    n_dims = len(M.shape)
    transpose_dims = np.arange(n_dims)
    transpose_dims[[n_dims - 1, n_dims - 2]] = transpose_dims[[n_dims - 2, n_dims - 1]]
    Mt = np.transpose(M, transpose_dims)

    # do least squares
    if W is None:
        coeff = np.linalg.inv(Mt @ M) @ Mt @ y
    else:
        coeff = np.linalg.inv(Mt @ W.data @ M) @ Mt @ W.data @ y

    # put back into xarray
    coeff = coeff[..., 0]
    dims = list(X.dims[:-2])
    coords = {k: X.coords[k] for k in dims}
    dims += ['coeff']
    coords['coeff'] = np.arange(coeff.shape[-1])
    out = xr.DataArray(coeff, dims=dims, coords=coords)

    return out
