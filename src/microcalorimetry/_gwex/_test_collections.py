from microcalorimetry._gwex import DataFormat, as_format
import xarray as xr
import numpy as np


def make_sample(
    dfm: DataFormat, xlen: int = 10, rand: bool = True, extra_xdims: list[int] = []
) -> xr.DataArray:
    """Generate a sample measurement of a format.

    Uses the definition of the provided DataFormat to generate an
    example with the correcto dimensions and coordinates. Fills with
    either random values or ones. Coordinate dimension is a 0 indexed
    integer array by default.

    Parameters
    ----------
    dfm : DataFormat
        Format to make an example of.
    xshape : int, optional
        Use this for xshape of the requried x-dims for the dataformat.
    rand : bool, optional
        Use np.random.random to generate random values
        to fill the matrix. By default True. Otherwise fills with 1's.
    extra_xdims : list[int], optional
        Add extra x-dimensions, DataFormats should support arbitrary number
        of xdims.

    Returns
    -------
    xr.DataArray
        xarray formatted according to dfm.
    """
    shape = [xlen for d in dfm.xdims] + list(dfm.yshape)
    shape = [xlen for ed in extra_xdims] + shape
    if rand:
        values = np.random.random(shape)
    else:
        values = np.ones(shape, float)
    xarr = xr.DataArray(values)
    xarr = as_format(xarr, dfm)
    return xarr
