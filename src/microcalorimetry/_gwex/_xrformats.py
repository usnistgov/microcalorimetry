"""
Definitions for defining dataformats in xarrays.

Data formats are instructions for how to define the coordinates and labelling
systes for xarray objects for types of data sets.

Dataformats often relate the xarray format (i.e coordinat set and dimensions) to
a csv-like file format, and include instructions for reading that format
and converting them to others.

Some analysis functions will expect inputs to have specific dimensions, and labels,
and will refer to a specific DataFormat in the documentation when describing
the input.
"""

import xarray as xr
import numpy as _np
from typing import Union
from pathlib import Path
from microcalorimetry._gwex._utils import underline

# delete accessors before redefining, avoids a warning
try:
    del xr.DataArray.dfm
except AttributeError:
    pass


# %% dataformat Class Definition
class DataFormat:
    """Class that stores information describing a specific type of data format."""

    formats = {}
    categorized = {}

    def __init__(
        self,
        name: str,
        xdims: list,
        ycoords: dict[list],
        dtype: type,
        from_csv: callable = None,
        csv_file_extension: str = None,
        to_csv: callable = None,
        category: str = 'uncategorized',
        description: str = '',
    ):
        """
        Create a new dataformat.

        The xdims and ycoords define the expected shape of a DataFormat in the
        xarray. The xdims list defines the name of a dependent dimension that
        is not known in advance. The ycoords define the names and coordinates
        of dimensions that are expected and always the same for a format.

        For example, for a w1p_c format there is always a frequency dimension,
        and their is always 2 columns a11 and b11 for the incoming and outgoing
        waves measured at the VNA.In that case the xdims are ('frequency') and
        the ycoords would be {'w':[a11,b11]}.

        Parameters
        ----------
        name : str
            string name identifying the format, must be unique.
        xdims : list
            list of labels for x dimensions (i.e. dependent dimensions). These
            are dimensions that the dataformat is always expected to have, but
            the coordinates are not known in advane. For exampe, w1p_c data
            is aways expected to have a frequency dimension, but the exact
            frequencies are not known in advance.
        ycoords : dict
            dict of lists identifying dimensions and their coordinates
            of a data set that are known in advance.Keys are the names
            of the dimensions and the values are the coordinates for that
            dimension. For example, the w1p_c
            format always has the a and b waves, so the ycoords are
            {'w':[a11,b11]}. This creates a dimension called 'w' with size 2.
            When converting between formats, the ycoords are used to determine
            what dimensions and coordinates should be changed.
        dtype : type
            Data type of the format (i.e. complex, float, str, etc.).
        from_csv : callable, optional
            A function that defines how to read csv-like data in this format
            The funcion MUST take in a string path and output a tuple of
            the xcoords and the values of the data set in the correct shape
            . The default is None.
        csv_file_extension : str, optional
            String identifying the expected file extension for the format
            . The default is None.
        to_csv : callable, optional
            A function defining how to write csv like data in this format.
            The function must take in a tuple of (data,path) and write
            the data to a file at the location path.
        category : str, optional
            String identifying the category of the format.
            Used when rmellipse.dataformats.list_formats is called to
            organize the defined formats. The default is 'uncategorized'.
        description : str, optional
            String describing the format. Used when rmellipse.dataformats.list_formats
            is called to organize the defined formats. . The default is ''.

        Raises
        ------
        Exception
            If name is not unique.

        Returns
        -------
        None.

        """
        # read takes in (narr_format, file)
        # outputs a named array

        self.name = name
        self.xdims = xdims
        # self.ydims = ydims
        self.ycoords = ycoords
        # self.yshape = yshape
        self.dtype = dtype
        self.category = category
        self.description = description
        # read functions
        self._from_csv = from_csv
        self.to_csv = to_csv
        self.csv_file_extension = csv_file_extension

        # conversions
        self.converters = {}

        # add too dictionairies (flat and categorized)
        if name in DataFormat.formats.keys():
            raise Exception('Data format ' + name + ' name already exists.')
        DataFormat.formats[name] = self
        try:
            DataFormat.categorized[self.category]
        except KeyError:
            DataFormat.categorized[self.category] = {}
        DataFormat.categorized[self.category][name] = self

    def __eq__(self, o):
        return self.name == o.name

    def __repr__(self):
        """Get representation of self."""
        return self.__str__()

    def __str__(self):
        """Get string describing the format."""
        msg = ''
        msg += underline(self.name, '-') + '\n'
        msg += self.description + '\n'
        msg += 'attributes :' + '\n'
        maxn = 0

        dct = {k: v for k, v in self.__dict__.items()}

        ignore = ['converters', 'description']

        # add some other things
        dct['yshape'] = self.yshape
        dct['ydims'] = self.ydims
        dct.pop('converters')
        dct['converters.keys()'] = list(self.converters.keys())

        for k, v in dct.items():
            maxn = max(maxn, len(k))
        for k, v in dct.items():
            if k not in ignore:
                msg += k.rjust(maxn) + ' | ' + str(v) + '\n'

        return msg

    @property
    def yshape(self):
        """Shape of ydims of the format."""
        return tuple(len(self.ycoords[i]) for i in self.ydims)

    @property
    def ydims(self):
        """List of names of  ydims of the format."""
        return list(self.ycoords.keys())

    @property
    def uid(self):
        """Get UID of the format."""
        return str(id(self))

    # def from_csv(self, path: str, **kwargs):
    #     """
    #     Call the _from_csv attribute, then create the coords and values for the xarray.

    #     Parameters
    #     ----------
    #     path : str
    #         path to the data.

    #     Returns
    #     -------
    #     coords : dict
    #         Dict of the coords of the new xr.DataArray to be reated.
    #     data : np.ndarray
    #         Array of the values of th new xr.DataArray to be created.

    #     """
    #     xcoords, data = self._from_csv(path, **kwargs)
    #     coords = {k: v for k, v in self.ycoords.items()}
    #     for k, v in xcoords.items():
    #         coords[k] = v
    #     return coords, data

    def add_converter(self, function: callable, *old_formats):
        """
        Define a converter function for the format.

        Stores the converter function from self to the new format in a dictionary.
        The function should take in the data of the old format and a
        preinitialized array of data in the new format. You can assume the
        pre-initialized array has the correct dimensions/coordinates
        created.

        Parameters
        ----------
        new_format : DataFormat
            DataFormat being converted to.
        function : callable
            Function that converts to the new format.

        Returns
        -------
        None.

        """
        dict_key = ''
        for i in old_formats:
            dict_key += i.name + '; '
        self.converters[dict_key[0:-2]] = function

    def convert(self, *arrs) -> xr.DataArray:
        """
        Convert to new_format.

        Parameters
        ----------
        new_format : DataFormat
            New format to convert to.

        old_format : DataFormat, optional
            DESCRIPTION. The default is None.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        new_arr : xr.DataArray
            New DataArray in the new format.

        """
        # format attrs don't neccesarily get passed, so
        # TODO for multiple input ones this makes it require a set order
        old_formats = [i.dataformat for i in arrs]

        keys = self.converters.keys()
        dict_key = ''
        for i in keys:
            check1 = all([i.__contains__(j) for j in old_formats])
            check2 = len(i.split(';')) == len(old_formats)
            if check1 and check2:
                dict_key = i
                break

        # dict_key = ""
        # for i in old_formats:
        #     dict_key += i + "; "
        # dict_key = dict_key[0:-2]
        # print(old_formats)
        # if old_format is None:
        #     old_format = self.dataformat
        #     if old_format is None:
        #         raise Exception('Could not infer the data format. Please specify old_format')

        # the arrs should be on a common grid
        new_arr = empty_like(arrs[0], self)
        try:
            c = self.converters[dict_key]
        except KeyError:
            raise Exception(
                'Converter to ' + self.name + ' from ' + dict_key + ' not defined.'
            )
        new_arr = c(new_arr, *arrs)
        # new_arr = old_format.convert(self._da, new_arr, new_format)
        new_arr.dfm.dataformat = self
        return new_arr


# %% Formatted Data Accessor


@xr.register_dataarray_accessor('dfm')
class _FormattedMeasArrayAccesor:
    """
    Defines an xarray.DataArray accessor that carries datformat information.

    Extends the xarray.DataArray objext so the dataformat is stored with the
    DataArray object as an attribute. Conversion function is accessible through
    the xarray object.

    Attributes
    ----------
    dataformat:DataFormat
        The DataFormat of the DataArray object.
    """

    def __init__(self, xarray_obj):
        self._da = xarray_obj

    @property
    def dataformat(self):
        try:
            return DataFormat.formats[self._da.attrs['dataformat']]
        except KeyError:
            return None

    @dataformat.setter
    def dataformat(self, dataformat: DataFormat):
        try:
            self._da.attrs['dataformat'] = dataformat.name
        except AttributeError:
            self._da.attrs['dataformat'] = dataformat

    def to_csv(self, path: str, **kwargs):
        """
        Saves the data as a file in the format specified by the data format.

        Attributes
        ----------
        path : str
            The output file path to save to

        **kwargs
            Additional keyword agruments specific to the data format being used
        """
        self.dataformat.to_csv(self._da, path, **kwargs)


def to_csv(folder: str, name: str, arr: xr.DataArray, ext: str = None, **kwargs):
    """Write a formated DataArray to csv-like file.

    Parameters
    ----------
    directory : str
        Path to write to.
    name: str
        Basename of file
    arr : xr.DataArray
        DataArray being written.
    ext: str, optional
        File extension to use. Defaults to the defined
        extension for the dataformat.
    kwargs:
        additional keywrod arguments specific to the data format
        being used.

    """
    dfm = arr.dfm.dataformat
    if ext is None:
        ext = dfm.csv_file_extension
    path = Path(folder) / (name + ext)

    arr.dfm.to_csv(path, **kwargs)
    return str(path)


def from_csv(
    file: Union[str, list[str]],
    dataformat: DataFormat,
    dim: str = 'repeat',
    coords: list = None,
    verbose: bool = False,
    verbose_title: str = None,
    **kwargs,
) -> xr.DataArray:
    """
    Read a file or list of files into a DataArray.

    IF a list of files are provided, the values are concatenaed along dim.

    Parameters
    ----------
    file : Union[str, list[str]]
        File or list of files to read.
    dataformat : DataFormat
        DataFormat being read.
    dim : str, optional
        Name of the dimension to concatenate along if a list of files
        are provided. The new dimension created is always at axis = 0.
        The default is 'repeat'.
    coords : list, optional
        Coordinates for the new dimension named dim that is created
        if a list of files are provided. Should be the same length as file. If
        None, a standard (0,len(file)) integer cooridinate is assigned.
        The default is None.
    verbose : bool, optional
        If true, prints infor about reading. The default is False.
    verbose_title : str, optional
        What to name the task of reading, printed out if verbose.
        The default is None.

    Returns
    -------
    xr.DataArray
        Read files in the DataFormat, if multiple files were provided they
        are concatenated along the first dimension. Meta data uses the first
        file in the case there are more than one. If individual data is desired,
        read the files one by one.

    """

    def single_read(f, **kwargs):
        # out_tuple = dataformat.from_csv(f, **kwargs)

        xcoords, data, meta = dataformat._from_csv(f, **kwargs)
        coords = {k: v for k, v in dataformat.ycoords.items()}
        for k, v in xcoords.items():
            coords[k] = v

        new_arr = xr.DataArray(
            data, dims=dataformat.xdims + dataformat.ydims, coords=coords
        )
        new_arr.attrs['dataformat'] = dataformat.name
        for i in meta:
            new_arr.attrs[i] = meta[i]

        return new_arr

    if isinstance(file, str):
        return single_read(file, **kwargs)
    else:
        if coords is None:
            dim = {dim: _np.arange(0, len(file))}
        else:
            dim = {dim: coords}
        out = single_read(file[0], **kwargs)
        out = out.expand_dims(axis=0, dim=dim).copy()
        for i in range(1, len(file)):
            # progress_bar(verbose_title + ':reading ' + str(len(file)) + ' csv of ' + dataformat.name + ' ',
            # len(file) - 1, i)
            out[i, ...] = single_read(file[i], **kwargs)[0]
    return out


def as_format(xarr: xr.DataArray, fmt: 'DataFormat') -> xr.DataArray:
    """
    Get xarr as fmt, assuming xarr has the correct shape.

    Re-assigns the dimension names and coordinates to match fmt. Returns a copy.

    Parameters
    ----------
    xarr : xr.DataArray
        DataArray to to.
    fmt : DataFormat
        Data format to assign the coordinats and dimensions of..

    Returns
    -------
    xr.DataArray.
        xr.DataArray with the proper format.

    """
    dims = list(xarr.dims[: -len(fmt.ydims) - len(fmt.xdims)])

    dims += list(fmt.xdims) + list(fmt.ydims)

    coords = {knew: xarr.coords[kold].values for knew, kold in zip(dims, xarr.dims)}
    for k, v in fmt.ycoords.items():
        coords[k] = v

    new = xr.DataArray(xarr.values, dims=dims, coords=coords)
    new.dfm.dataformat = fmt
    return new


make_into = as_format


def zeros_like(use_x: xr.DataArray, new_format: DataFormat) -> xr.DataArray:
    """
    Create a zeros DataArray using the xdims of use_x and the ydims of new_format.

    Parameters
    ----------
    use_x : xr.DataArray
        DataArray with a DataFormat to copy the coordinates from for the xdims.
    new_format : DataFormat
        New format to copy the y coordinates from.

    Returns
    -------
    new_arr : xr.DataArray
        New empty dataarray with the same xdims and coordinates as use_x and
        the same ycoordinates of new format.

    """
    new_arr = empty_like(use_x, new_format)
    new_arr.values[...] = 0

    return new_arr


def empty_like(use_x: xr.DataArray, new_format: DataFormat) -> xr.DataArray:
    """
    Create an empty DataArray using the xdims of use_x and the ydims of new_format.

    Parameters
    ----------
    use_x : xr.DataArray
        DataArray with a DataFormat to copy the coordinates from for the xdims.
    new_format : DataFormat
        New format to copy the y coordinates from.

    Returns
    -------
    new_arr : xr.DataArray
        New empty dataarray with the same xdims and coordinates as use_x and
        the same ycoordinates of new format.

    """
    old_format = use_x.dfm.dataformat

    # get shape of new data, by takeing shape of current data's xdimensions
    # and adding shape of new formats y dimensions
    new_shape = []
    for i in range(len(use_x.shape) - len(old_format.ydims)):
        new_shape.append(use_x.shape[i])
    new_shape += list(new_format.yshape)

    # use coords of current data that isn't a y coord, at it to ycoords
    # of new format
    new_coords = {
        k: v for k, v in use_x.coords.items() if k not in old_format.ycoords.keys()
    }
    for k, v in new_format.ycoords.items():
        new_coords[k] = v

    # get new dimensions
    new_dims = [d for d in use_x.dims if d not in old_format.ydims]
    new_dims += new_format.ydims

    # make an empty array
    new_arr = xr.DataArray(
        _np.empty(new_shape, dtype=new_format.dtype), coords=new_coords, dims=new_dims
    )

    new_arr.dfm.dataformat = new_format

    return new_arr


initialize_empty = empty_like


def convert(out_format: DataFormat, *arr) -> xr.DataArray:
    """
    Convert arr to out_format.

    Exposed static call to make it easier to find/wrap in propagators.

    Parameters
    ----------
    arr : xr.DataArray
        Array being converted.
    out_format : DataFormat
        Dataformat to convert to.

    Returns
    -------
    xr.DataArray:
        new converted dataarray.

    """

    # TODO legacy check
    if type(arr[0]) == DataFormat:
        temp = out_format
        out_format = arr[0]
        arr = tuple([temp])
        print(
            "WARNING: Had to do the switchero, go find / replace 'convert' and switch the dfm to the first position"
        )
    # if its already the right format, just return it.
    if len(arr) == 1 and arr[0].dfm.dataformat == out_format:
        return arr[0]

    return out_format.convert(*arr)


def list_formats(category: str = None):
    """
    List all the defined data formats.

    Parameters
    ----------
    category : str, optional
        If provided, will only print formats with the matching
        category. The default is None.

    Returns
    -------
    None.

    """

    def print_cat(k):
        print('')
        print(k)
        print('-' * len(k))
        for k2, f in DataFormat.categorized[k].items():
            name = f.name.rjust(10)
            try:
                ext = f.csv_file_extension.center(6)
            except AttributeError:
                ext = ' ' * 6
            descr = f.description
            delim = ' | '
            print(name, delim, ext, delim, descr)

    if category is None:
        for k, d in DataFormat.categorized.items():
            print_cat(k)
    else:
        print_cat(category)


def get_header(path, comment):
    """
    Returns the header of a file.

    Parameters
    ----------
    path : str
        File path.

    comment : str
        The comment character for the header.

    Returns
    -------
    list
        list of each line of the header.

    """
    with open(path) as file:
        lines = [line.rstrip() for line in file if line[0] == comment]

    return lines


if __name__ == '__main__':
    import microcalorimetry._gwex.dataformats as dfm

    print(dfm.w1p_c)
    dfm.list_formats()
