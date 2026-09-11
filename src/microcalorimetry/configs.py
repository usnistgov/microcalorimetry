"""
Module for defining configuration objects.

Configuration objects are either:

* Objects that inherit from DataModelContainer, and hold either a datset itsself or a file pointer to the dataset
* Objects that inherit from DictLike, which JSON compatable dictionary objects that can be instantiate from a file pointer or the dictionary itself.

Supported serialization formats are:

* JSON
* YAML (restricted to a JSON compatable subset using pyyaml's safe_load method)
* ExperimentParameters (spreadsheet editable JSON like object defined in Rocky Mountain Instruments)
* HDF5 Group Saveable Dictionaries (A generic object serialization format defined in Rocky Mountain Ellipse)

Each configuration object has a corresponding JSON schema that can be used to provide
validation.

"""

# third party dependencies

import h5py
from pathlib import Path
import pandas as pd

# uncertainty propagation
from rmellipse.utils import load_object, GroupSaveable
import numpy as np

# file formats
import json
from typing import Any, Callable
from yaml import safe_load
import importlib.util
import sys

# I think at somepoint this should be updated to a new draft
from referencing.jsonschema import DRAFT7 as DRAFT
from jsonschema import Draft7Validator as DraftValidator
from referencing import Registry
from rminstr.data_structures import ExptParameters
from rmellipse.uobjects import RMEMeas
from rmellipse.propagators import RMEProp
import xarray as xr
import microcalorimetry._gwex as gwex
import microcalorimetry.arrays as arrays

# build an internal registry of JSON schema
# that map the filenames to the schema object
# in a way that is compatable with the file
# referencing that the sphinx-jsonschema extension
# uses
# this way i can do "$ref":"filename.json#/definitions/schema"
# and build up schemas in local files of the same directory.
SCHEMA_DIR = Path(__file__).parents[0] / 'schema'
resources = []
for file in SCHEMA_DIR.iterdir():
    if file.is_file():
        with open(file, 'r') as fio:
            contents = DRAFT.create_resource(json.load(fio))
            resources.append((file.name, contents))
REGISTRY = Registry().with_resources(resources)

__all__ = [
    'DictLike',
    'MeasLike',
    'S11Like',
    'KDCLike',
    'DCSweepLike',
    'RFSweepLike',
    'ParsedRFSweep',
    'ParsedDCSweep',
    'CorrectionFactorModelInputs',
    'RFSweepParserConfig',
]

HDF5_EXTENSIONS = ['.h5', '.hdf5', '.hdf']


# %% File loaders for configs
def _load_file(obj, group=None):
    # assume string objects or Paths are files
    path = Path(obj)
    valid = ['.yml', '.yaml', '.csv'] + HDF5_EXTENSIONS
    # fo yml, yaml files
    if path.suffix == '.yml' or path.suffix == '.yaml':
        data = _load_yaml(path)
    # if csv, assume experiment parameters
    elif path.suffix == '.csv':
        data = __load_experiment_parameters(path)
    elif path.suffix == '.json':
        data = _load_json(path)
    elif path.suffix in HDF5_EXTENSIONS:
        data = _load_h5(path, group)
    else:
        raise ValueError(
            f'File extension of {path} not recognized (must be one of {valid})'
        )
    return data


def _load_json(file):
    with open(file, 'r') as f:
        obj = json.load(f)
    return obj


def _load_h5(file, group=None):
    with h5py.File(file, 'r') as f:
        g = f
        if group is not None:
            g = f[group]
            data = load_object(g, load_big_objects=True)
        else:
            try:
                data = load_object(g, load_big_objects=True)
            except KeyError as e:
                subgroups = [gi for gi in g]
                if len(subgroups) == 1:
                    g = g[subgroups[0]]
                    data = load_object(g, load_big_objects=True)
                else:
                    msg = 'If pointing to a file, the root group must be group saveable or their must be a single group saveable group below it.'
                    msg = str(e) + ': ' + msg
                    raise e from e

    return data


def _load_yaml(path: Path | str):
    with open(Path(path), 'r') as f:
        data = safe_load(f)
    return data


def __load_experiment_parameters(path: Path | str):
    config = ExptParameters(str(path)).config
    return config


def _load_schema_from_definition(fname: str, definition_name: str = None):
    """
    Convenienve method for loading in schema stored in a definition.

    Parameters
    ----------
    fname : str
        Name of schema (assumed to be in SCHEMA_DIR) without the extension.
        Assumed to be json.
    definition_name : str, optional
        Name of definition, by default None. If None, assumed
        to be the same as fname.

    Returns
    -------
    dict:
        JSON Schema
    """
    schema = load_config(SCHEMA_DIR / f'{fname}.json')
    if definition_name:
        schema = schema['definitions'][definition_name]
    return schema


def _group_saveable_from_measlike(
    data_model: 'MeasLike',
) -> RMEMeas:
    """
    Pass through data or load in data from an HDF5 group saveable.

    Parameters
    ----------
    data : object | Path
        File path for data or an object.
    group : _type_, optional
        _description_, by default None

    Returns
    -------
    object
        _description_
    """
    data = data_model.data
    path = data_model.path
    group = data_model.group
    if data is not None:
        return data
    else:
        with h5py.File(path, 'r') as f:
            g = f
            if group is not None:
                g = f[group]
                data = load_object(g, load_big_objects=True)
            else:
                try:
                    data = load_object(g, load_big_objects=True)
                except KeyError as e:
                    subgroups = [gi for gi in g]
                    if len(subgroups) == 1:
                        g = g[subgroups[0]]
                        data = load_object(g, load_big_objects=True)
                    else:
                        msg = 'If pointing to a file, the root group must be group saveable or their must be a single group saveable group below it.'
                        msg = str(e) + ': ' + msg
                        raise e from e

            # incase the person pointed to a container, and not the data set itself
            if isinstance(data, MeasLike):
                data = data.data

    return data


def _group_saveable_from_deprecatedmufmeas(
    data_model: 'MeasLike',
) -> RMEMeas:
    """
    Pass through data or load in data from an HDF5 group saveable.

    Parameters
    ----------
    data : object | Path
        File path for data or an object.
    group : _type_, optional
        _description_, by default None

    Returns
    -------
    object
        _description_
    """
    data = data_model.data
    path = data_model.path
    group = data_model.group
    if data is not None:
        return data
    else:
        with h5py.File(path, 'r') as f:
            g = f
            if group is not None:
                g = f[group]
            data = read_legacy_h5_MUFMeas(g)

    return data


def read_legacy_h5_MUFMeas(
    group: h5py.Group, nominal_only: bool = False, keep_attrs: bool = True
) -> 'RMEMeas':
    """
    Read a RMEMeas object from HDF5.

    Supports legacy formats from early development of rmellipse where umech_id
    had a different name, and where the class was called a MUFMeas object. These
    name changes will cause error in microcalorimetry code and/or the GroupSaveable
    loader.

    Parameters
    ----------
    group:
        HDF5 Group object
    nominal_only: bool, optional
        If true, will only load the nominal value of the data set. Saves
        time and memory if you don't need that data.
    keep_attrs: bool, optional
        If true, copies over all metadata in attrs. Otherwise, only
        keeps metadata required for class instantiation.

    Returns
    -------
    RMEMeas
        RMEMeas object.

    """

    class_name = group.attrs['__class__.__name__']
    if class_name == 'RMEMeas':
        umech_dim = 'umech_id'
    elif class_name == 'MUFmeas':
        umech_dim = 'parameter_locations'
    else:
        raise AttributeError('group ', group, ' isnt RMEMeas object')

    if not nominal_only:
        cov = load_object(group['cov'], load_big_objects=True)
        mc = load_object(group['mc'], load_big_objects=True)

        covdofs = None
        covcats = None
        try:
            covdofs = load_object(group['covdofs'], load_big_objects=True)
            covcats = load_object(group['covcats'], load_big_objects=True)

        except KeyError:
            print(
                'no covdofs/covcats. Possibly trying to read an earlier version of format'
            )

        name = group.attrs['name']
    else:
        things = {'cov': [], 'mc': []}
        for k in things.keys():
            data_set = group[k]
            if data_set.attrs['__class__.__name__'] != 'NoneType':
                print(data_set.attrs['__class__.__name__'])
                vals = np.array(data_set['values'][[0], ...])

                dims = []
                i = 0
                getting_dims = True
                while getting_dims:
                    try:
                        dims.append(data_set['values'].attrs['dim' + str(i)])
                        i += 1
                    except KeyError:
                        getting_dims = False
                coords = {}
                for k2 in dims:
                    coords[k2] = np.array(data_set[k2])
                    if coords[k2].dtype == np.dtype('O'):
                        coords[k2] = coords[k2].astype(str)
                coords[umech_dim] = [coords[umech_dim][0]]
                thing = xr.DataArray(vals, dims=dims, coords=coords)

                try:
                    thing.dfm.dataformat = data_set.attrs['dataformat']
                except KeyError:
                    pass
                things[k] = thing
            else:
                things[k] = None
        cov = things['cov']
        mc = things['mc']
        covcats = None
        covdofs = None
        name = group.attrs['name']

    # assign names to support old datasets
    if cov is not None and cov.dims[0] == 'parameter_locations':
        cov = cov.rename({'parameter_locations': 'umech_id'})
    if mc is not None and mc.dims[0] == 'parameter_locations':
        mc = mc.rename({'parameter_locations': 'umech_id'})
        mc = mc.rename({'umech_id': 'sample_id'})
    if covdofs is not None and covdofs.dims[0] == 'parameter_locations':
        covdofs = covdofs.rename({'parameter_locations': 'umech_id'})
    if covcats is not None and covcats.dims[0] == 'parameter_locations':
        covcats = covcats.rename({'parameter_locations': 'umech_id'})

    out = RMEMeas(name=name, cov=cov, mc=mc, covdofs=covdofs, covcats=covcats)
    if keep_attrs:
        for k in group.attrs:
            out.attrs[k] = group.attrs[k]
    else:
        out.attrs['unique_id'] = group.attrs['unique_id']
    # rename incase it's an old one with MUFmeas
    out.attrs['__class__.__name__'] = 'RMEMeas'
    return out


def split_h5path(path: Path | str) -> tuple[Path, str | None]:
    """
    Split a path formatted as path/to/file.h5/path/to/group

    Used to combine file and group paths into a single string.

    Parameters
    ----------
    path : Path | str
        Path spec formatted as path/to/hdf5_file.<ext>/path/to/group
        where <ext> is one of .h5, or .hdf5 file extensions.

    Returns
    -------
    tuple[Path, str | None]
        Seperated file path and group path.
    """
    full_path = Path(path)
    # split file paths and determine
    # if there is a valid h5 extension
    p = str(full_path.as_posix())
    possible_h5_extensions = HDF5_EXTENSIONS
    h5_ext = None
    for ext in possible_h5_extensions:
        if ext in p:
            h5_ext = ext
            break
    # if there is an h5 extension split into path and group
    if h5_ext is not None:
        split = p.split(h5_ext)
        assert len(split) <= 2 and len(split) >= 1
        path = split[0] + h5_ext
        if len(split) == 2:
            group = split[1]
            if group == '':
                group = None
        else:
            group = None
        return Path(path), group
    # otherwise return the path with
    # no group
    else:
        return full_path, None


def load_config(obj: Any | str | Path | list[Path | str], schema: dict | Path = None):
    """
    Validate a JSON like-object and/or load it from a file.

    Parameters
    ----------
    obj : Any | str | Path
        Any object to validate, or a file path to that object.
    schema : dict | Path, optional
        Schema or path to a jsonschema in .json format. by default None

    Returns
    -------
    _type_
        _description_
    """
    # if a file was passed (json or equivalent) ,load that
    if issubclass(type(obj), str) or issubclass(type(obj), Path):
        obj_path, grp = split_h5path(Path(obj))
        obj = _load_file(obj_path, grp)

    # if schema was passed, use it to validate the object
    if schema is not None:
        if issubclass(type(schema), str) or issubclass(type(schema), Path):
            schema = _load_json(schema)
        DraftValidator(schema, registry=REGISTRY).validate(obj)

    return obj


class PythonFunction:
    """
    A callable object that can handle file path pointers or module specs.

    e.g "python_module.py:function_name" will import function_name from
    the python module, so file paths to functionc an be provided.

    Or, module.submodule:function will import function from module.submodule.

    Or, you can just provide a function. This in an interface object for
    providing user defined funtions via CLI's or GUI's.
    """

    def __init__(self, obj: str | Path | Callable):
        """
        Provide either a path to a module, or a pathspec, or a function itself.

        Parameters
        ----------
        obj : str | Path | Callable
            Either a function, or a path spec (e.g. module.submodule:function_name)
            or a path to a python file (e.g. path/to/file.py:function_name)

        Returns
        -------
        None.

        """
        if isinstance(obj, str) or isinstance(obj, Path):
            # check if a valid file path
            path = Path(':'.join(str(obj).split(':')[:-1]))
            fun = str(obj).split(':')[-1]

            if path.exists():
                self._fn = self._from_path(path, fun)
                return

            # path is probably a module at this point
            self._fn = self._from_module_spec(path, fun)
        # assume anything else is just a function
        else:
            self._fn = obj

    @staticmethod
    def _from_path(file: Path, fname: str):
        module_name = Path(file).stem
        spec = importlib.util.spec_from_file_location(module_name, Path(file).resolve())
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, fname)

    @staticmethod
    def _from_module_spec(name: str, fname: str):
        module = importlib.import_module(str(name))
        return getattr(module, fname)

    @property
    def name(self):
        return self._fn.__name__

    def __call__(self, *args, **kwargs):
        """
        Call the associated function.
        """
        return self._fn(*args, **kwargs)


# %% Thinks that look like arrays (i.e measurements)


class MeasLike(GroupSaveable):
    """
    Describes any object that is like an RMEMeas.
    """

    """Supported file formats e.g: ['Group Saveable', '.h5']"""
    supported_files: list[tuple[str, str]]

    def __init__(
        self, *data_or_path, attrs=None, parent=None, **data_or_path_key_value_pair
    ):
        # Find where the data or path is stored.
        if len(data_or_path) == 1 and len(data_or_path_key_value_pair) == 0:
            data_or_path = data_or_path
        elif len(data_or_path) == 0 and len(data_or_path_key_value_pair) == 1:
            data_or_path = list(data_or_path_key_value_pair.values())
        else:
            raise ValueError(
                'Expected exactly one inpuyt for data_or_path positional or keyword argument.'
            )

        if len(data_or_path) == 1:
            obj = data_or_path[0]
        elif len(data_or_path) == 0:
            if 'path' in attrs:
                obj = attrs['path']
            else:
                obj = attrs['data']
        else:
            raise ValueError('Expected 1 kwarg/arg OR a field called "path" in attrs.')

        # if it's already a datamodel container (casting as self)
        # make a shallow copy
        if issubclass(type(obj), MeasLike):
            self._obj = obj._obj
        else:
            self._obj = obj

        GroupSaveable.__init__(
            self,
            attrs={'is_big_object': False, 'name': 'DataModelContainer'},
            parent=parent,
        )

        if issubclass(type(obj), MeasLike):
            self.data = obj.data
            self.path = obj.path
            self.group = obj.group

        elif isinstance(self._obj, str) or isinstance(self._obj, Path):
            self.data = None
            self.path, self.group = split_h5path(self._obj)
            self.attrs['path'] = Path(self._obj).as_posix()

        # store single basic types as metadata too
        elif any([isinstance(self._obj, t) for t in [float, int, bool, complex]]):
            self.data = self._obj
            self.path = None
            self.group = None
            self.attrs['data'] = self._obj

        elif isinstance(self._obj, GroupSaveable):
            self.data = self._obj
            self.path = None
            self.group = None

            self.add_child(
                key=self._obj.name, data=self._obj, is_big_object=self.data is not None
            )

        else:
            raise TypeError(f'{type(self._obj)} not supported')

    def load(self) -> RMEMeas:
        """Load data from self."""
        return _group_saveable_from_measlike(self)

    @classmethod
    def try_from(cls, obj: Any):
        """
        Try to turn obj into a DataModel, and return obj if it already is.

        Parameters
        ----------
        obj : Any
            Any object you think should already be a data model but
            you aren't quite sure.
        """
        try:
            return cls(obj)
        except TypeError:
            return obj


class S11Like(MeasLike):
    """
    Something that looks like a reflection coefficient.
    """

    supported_files = [
        ('NIST Cal services 1 port S-parameters text format.', '.dut'),
        ('Group Saveable', '*.h5 *.hdf5 *.hdf'),
    ]

    def __init__(self, *data_or_path, **data_or_path_kwargs):
        MeasLike.__init__(self, *data_or_path, **data_or_path_kwargs)

    def load(self) -> RMEMeas[arrays.S11]:
        container = self

        prp = RMEProp(sensitivity=True)
        convert = prp.propagate(gwex.convert)

        # load from hdf
        if container.data:
            return container.data

        else:
            path = container.path
            suffix = path.suffix

            if suffix in HDF5_EXTENSIONS:
                try:
                    d = _group_saveable_from_measlike(container)
                except ModuleNotFoundError:
                    # probably an old MUFmeas naning of the objects,
                    # try to use the deprecated function
                    d = _group_saveable_from_deprecatedmufmeas(container)

            # not hdf5 files
            elif '.meas' in suffix:
                raise NotImplementedError('not added .meas support yet')

            elif '.dut' in suffix:
                data = gwex.from_csv(str(path), gwex.dut_s1p)
                data = gwex.convert(gwex.s1p_c, data)
                # older code might rename these to something different
                try:
                    data = data.rename({'parameter_locations': 'umech_id'})
                except ValueError:
                    pass
                assert data.umech_id[0] == 'nominal'

                d = RMEMeas('sensor_id', data)
                d.make_umechs_unique(same_uid=True)
                d.assign_categories_to_all(Origin=path.name + r' $\Gamma$')
                d.name = path.name

            else:
                raise ValueError(f'Extension {suffix} not supporte for {path}')

            # convert to s1p_c complex
            if d.nom.dataformat != gwex.s1p_c.name:
                try:
                    d = convert(gwex.s1p_c, d)
                except KeyError as e:
                    raise (e)

        d = arrays.as_annotated(d, arrays.S11, validate=True)
        return d


class EtaLike(MeasLike):
    """
    Something that looks like an effective efficiency measurement.
    """

    supported_files = [
        ('NIST Cal services effective efficiency text format.', '.eff'),
        ('Group Saveable', '*.h5 *.hdf5 *.hdf'),
    ]

    def __init__(self, *data_or_path, **data_or_path_kwargs):
        MeasLike.__init__(self, *data_or_path, **data_or_path_kwargs)

    def load(self) -> RMEMeas[arrays.Eta]:
        """Load data from file or access data in container."""
        container = self
        # print(container.path)

        # if already data return data
        if container.data:
            return container.data

        path = container.path
        suffix = path.suffix
        # print(' ')
        # print(suffix)

        if suffix in HDF5_EXTENSIONS:
            try:
                d = _group_saveable_from_measlike(container)
            except ModuleNotFoundError:
                # probably an old MUFmeas naning of the objects,
                # try to use the deprecated function
                d = _group_saveable_from_deprecatedmufmeas(container)

            # sometimes the eta dimension is left off, if it's not there
            # add it in for compatability.
            if len(d.dims) == 1 and 'eta' not in d.cov.coords:
                prop = RMEProp(sensitivity=True)

                @prop.propagate
                def add_eta_coord(x):
                    return x.expand_dims('eta', axis=-1)

                d = add_eta_coord(d)

        elif '.eff' in suffix:
            df = pd.read_csv(path, sep=r'\s+', comment='#', header=None)
            # old format, not uncertainties
            if len(df.columns) == 4:
                eta = xr.DataArray(
                    np.expand_dims(df[3].to_numpy(), -1),
                    dims=('frequency', 'eta'),
                    coords={'frequency': df[0].to_numpy(), 'eta': [0]},
                )
                eta = gwex.as_format(eta, gwex.eff)
                d = RMEMeas.from_nom(path.stem, eta)
            # some format with uncertainties
            else:
                read = False
                fmts = [gwex.eff_7, gwex.eff_5, gwex.eff_2]
                for fmt in fmts:
                    if not read:
                        try:
                            data = gwex.from_csv(str(path), fmt)
                            read = True
                            # print(' ' * 8, 'reading ', path)
                        except Exception as e:
                            read = False
                            # print(e)
                if not read:
                    raise ValueError(f'Failed to read {path} as any of {fmts}')
                data = data.sel(col=['eta'])
                data = gwex.as_format(data, gwex.eff)
                d = RMEMeas(str(path), data)

        else:
            raise ValueError(f'Extension {suffix} not supported for {path}.')
        d = arrays.as_annotated(d, arrays.Eta, validate=True)
        return d


class GCLike(MeasLike):
    """
    Something that looks like a correction factor measurement.
    """

    supported_files = [('Group Saveable', '*.h5 *.hdf5 *.hdf')]

    def __init__(self, *data_or_path, **data_or_path_kwargs):
        MeasLike.__init__(self, *data_or_path, **data_or_path_kwargs)

    def load(self) -> RMEMeas[arrays.GC]:
        """Load data from self."""
        return super().load()


class KDCLike(MeasLike):
    """Something that looks like a KDC model."""

    supported_files = [('Group Saveable', '*.h5 *.hdf5 *.hdf')]

    def __init__(self, *data_or_path, **data_or_path_kwargs):
        MeasLike.__init__(self, *data_or_path, **data_or_path_kwargs)

    def load(self) -> RMEMeas[arrays.KDCTemInd | arrays.KDCTempDep]:
        """Load data from self."""
        return super().load()


class RFSweepLike(MeasLike):
    """
    Something that looks like an RFSweep measurement result.
    """

    supported_files = [('Group Saveable', '*.h5 *.hdf5 *.hdf')]

    def __init__(self, *data_or_path, **data_or_path_kwargs):
        MeasLike.__init__(self, *data_or_path, **data_or_path_kwargs)

    def load(self) -> RMEMeas[arrays.RFSweep]:
        """Load data from self."""
        return super().load()


class DCSweepLike(MeasLike):
    """
    Somthing that looks like a DC sweep measurement result.
    """

    supported_files = [('Group Saveable', '*.h5 *.hdf5 *.hdf')]

    def __init__(self, *data_or_path, **data_or_path_kwargs):
        MeasLike.__init__(self, *data_or_path, **data_or_path_kwargs)

    def load(self) -> RMEMeas[arrays.DCSweep]:
        """Load data from self."""
        return super().load()


# %% Things that look like dictionaries
class DictLike(dict):
    """A typed dictionary defined by a JSON schema that can be laoded from a file."""

    def __init__(
        self, obj: dict | str | Path | zip | list[Path | str], schema: dict | Path
    ):
        dict.__init__(self)
        if isinstance(obj, zip):
            obj = dict(obj)
        obj = load_config(obj, schema=schema)
        # make a shallow copy
        for k in obj:
            self[k] = obj[k]


class ParsedRFSweep(DictLike):
    """Dictionary of parsed RFSweep data measured in a microcalorimeter."""

    _SCHEMA_FILE = SCHEMA_DIR / 'ParsedRFSweep.json'
    SCHEMA = _load_json(_SCHEMA_FILE)['definitions']['ParsedRFSweep']

    def __init__(self, d: dict | Path | zip):
        DictLike.__init__(self, d, schema=self.SCHEMA)
        for k, v in self.items():
            self[k] = RFSweepLike.try_from(v)


class ParsedDCSweep(DictLike):
    """Dictionary of parsed DCSweep data measured in a microcalorimeter."""

    _SCHEMA_FILE = SCHEMA_DIR / 'ParsedDCSweep.json'
    SCHEMA = _load_json(_SCHEMA_FILE)['definitions']['ParsedDCSweep']

    def __init__(self, d: dict | Path | zip):
        DictLike.__init__(self, d, schema=self.SCHEMA)
        for k, v in self.items():
            self[k] = DCSweepLike.try_from(v)


class EtaHistorical(DictLike):
    """
    Configuration of paths to historical datasets.
    """

    _SCHEMA_FILE = SCHEMA_DIR / '.json'
    SCHEMA = load_config(SCHEMA_DIR / 'EtaHistorical.json')

    def __init__(self, obj: dict | Path | zip):
        DictLike.__init__(self, obj, schema=self.SCHEMA)

    def load_nominals(self, fail_on_error: bool = False) -> dict[xr.DataArray]:
        """
        Load in the nominal values of datasets in the config.

        Parameters
        ----------
        fail_on_error : bool, optional
            If true, throws an Exception if a dataset can't be laoded in
            for whatever reason. Otherwise warns that a dataset
            wasn't loaded and continues. The default is False.

        Raises
        ------
        e
            Exception that describes why a file wasnt loaded.

        Returns
        -------
        nominals : dict[xr.DataArray]
            dictionary of nominal effective efficiency datasets.

        """
        nominals = {}
        for name, datamodel in self.items():
            try:
                nom = EtaLike(datamodel).load().nom
                nominals[name] = nom
            except Exception as e:
                print(f'Failed to load {name} for exception: \n {e}')
                if fail_on_error:
                    raise e from e
        return nominals


# %% Mappings for the different inputs used to calculate a correction factor
class CorrectionFactorModelInputs(DictLike):
    """
    Mapping of all the measurement inputs required for computing a correction factor.

    Maps each every combination of reflective and load standard sensor used in a series
    of correction factor measurement, to the neccesary RF sweep measurements, reflection
    coefficients, and measurement of the splitter's port to port mismatch.
    """

    SCHEMA = _load_file(SCHEMA_DIR / 'CorrectionFactorDataInputs.json')

    def __init__(self, obj: dict | Path | zip):
        DictLike.__init__(self, obj, schema=self.SCHEMA)
        # cast into their containing objects
        for row in self.values():
            row['splitter'] = S11Like.try_from(row['splitter'])
            for key in ['special', 'standard']:
                row[key]['s11'] = S11Like.try_from(row[key]['s11'])
                row[key]['clrm_coeffs'] = KDCLike.try_from(row[key]['clrm_coeffs'])
                row[key]['parsed_calibration'] = ParsedRFSweep(
                    row[key]['parsed_calibration']
                )
            self._cache = {'s11': {}}


# %% Measurement configuration classes
class RFSensorMasterList(DictLike):
    SCHEMA = SCHEMA = load_config(SCHEMA_DIR / 'RFSensorMasterList.json')

    def __init__(self, obj: dict | Path):
        DictLike.__init__(self, obj, schema=self.SCHEMA)


class RFSweepMeasurementDescription(DictLike):
    SCHEMA = _load_schema_from_definition('RFSweepMeasurementDescription')

    def __init__(self, obj: dict | Path):
        DictLike.__init__(self, obj, schema=self.SCHEMA)


class RFSweepLevellingSettings(DictLike):
    SCHEMA = _load_schema_from_definition('RFSweepLevellingSettings')

    def __init__(self, obj: dict | Path):
        DictLike.__init__(self, obj, schema=self.SCHEMA)


class RFSweepSignalConfig(DictLike):
    """Mapping of raw-data columns to a model of each sensor and the calorimter's thermopile element for an RF sweep."""

    SCHEMA = load_config(SCHEMA_DIR / 'RFSweepSignalConfig.json')

    def __init__(self, obj: dict | Path):
        DictLike.__init__(self, obj, schema=self.SCHEMA)


class RFSweepRunSettingsColumns(DictLike):
    SCHEMA = _load_schema_from_definition('RFSweepRunSettingsColumns')

    def __init__(self, obj: dict | Path):
        DictLike.__init__(self, obj, schema=self.SCHEMA)


class RFSweepStatsSettings(DictLike):
    SCHEMA = _load_schema_from_definition('RFSweepStatsSettings')

    def __init__(self, obj: dict | Path):
        DictLike.__init__(self, obj, schema=self.SCHEMA)


class RFSweepOutputSettings(DictLike):
    SCHEMA = _load_schema_from_definition('RFSweepOutputSettings')

    def __init__(self, obj: dict | Path):
        DictLike.__init__(self, obj, schema=self.SCHEMA)


class RFSweepMeasurementMode(DictLike):
    SCHEMA = _load_schema_from_definition(
        'InstrumentRoles', definition_name='RFSweepMeasurementMode'
    )

    def __init__(self, obj: dict | Path):
        DictLike.__init__(self, obj, schema=self.SCHEMA)


class RFSweepConfiguration(DictLike):
    SCHEMA = load_config(SCHEMA_DIR / 'RFSweepConfiguration.json')

    def __init__(self, obj: dict | Path):
        obj['stats_settings'] = RFSweepStatsSettings(obj['stats_settings'])
        obj['output_settings'] = RFSweepOutputSettings(obj['output_settings'])
        obj['measurement_description'] = RFSweepMeasurementDescription(
            obj['measurement_description']
        )
        obj['levelling_settings'] = RFSweepLevellingSettings(obj['levelling_settings'])
        obj['run_settings_columns'] = RFSweepRunSettingsColumns(
            obj['run_settings_columns']
        )
        obj['signal_config'] = RFSweepSignalConfig(obj['signal_config'])
        DictLike.__init__(self, obj, self.SCHEMA)


class DCSweepConfiguration(DictLike):
    SCHEMA = load_config(SCHEMA_DIR / 'DCSweepConfiguration.json')

    def __init__(self, obj: dict | Path):
        DictLike.__init__(self, obj, schema=self.SCHEMA)


class RFSweepParserConfig(DictLike):
    """Defines settings for parsing output data of a parser."""

    SCHEMA = load_config(SCHEMA_DIR / 'RFSweepParserConfig.json')

    def __init__(self, obj: dict | Path):
        DictLike.__init__(self, obj, schema=self.SCHEMA)


# %% misc loader functions
