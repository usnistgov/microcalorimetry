"""
This module containes misceallaneous helper functions.

Specifically, these are functions that aren't intended for exposure
as part of the public API.
"""

import xarray as xr
import numpy as np
from rmellipse.uobjects import RMEMeas
from pathlib import Path
from importlib.metadata import version


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
    out = xr.concat(arrs, dim, join='outer')
    out = out.assign_coords({dim: new_coords})
    return out


def get_version(package: str) -> str:
    """Get the version number of a package."""
    return version(package)


def get_data_record_metadata(
    paths: str | Path | list[Path], metadata_pattern: str = '*metadata*'
) -> list[Path]:
    """
    Take a list of paths and turn it to a list of paths pointing to metadata.

    Paths can be folders or the metadata files themselves.
    """
    metadata = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f'no such thing as {path}')
        if path.is_dir():
            files = [f for f in path.glob(metadata_pattern)]
            if len(files) < 1:
                raise FileNotFoundError(
                    f'No file matching {metadata_pattern} in {str(path)}'
                )
            elif len(files) > 1:
                raise FileNotFoundError(
                    f'More than 1 file matching {metadata_pattern} in {str(path)}'
                )
            path = files[0]
        metadata.append(path)
    return metadata


def get_git_info(path: Path) -> dict:
    """
    Get a dictionary of git metadata.

    Parameters
    ----------
    path : Path
        Walks up the parents of path untila git repo is found. If none
        is found, empty dictionary is returned.

    Returns
    -------
    output : dict
        dicitionary of git repo metadata that contains path. If path is
        not a part of a git repo, then no metadata is given.

    """
    from git import Repo, InvalidGitRepositoryError

    repo = None

    # walk up till I find a directory

    for parent in [Path(__file__)] + list(Path(__file__).parents):
        try:
            repo = Repo(parent)
            name = parent.stem
            print(parent)
            break
        except InvalidGitRepositoryError:
            pass

    output = {}
    if repo is not None:
        try:
            output[f'{name}:branch'] = str(repo.active_branch)
        except Exception as e:
            print(f'Failed to get {name}:branch for {e}')
        try:
            output[f'{name}:head'] = str(repo.head.commit.hexsha)
        except Exception as e:
            print(f'Failed to get {name}:head for {e}')
        try:
            output[f'{name}:unstaged_diffs?'] = len(repo.index.diff(None)) > 0
        except Exception as e:
            print(f'Failed to get {name}:unstaged_diffs? for {e}')
        try:
            output[f'{name}:staged_diffs?'] = len(repo.index.diff('HEAD')) > 0
        except Exception as e:
            print(f'Failed to get {name}:staged_diffs? for {e}')
        for r in repo.remotes:
            try:
                output[f'{name}:remote:{r.name}'] = r.url
            except Exception as e:
                print(f'Failed to get {name}:remote:{r.name} for {e}')

    return output


if __name__ == '__main__':
    import json

    info = get_git_info(__file__)
    print(json.dumps(info, indent=True))
