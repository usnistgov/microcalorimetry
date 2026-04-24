from rminstr.data_structures import ExistingRecord
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
from fnmatch import fnmatch
import numpy as np
__all__ = ['view']


def _apply_tunit(x: np.array, time_units: str, relative: bool)->np.array:
    out = x

    if relative:
        out = out - out[0]

    if time_units == 'seconds':
        return out
    elif time_units == 'hours':
        return out/3600
    elif time_units == 'days':
        return out/3600/24
    elif time_units == 'datetime':
        return np.array([datetime.fromtimestamp(oi) for oi in out])
    else:
        raise NotImplementedError(f'{time_units} not implemented.')

def view(
    metadata: Path,
    include_columns:list[str] = ['*'],
    exclude_columns: list[str] = [],
    time_units: str = 'seconds',
    relative_time: bool = True
) -> tuple[plt.Figure]:
    """
    Generic plot columns in a data record.

    Parameters
    ----------
    metadata : Path
        Path to the metadata file of an active experiment
    include_columns : list[str]
        Columns to include with glob patters. Leave as ['*']
        to include all.
    time_units : str, optional
        Options Format
        --------------
        {
        seconds
            plot in seconds
        hours
            plot in hourse
        datetime
            plot in datetime
        }
    relative_time : bool, optional
        Use time since start of record on instead of
        absolute time.

    Returns
    -------
    figures : tuple[Figure]
        Tuple of output figures.
    """
    metadata = Path(metadata)
    if metadata.is_dir():
        metadata = [p for p in metadata.glob('*metadata*')][0]
    meta_dir = metadata.parents[0]
    dr = ExistingRecord(metadata)

    d_full = dr.batch_read()
    d_full = {k:v for k,v in d_full.items() if any([fnmatch(k,pattern) for pattern in include_columns])}

    # otherwise, plot each sensors raw time series in a seperate window
    sensor_figs = []

    # to label the x axis as relative or
    # absolute time
    relative_or_not = {
        True:'Relative',
        False:'Absolute'
    }[relative_time]


    for k in d_full:
        fig_i, axs_i = plt.subplots(1, 1)
        fig_i.suptitle(f'{k} : Raw Data')
        ts = d_full[k]
        time = _apply_tunit(ts.t,time_units,relative_time)
        axs_i.plot(time, ts.values, 'o-', ds='steps-post')
        axs_i.set_ylabel(k)
        axs_i.set_xlabel(f'{relative_or_not} Time ({time_units})')
        sensor_figs.append(fig_i)

    return tuple(sensor_figs)
