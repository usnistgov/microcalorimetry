from rminstr.data_structures import ExistingRecord
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
from fnmatch import fnmatch
import numpy as np
from microcalorimetry._tkquick.gui_dtypes import Folder

__all__ = ['view']


def _apply_tunit(x: np.array, time_units: str, relative: bool) -> np.array:
    out = x
    print(out.shape)
    if relative:
        out = out - out[0]

    if time_units == 'seconds':
        return out
    elif time_units == 'hours':
        return out / 3600
    elif time_units == 'days':
        return out / 3600 / 24
    elif time_units == 'datetime':
        return np.array([datetime.fromtimestamp(oi) for oi in out])
    else:
        raise NotImplementedError(f'{time_units} not implemented.')


def view(
    datarecord: Folder,
    include_columns: list[str] = ['*'],
    time_units: str = 'datetime',
    relative_time: bool = False,
) -> tuple[plt.Figure]:
    """
    Generic plot columns in a data record.

    Parameters
    ----------
    datarecord : Folder
        Path to the metadata file of an active experiment
    include_columns : list[str]
        Columns to include with glob patterns. Leave as ['*']
        to include all. The default is ['*']
    time_units : str, optional
        Options Format
        --------------
        {
        seconds
            plot in seconds
        hours
            plot in hours
        datetime
            plot in datetime
        }
    relative_time : bool, optional
        Use time since start of record instead of
        absolute time as the x axis.

    Returns
    -------
    figures : tuple[Figure]
        Tuple of output figures.
    """
    metadata = Path(datarecord)
    if metadata.is_dir():
        metadata = [p for p in metadata.glob('*metadata*')][0]

    meta_dir = metadata.parents[0]
    dr = ExistingRecord(metadata)
    t0 = float(dr.metadata['time_zero'])

    d_full = dr.batch_read()
    d_full = {
        k: v
        for k, v in d_full.items()
        if any([fnmatch(k, pattern) for pattern in include_columns])
    }

    if time_units == 'datetime' and relative_time:
        print("Datetime units don't support relative time.")
        relative_time = False

    # otherwise, plot each sensors raw time series in a seperate window
    sensor_figs = []

    # to label the x axis as relative or
    # absolute time
    relative_or_not = {True: 'Relative', False: 'Absolute'}[relative_time]

    for k in d_full:
        try:
            fig_i, axs_i = plt.subplots(1, 1)
            fig_i.suptitle(f'{k} : Raw Data')
            ts = d_full[k]
            time = _apply_tunit(ts.t + t0, time_units, relative_time)
            axs_i.plot(time, ts.values, 'o-', ds='steps-post')
            axs_i.set_ylabel(k)
            axs_i.set_xlabel(f'{relative_or_not} Time ({time_units})')
            sensor_figs.append(fig_i)
        except Exception as e:
            print(f'Failed to plot {k} for - {e}')

    return tuple(sensor_figs)
