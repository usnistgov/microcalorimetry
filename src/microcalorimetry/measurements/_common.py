from rminstr.utilities.path import new_dir
from microcalorimetry.math import rfpower
from rmellipse.uobjects import RMEMeas
from rmellipse.propagators import RMEProp
from rminstr.data_structures import ExistingRecord, ExptParameters
from pathlib import Path
from os.path import join, dirname, basename
import microcalorimetry.measurements.rfsweep._parser as microparser
import microcalorimetry.measurements.rfsweep._runner as microrunner
import microcalorimetry.configs as configs
import microcalorimetry._helpers._intf_tools as clitools
from microcalorimetry._tkquick.dtypes import Folder
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import json
import click
from itertools import cycle
from decimal import Decimal
from fnmatch import fnmatch

__all__ = ['view']

def view(
    metadata: Path,
    include_columns:list[str] = ['*'],
    exclude_columns: list[str] = [],
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

    Returns
    -------
    figures : tuple[Figure]
        Tuple of output figures.
    """
    dr = ExistingRecord(metadata)

    d_full = dr.batch_read()
    d_full = {k:v for k,v in d_full.items() if any([fnmatch(k,pattern) for pattern in include_columns])}

    # otherwise, plot each sensors raw time series in a seperate window
    sensor_figs = []

    for k in d_full:
        fig_i, axs_i = plt.subplots(1, 1)
        fig_i.suptitle(f'{k} : Raw Data')
        ts = d_full[k]
        axs_i.plot(ts.t, ts.values, 'o-', ds='steps-post')
        axs_i.set_ylabel(k)
        axs_i.set_xlabel('Time (s)')
        sensor_figs.append(fig_i)

    return tuple(sensor_figs)
