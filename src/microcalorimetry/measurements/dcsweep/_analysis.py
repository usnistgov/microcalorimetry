from rmellipse.uobjects import RMEMeas
from rmellipse.propagators import RMEProp
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import microcalorimetry.configs as configs
import importlib

from rminstr.data_structures import ExistingRecord, ExptParameters, TimeSeries
from pathlib import Path
import rminstr_specs.K2450 as kspecs
import rminstr_specs.HP34420A as nvmspecs
import rminstr_specs
import matplotlib as mpl
from dataclasses import dataclass
from typing import (
    Protocol,
    SupportsFloat,
    Tuple,
    Literal,
)
type FloatType = (
    type[SupportsFloat]
    | np.dtypes.Float16DType
    | np.dtypes.Float32DType
    | np.dtypes.Float64DType
)


class TimeSeriesProtocol[T](Protocol):
    shape: Tuple[int]
    dims: Tuple[Literal['time']]
    dtype: T


@dataclass
class RunInput:
    path: str | Path
    Tbath: float
    r_heater: float

class DCCalibrationData(Protocol):
    """Protocol for an xarray.Dataset"""
    # variables
    heater_v: TimeSeriesProtocol[FloatType]
    heater_i: TimeSeriesProtocol[FloatType]
    heater_p: TimeSeriesProtocol[FloatType]
    heater_r: TimeSeriesProtocol[FloatType]
    e: TimeSeriesProtocol[FloatType]
    pwr_setting: TimeSeriesProtocol[FloatType]
    env_temp: TimeSeriesProtocol[FloatType]
    steps: TimeSeriesProtocol[FloatType]# [IntType]
    # coordinates
    time: TimeSeriesProtocol[FloatType]

class DCCalibrationDataTempWithThermometer(DCCalibrationData,Protocol):
    """Protocol for an xarray.Dataset"""
    # variables
    therm_v: TimeSeriesProtocol[FloatType]
    therm_i: TimeSeriesProtocol[FloatType]
    therm_p: TimeSeriesProtocol[FloatType]
    therm_r: TimeSeriesProtocol[FloatType]


def colorbar(fig: plt.Figure, ax: plt.axes,  values: np.array, map: str = "jet", label: str = ""):
    cmap = plt.get_cmap("jet", len(values))
    norm = mpl.colors.Normalize(vmin=min(values), vmax=max(values))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax = ax, label = label)
    return cmap

def trim(ds: DCCalibrationData, trim: list[xr.DataArray]) -> DCCalibrationData:
    out = ds
    for condition in trim:
        out = out.where(condition, drop = True)
    return out

def mean_sub(x: np.array):
    return x - np.mean(x)

def zero_one_norm(x: np.array):
    out = x - np.min(x)
    out = out/np.max(out)
    return out

def zero_hrs(t, t0):
    return (t - t0)/3600

# %% Version 0 Stuff
# These are functions and classes for parsing the version 1 of the experiment.

def read_experiment_v0(metadata_path: str, settings: str, meas_list: str):
    """
    Read a measurement.

    Parameters
    ----------
    metadata_path : str
        Path to metadata file. The default is None.
    settings : str
        Path to settings file. The default is None.
    meas_list : str
       Path to measurement list file. The default is None.


    Returns
    -------
    raw : dict[tuples]
        Raw Data.
    ep : parameter_tree
        Measurement settings.

    """
    # read data record
    data = ExistingRecord(metadata_path, maxlen=int(1e7)).batch_read()

    # read config file
    ep = ExptParameters(settings, meas_list)

    # make some attribute
    return data, ep

def timeseries_to_dataarray(ts: TimeSeries) -> xr.DataArray:
    arr = xr.DataArray(
        ts.values,
        dims = ('time'),
        coords = {'time':ts.t}#-ts.t[0]}
    )
    return arr


def mean_settled_by_group(
        x: xr.DataArray, 
        group_arr: xr.DataArray,
        values: xr.DataArray,
        window: float        
    ) -> tuple[xr.DataArray,xr.DataArray,list[xr.DataArray]]:
    """
    Return mean and average of settled samples in x corresponding to group_arr.
    
    Where samples in group_arr == each value in values, takes the values
    in the provied window of x.

    Parameters
    ----------
    x : xr.DataArray
        Values to take mean and average of.
    group_arr : xr.DataArray
        Array alligned to x that identifies the groups.
    values : xr.DataArray
        Values in group_arr to use.
    window : float
        Time window to use.

    Returns
    -------
    mean : xr.DataArray
        mean of samples per group.
    std : xr.DataArray
        std of samples per group.
    samples : list[xr.DataArray]
        samples used in each group
    """
    avg = []
    std = []
    times= []
    samples = []
    for osn in values:
        # print(osn)
        group = x[group_arr == osn]
        # pick out settled values and do mean/std
        settled = group[
            group.time > group.time[-1] - window
            ]
        samples.append(settled)
        avg.append(np.mean(settled))
        std.append(np.std(settled))
        times.append(np.mean(settled.time))
    output = []
    for i in [avg, std]:
        output.append(xr.DataArray(
            i,
            dims = ('time'),
            coords = {'time':times}
            ))
    return *output, samples

def metered_to_linmeas(
        name: str,
        mean: xr.DataArray,
        std: xr.DataArray | None = None,
        specs: rminstr_specs.Specification | None = None, 
        umech_prefix: str = '',
        origin: str = ''
        ) -> RMEMeas:
    """
    Turn an array of metered data to a RMEmeas object with lin prop uncertainties.
    
    Provide a 1d array of average and optionally std of the averages. Can
    provide a data sheet specification as well. Will generate an RMEMeas object
    with correlated data sheet uncertainties and uncorrelated std uncertainties.
    
    Parameters
    ----------
    name : str
        Name to give to the meas object.
    mean : xr.DataArray
        Average values.
    std : xr.DataArray | None, optional
        Standard deviation of measurements. The default is None.
    specs : rminstr_specs.Specification | None, optional
        Specsheet to use. The default is None.
    umech_prefix : str, optional
        Prefix to give to umech_ids. The default is ''.

    Returns
    -------
    RMEMeas
        Contains correlated linear uncertainties of provided metereed
        data.
    """
    
    # if std provided, generate in a fast way since add_umech is a little slow 
    if std is None:
        meas: RMEMeas = RMEMeas.from_nom(name, mean)
    else:
        umechs = ['nominal']
        for i,si in  enumerate(std):
            umechs.append(f'{umech_prefix}_std_{i}')
        cov = mean.expand_dims({'umech_id':umechs}, axis = 0).copy()
        perts =  np.diag(std).copy()
        perts = np.vstack((std*0,perts))
        cov= cov+perts
        meas = RMEMeas(name, cov = cov)
        meas.make_umechs_unique(same_uid=True)
        meas.assign_categories_to_all(Origin = origin + ' Noise', Type = 'A')
        
    
    if specs is not None:
        unc_array = specs.all_manufacturer_errors(mean.values)
        pert = mean + unc_array
        umech_i= f'{umech_prefix}_spec_{specs.name}'
        meas.add_umech(
            umech_i, pert,add_uid = True,category = {'Type':'B','Origin':origin + ' Traceability'}
            )

    return meas

# %% Parsing functions
# parsing functions should taking data paths and options
# and return a configs.ParsedDC object and a list of figures for review.

def parse_v0(
    metadata_path: str,
    settings: str,
    meas_list: str,
    on_time_window: float = 300,
    off_time_window: float = 300,
    heater_instr_name: str = 'SMU',
    sensor_instr_name: str = 'NVM',
    zero_threshhold: float = 1e-6,
    transition_threshhold_watts: float = 0.1e-4,
    ) -> tuple[configs.ParsedDCSweep, tuple[plt.Figure]]:
    """
    This is a parsing function for the original draft of the dcsweep measurement.
    
    This draft of the measurement did not include a thermometer, and did not
    assume that the heater and sensors time series data were alligned by samples.
    Since the settings were not recorded as a function of time, this function
    requires a guess about the step sizes and what measured powers should be
    considered off in order to properly select samples out of the time series.

    Parameters
    ----------
    metadata_path : str
        Path to measurement.
    settings : str
        Path to measurement settings.
    meas_list : str
        Path to measlist.
    on_time_window : float, optional
        Time window for selecting on samples. The default is 300.
    off_time_window : float, optional
        Time window for selecting off samples. The default is 300.
    heater_instr_name : str, optional
        Name of heater instrument. The default is 'SMU'.
    sensor_instr_name : str, optional
        Name of sensor instrument. The default is 'NVM'.
    zero_threshhold : float, optional
        Threshhold (in Watts) of samples below this where the heater
        is believed to be turned off. The default is 1e-6.
    transition_threshhold_watts : float, optional
        Thresh hold (in Watts) where changes in the applied power to the heater
        constitutes a new step in the sweep. The default is 0.1e-4.

    Returns
    -------
    e : RMEMeas
        Offset corrected voltage of the sensors thermoelectric element.
    v : RMEMeas
        Offset corrected voltage of the heating element.
    i : RMEMEas
        Offset corrected current of the heating element.
    tuple[plt.Figure]
        Tuple of plt.Figure objects detailing the analysis.

    Raises
    ------
    ValueError
        Fails to parse.
    """
    
    print(metadata_path)
    data, ep = read_experiment_v0(metadata_path, settings, meas_list)
    
    # get the source values from the config
    # tells us how many source settings and
    # the expected number of transitions between source
    # settings but not where they are in the data record 
    # (sorry)
    all_source_settings = ep.get_column(ep.columns[0])
    mask = np.append([True], np.diff(all_source_settings) != 0)
    source_settings = all_source_settings[mask]
    expected_n_transitions = len(source_settings) - 1

    heater_v = timeseries_to_dataarray(data[f'V_{heater_instr_name} (V)'])
    heater_i = timeseries_to_dataarray(data[f'I_{heater_instr_name} (A)'])
    e = timeseries_to_dataarray(data[f'V_{sensor_instr_name} (V)'])

    # trim smu data to smallest size in case the metadata only dumped some of
    # voltage or current for a step
    min_heater_size = min([value.shape[0] for value in [heater_v, heater_i]])
    heater_v = heater_v[:min_heater_size]
    heater_i = heater_i[:min_heater_size]

    # calculate pwr
    heater_p = heater_v * heater_i



    # figure out where the transitions are  
    transitions = np.append([0],abs(np.diff(heater_p))) > transition_threshhold_watts
    n_transitions = np.where(transitions)[0].shape[0]
    if n_transitions != expected_n_transitions:
        raise ValueError(
            f"""Did not find the right number of source transitions
            (found {n_transitions} instead of {expected_n_transitions}).
            Try adjusting the transition_threshhold_watts (currently
            {transition_threshhold_watts})"""
            )
    
    # create arrays of time since adjustment
    start = 0
    # go back 2 samples for safety
    where_transitions= np.where(transitions)[0]
    adjust_times = np.zeros(len(heater_p))
    step_numbers = np.zeros(len(heater_p),dtype = int)
    for i, ti in enumerate(np.append(where_transitions,len(heater_p))):
        end = ti
        times = heater_p.time[start:ti]
        adjust_times[start:ti] = heater_p.time[start:ti] - heater_p.time[start]
        step_numbers[start:ti] = i
        start = ti

    # arrays of adjustment times and numbered steps
    heater_adjust_times = xr.DataArray(
        adjust_times,
        dims = ('time'), coords = {'time':heater_p.time}
        )
    heater_step_numbers = xr.DataArray(
        step_numbers,
        dims = ('time'), coords = {'time':heater_p.time}
        )

    # shift the sensor adjust times back by 1 sample inc case they are out
    # of sync by 1 sample
    sensor_adjust_times = heater_adjust_times.interp(time = e.time, method= 'nearest') \
        .shift(time = -2)
    sensor_adjust_times[-2:] = sensor_adjust_times[-3]
    sensor_step_numbers = heater_step_numbers.interp(time = e.time, method = 'nearest')\
        .shift(time = -2)
    sensor_step_numbers[-2:] = sensor_step_numbers[-3]


    # timestamps in hours
    t0 = e.time[0]
    te_hrs = (e.time-e.time[0])/3600
    h_hrs = (heater_v.time - heater_v.time[0])/3600

    # make a figure to start plotting analysis
    # this will get added to through the rest
    # of this function
    fig,ax= plt.subplots(2,1, sharex = True)
    # fig2,ax2 = plt.subplots(1,1, sharex = True)
    l0 = []
    h0 = []
    l1 = []
    h1 = []
    
    # ax = [ax1,ax2]
    # for us in np.unique(sensor_step_numbers):
    #     ind_sensor = sensor_step_numbers == us
    #     ind_heater = heater_step_numbers == us
    rawe_line, = ax[0].plot(zero_hrs(e.time, t0), e*1e3)
    rawp_line, = ax[1].plot(zero_hrs(heater_p.time, t0), heater_p*1000)
    # ax[0].set_xlabel('Time (hrs)')
    ax[1].set_xlabel('Time (hrs)')
    ax[0].set_ylabel(r'$e\:\left(\mathrm{mV}\right)$')
    ax[1].set_ylabel(r'$P_{dc}\:\left(\mathrm{mW}\right)$')
    heater_v = xr.DataArray(heater_v)
    
    l0.append(('Raw Data'))
    h0.append(rawe_line)
    
    l1.append(('Raw Data'))
    h1.append(rawp_line)
    
    # final values
    ons = xr.Dataset()
    offs = xr.Dataset()

    # get all the on values

    on_step_numbers = np.unique(heater_step_numbers[heater_p > zero_threshhold])
    
    # get on values
    e_ons, e_ons_std, e_on_samples = mean_settled_by_group(
        e,
        sensor_step_numbers,
        values = on_step_numbers,
        window = on_time_window
        )
    
    
    for sample in e_on_samples:
        ax[0].plot(zero_hrs(sample.time,t0), sample*1000,'k')
    
    e_ons_line, = ax[0].plot(zero_hrs(e_ons.time,t0), e_ons*1000, 'ko')
    l0.append(('On Samples'))
    h0.append(e_ons_line)
    
    # get the heater v ons
    heater_v_ons, heater_v_ons_std,  heater_v_ons_samples = mean_settled_by_group(
        heater_v,
        heater_step_numbers,
        values = on_step_numbers,
        window = on_time_window
        )
    
    # get the heater i ons
    heater_i_ons, heater_i_ons_std,  heater_i_ons_samples = mean_settled_by_group(
        heater_i,
        heater_step_numbers,
        values = on_step_numbers,
        window = on_time_window
        )
    
    
    p_ons_line, = ax[1].plot(zero_hrs(heater_v_ons.time, t0), heater_v_ons*heater_i_ons*1000,'ko')
    l1.append(('On Samples'))
    h1.append(p_ons_line)
    for v,i in zip(heater_v_ons_samples, heater_i_ons_samples):
        ax[1].plot(zero_hrs(v.time,t0), v*i*1000,'k')


    # get the off measurements
    off_step_numbers = np.unique(heater_step_numbers[heater_p < zero_threshhold])

    # get mean off values
    e_offs_mn, e_offs_std, e_off_samples = mean_settled_by_group(
        e,
        sensor_step_numbers,
        values = off_step_numbers,
        window = off_time_window
        )
    for sample in e_off_samples:
        ax[0].plot(zero_hrs(sample.time,t0), sample*1000,'kx')

    e_offs_line, = ax[0].plot(zero_hrs(e_offs_mn.time,t0), e_offs_mn*1000,'kx')
    l0.append(('Off Samples'))
    h0.append(e_offs_line)
    
    # interpolate to the on times
    e_offs = e_offs_mn.interp(time = e_ons.time)
    # ax[0].plot(zero_hrs(e_offs.time,t0), e_offs*1000,'ko--')
    
    
    # get the heater v offs
    heater_v_offs, heater_v_offs_std,  heater_v_offs_samples = mean_settled_by_group(
        heater_v,
        heater_step_numbers,
        values = off_step_numbers,
        window = off_time_window
        )
    
    # get the heater i offs
    heater_i_offs, heater_i_offs_std,  heater_i_offs_samples = mean_settled_by_group(
        heater_i,
        heater_step_numbers,
        values = off_step_numbers,
        window = off_time_window
        )
    
    
    for v,i in zip(heater_v_offs_samples, heater_i_offs_samples):
        p_offs_line, = ax[1].plot(zero_hrs(v.time,t0), v*i*1000,'kx')    
    l1.append(('Off Samples'))
    h1.append(p_offs_line)
    
    # for v,i in zip(heater_v_ons_samples, heater_i_ons_samples):
    #     ax[1].plot(zero_hrs(v.time,t0), v*i*1000,'kx')
    
    heater_v_offs = heater_v_offs.interp(time = heater_v_ons.time)
    heater_i_offs = heater_i_offs.interp(time = heater_i_ons.time)


    # make spec sheets, assume worst case which is default
    heater_i_specs = kspecs.DatasheetMeasureDCI('K2450',serial = 'xxx',suppress_warnings = True)
    heater_v_specs = kspecs.DatasheetMeasureDCV('K2450',serial = 'xxx',suppress_warnings = True)
    e_specs = nvmspecs.DatasheetDCV('HP34420A',serial = 'xxx',suppress_warnings = True)
    
    # put in to RMEMEas objects with correlated uncertainties
    e_ons = metered_to_linmeas('e_on',e_ons,std = e_ons_std,specs = e_specs, umech_prefix= 'DC Sweep', origin= 'DC Sweep')
    e_offs = metered_to_linmeas('e_off',e_offs,specs = e_specs, umech_prefix= 'DC Sweep', origin= 'DC Sweep')
    
    heater_v_ons = metered_to_linmeas(
        'heater_v_on',heater_v_ons,std = heater_v_ons_std,specs = heater_v_specs, umech_prefix= 'DC Sweep', origin= 'DC Sweep')
    heater_v_offs = metered_to_linmeas(
        'heater_v_off',heater_v_offs,specs = heater_v_specs, umech_prefix= 'DC Sweep', origin= 'DC Sweep')
    
    heater_i_ons = metered_to_linmeas(
        'heater_i_on',heater_i_ons,std = heater_i_ons_std,specs = heater_i_specs, umech_prefix= 'DC Sweep', origin= 'DC Sweep')
    heater_i_offs = metered_to_linmeas(
        'heater_i_off',heater_i_offs,specs = heater_i_specs, umech_prefix= 'DC Sweep', origin= 'DC Sweep')
    
    
    # correct for the offset
    prop = RMEProp(sensitivity = True)
    @prop.propagate
    def sub(x,y): return x - y
    
    @prop.propagate
    def mul(x,y): return x*y
    
    e = sub(e_ons, e_offs)
    heater_v = sub(heater_v_ons, heater_v_offs)
    heater_i = sub(heater_i_ons, heater_i_offs)
    p = mul(heater_v, heater_i)
    
    # ax[0].legend(h0,l0,loc = 'best')
    # ax[1].legend(h1,l1,loc = 'best')
    fig.subplots_adjust(top = 0.95)
    # h, l = ax[-1].get_legend_handles_labels()
    fig.legend(h0,l0,loc='center', bbox_to_anchor=(0.5, 1), ncol = 3)
    
    return e, heater_v, heater_i, fig


# %% Version 1 Stuff
# These are functions and classes for parsing the version 1 of the experiment.


def read_v1(
    runs: list[Path],
    thermopile_monitor: str,
    thermometer_monitor: str,
    heater: str
    ) -> DCCalibrationData:
    """
    Reads version 1 of the DC sweep experiment in an XArrayDataSet.
    
    Can take in multiple runs.

    Parameters
    ----------
    runs : list[Path]
        List of folders containing or metadata files them selves from 
        dc sweep runs.
    thermopile_monitor : str, optional
        Name of thermopile instrument.
    thermometer_monitor : str, optiona
        Name of thermometer instrument.
    heater : str, optional
        Name of heater instrument.

    Returns
    -------
    DCCalibrationData
        Dataset containing raw data on an alligned coordinate set.

    Raises
    ------
    ValueError
        If e_col isn't present.
    """
    all_data = {}
    e_col = f'V_{thermopile_monitor} (V)'
    
    def try_get(d, key, p: str):
        try:
            return d[key]
        except KeyError as e:
            raise KeyError(f'expected {key} field in config file {p}. Not present, add manually.') from e
    
    for file in runs:
        print(file)
        file = Path(file)
        if not file.exists():
            raise FileNotFoundError(f'no such thing as {file}')
        if file.is_dir():
            file = [f for f in file.glob('*metadata*')][0]
            print('found', file, 'in', file.parent)
        try:
            config = [f for f in file.parent.glob('*settings*')][0]
        except IndexError:
            raise FileNotFoundError(f'Cant find *settings* file in {file.parent}')
        print(config)
        cfile = str(config)
        # read in data and configuration
        data = ExistingRecord(file).batch_read()
        config = ExptParameters(config).config
        
        # get config file inputs that should be present to do analysis
        Tbath = float(try_get(config, 't_bath_c', cfile))
        r_heater = float(try_get(config, 'r_heater_ohms', cfile))
    
        # trim data to matching lengths
        min_shape = min(v.values.shape[0] for v in data.values())
        data = {k:TimeSeries(v.t[:min_shape],v.values[:min_shape]) for k,v in data.items()}
        for k,v in data.items():
            # print(k, v.t.shape)
            try:
                all_data[k] = np.append(all_data[k], data[k].values)
            except KeyError:
                all_data[k] = data[k].values
        try:
            all_data['bath'] = np.append(all_data['bath'], data[k].values*0 + Tbath)
        except KeyError:
            all_data['bath'] = data[k].values*0 + Tbath

        try:
            all_data['Timestamp (s)'] = np.append(all_data['Timestamp (s)'], data[k].t)
        except KeyError:
            all_data['Timestamp (s)'] = data[k].t
        
        pwr = data['SOURCE_SETTING (A)'].values**2*r_heater
        
        try:
            all_data['pwr_setting'] = np.append(all_data['pwr_setting'], pwr)
        except KeyError:
            all_data['pwr_setting'] = pwr
        
    try:
        e = (['time'],all_data[e_col])
    except KeyError:
        raise ValueError(f'{e_col} not in {list(all_data.keys())}')
    # put into a dataset
    data = xr.Dataset(
        data_vars = dict(
            heater_v =  (['time'],all_data[f'V_{heater} (V)']),
            heater_i = (['time'],all_data[f'I_{heater} (A)']),
            heater_p = (['time'],all_data[f'I_{heater} (A)']*all_data[f'V_{heater} (V)']),
            heater_r = (['time'],all_data[f'I_{heater} (A)']/all_data[f'V_{heater} (V)']),
            e = e,
            pwr_setting =  (['time'],all_data['pwr_setting']),
            adjust_time = (['time'],all_data['time_since_source_adjust (s)']),
            env_temp = (['time'],all_data['bath'])
        ),
        coords = dict(
            time = all_data['Timestamp (s)'],
        )
    )

    if thermometer_monitor:
        data = data.assign(
            therm_i = (['time'],all_data[f'I_{thermometer_monitor} (A)']),
            therm_v = (['time'],all_data[f'V_{thermometer_monitor} (V)']),
            therm_p = (['time'],all_data[f'V_{thermometer_monitor} (V)']*all_data[f'I_{thermometer_monitor} (A)']),
            therm_r = (['time'],all_data[f'V_{thermometer_monitor} (V)']/all_data[f'I_{thermometer_monitor} (A)'])
            )

    # make something that tracks where the steps happened
    where_steps =  np.where(np.diff(data.pwr_setting, prepend = False) != 0)[0]
    where_steps = np.append(where_steps,len(data.time))
    steps = np.zeros(data.time.shape, dtype = int)
    follow = 0
    for i,crawl in enumerate(where_steps):
        steps[follow:crawl] = i
        follow = crawl
    data = data.assign(
        steps = xr.DataArray(
            steps,
            dims = ('time',),
            coords = dict(time = data.time)
        )
    )

    return data, config





def review_plot_v1(
        var_ls: list[str], 
        raw: DCCalibrationData, 
        corr: DCCalibrationData | None = None,
        offs: DCCalibrationData | None = None,
        offs_samples: DCCalibrationData | None = None,
        ons: DCCalibrationData | None = None,
        ylabels: list[str | None] = None,
        yscales: list[float] = 1) -> plt.Figure:
    if isinstance(var_ls, str):
        var_ls = [var_ls]
    if isinstance(ylabels, str) or ylabels is None:
        ylabels = [ylabels]
    if isinstance(yscales,float) or yscales is None:
        yscales = [yscales]
        
    fig,axs = plt.subplots(len(var_ls),1)
    if len(var_ls) == 1:
        axs = [axs]
    for var, ax, yscale, ylabel in zip(var_ls, axs, yscales, ylabels):
        handles = []
        labels = []
        t0 = raw.time[0]
        def zero(x):
            return (x - t0)/3600
        ax.plot(zero(raw.time), raw[var]*yscale,'-',label = 'Raw Data')
        if ons is not None:
            ax.plot(zero(ons.time), ons[var]*yscale, 'ko', label = 'On Samples')
        if corr is not None:
            ax.plot(zero(corr.time), corr[var]*yscale,'*',label = 'Offset Corrected')
        if offs_samples is not None:
            ax.plot(zero(offs_samples.time), offs_samples[var]*yscale,'kx', label = 'Off Samples')
    
        # if offs is not None:
        #     ax.plot(zero(offs.time), offs[var]*yscale,'k.', label = 'Offs Interp')
    
        # cbar.set_ticklabels([str(v) for v in unq_pwr])
        if ylabel is None:
            ax.set_ylabel(var)
        else:
            ax.set_ylabel(ylabel)
    fig.subplots_adjust(top = 0.95)
    h, l = axs[-1].get_legend_handles_labels()
    fig.legend(h,l,loc='center', bbox_to_anchor=(0.5, 1), ncol = 3)
    # ax.legend(loc = 'best')
    axs[-1].set_xlabel(r'Time (hrs)')
    # if ons is not None:
    #     # pick the closeses on value at 10
    #     i = np.searchsorted(ons.time, np.max(np.diff(zero(ons.time)))*10,side = 'right')
    #     ax.set_xlim(0,zero(ons.time)[i])
    return fig

def pick_window(x: DCCalibrationData, window: tuple[float, float]) -> DCCalibrationData:
    """
    Takes in a DCCalibrationDatase step and selects a time window.

    Parameters
    ----------
    x : DCCalibrationData
        DESCRIPTION.
    window : tuple[float, float] | tuple[float]
        positive numbers are relative to the beginning of the step,
        negative numbers are relative to the end of the step.
        [5, 100] = 5 to 100 seconds into the step
        [-100, 0], last 100 seconds of the step
        [-100, -5], last 95 seconds of the step.
    Returns
    -------
    DCCalibrationData
        indexed DCCalibrationData based on the time window.
    """

    # in case there are consecutive steps
    # with the same power level, concat them to a single step
    step_changes =  np.where(np.diff(x.adjust_time) < 0)[0]
    x2 = x.copy()
    for change in step_changes:
        x2.adjust_time[change+1:]+= x2.adjust_time[change]

    end = x.adjust_time.max()
    min_wait_time = float(window[0])    

    if min_wait_time < 0:
        min_wait_time = end + min_wait_time

    max_wait_time = float(window[1])
    if max_wait_time <= 0:
        max_wait_time = end + max_wait_time
    

    # print(float(min_wait_time), float(max_wait_time))


    out = x.where(
        (x.adjust_time >= min_wait_time) & \
        (x.adjust_time <= max_wait_time),
        drop = True
    ) 

    return  out

def parse_v1(
    *args, 
    metadata: list[Path | str],
    thermometer_monitor: str,
    thermopile_monitor: str,
    heater: str,
    on_min_wait: float,
    on_window: tuple[float, float] | tuple[float],
    off_min_wait: float,
    off_window: tuple[float, float] | tuple[float], 
    min_pwr_setting: float,
    ):
    """
    Parse version 1 of the DC sweep experiment.
    """


    min_pwr_setting = float(min_pwr_setting)
   
    data, config = read_v1(
        metadata, 
        heater = heater,
        thermopile_monitor = thermopile_monitor,
        thermometer_monitor = thermometer_monitor,
        )



    # Index into on samples using the on time window
    on_steps = data.where(
       (data.pwr_setting > 0) & \
       (data.pwr_setting >= float(min_pwr_setting)) & \
       (data.adjust_time >= on_min_wait),
       drop = True
       )
    ons = on_steps.groupby(on_steps.steps).map(pick_window, args = (on_window,))


    # index into the off samples using the off time window
    
    # pick off values and interpolate to the
    # on times
    offs_steps = data.where(
        (data.pwr_setting == 0) & \
        (data.adjust_time >= off_min_wait),
        drop =True
        )
    offs_samples = offs_steps.groupby(offs_steps.steps).map(pick_window, args = (off_window,))
        




    # interpolate, use average value if not
    # enough zero measurements are available to interpolate.
    offs = ons.copy()
    for v in offs_samples:
        offs[v] = offs_samples[v].interp(
        time = ons.time,
        kwargs = dict(fill_value = offs_samples[v].mean())
        )

        
    # keep raw data around
    raw = data.copy()

    # make data offset corrected measurements
    zero = ['e','heater_i','heater_v','heater_p']
    data = ons.copy()
    for name, var in data.items():
        if name in zero:
            data[name] = var - offs[name]


    unq_pwr = np.unique(data.pwr_setting)


    cmap = plt.cm.viridis
    # make review plots
    figs = []
    fig = review_plot_v1(
        ['heater_p','e'],
        raw = raw,
        # corr = data,
        # offs = offs,
        offs_samples=offs_samples,
        ons = ons,
        ylabels = [r'$P_{dc}\:\left(\mathrm{mW}\right)$', r'$e$ (mV)'],
        yscales = [1e3, 1e3]
    )
    figs.append(fig)

    if thermometer_monitor:
        fig = review_plot_v1(
            'therm_r',
            raw,
            ons = ons,
            # corr = data,
            ylabels = r'$R_{thermometer}\:\left(\mathrm{k}\Omega\right)$',
            yscales = 1e-3
        )
        figs.append(fig)
    # fig = review_plot_v1(
    #     'heater_p',
    #     raw,
    #     ons = ons,
    #     offs_samples=offs_samples,
    #     ylabel = r'$P_{dc}\:\left(\mathrm{mW}\right)$',
    #     yscale = 1e3
    # )
    # figs.append(fig)


    fig,ax = plt.subplots(1,1)
    # color by approximate unique powers
    scatter = ax.scatter(data.heater_p*1e3, data.e, c= data.env_temp, cmap = cmap)
    fig.colorbar(scatter, label = 'Ambient Temperature (°C)')
    ax.set_xlabel(r'$P_{dc}\:\left(\mathrm{mW}\right)$')
    ax.set_ylabel(r'$e-e_0\:\left(\mathrm{mV}\right)$')
    figs.append(fig)

    # fig,ax = plt.subplots(1,1)
    # # color by approximate unique powers
    # colors = colorbar(fig,ax, unq_pwr, label = 'Power Setting (mW)')
    # for i, pi in enumerate(unq_pwr):
    #     di = data.where(data.pwr_setting == pi)
    #     # print(color)
    #     ax.plot(di.heater_p*1000, di.e*1e3,'o',color = colors(i))

    # # cbar.set_ticklabels([str(v) for v in unq_pwr])
    # ax.set_xlabel(r'$P_{dc}$ (mW)')
    # ax.set_ylabel(r'$e-e_0$ (mV)')
    # figs.append(fig)

    # if thermometer_monitor:
    #     fig,ax = plt.subplots(1,1)
    #     # color by approximate unique powers
    #     colors = colorbar(fig,ax, unq_pwr*1e3, label = 'Power Setting (mW)')
    #     for i, pi in enumerate(unq_pwr):
    #         di = data.where(data.pwr_setting == pi)
    #         # print(color)
    #         ax.plot(di.therm_r/1000, mean_sub(di.e)*1e6,'o',color = colors(i))
    #     ax.set_xlabel(r'$R_{thermometer}\:\left(\mathrm{k}\Omega\right)$')
    #     ax.set_ylabel(r'$e-e_0 $ - $\mu_{e-e_0}\:\left(\mu\mathrm{V}\right)$ by Power Setting')
    #     figs.append(fig)

    fig,ax = plt.subplots(1,1)
    # color by approximate unique powers
    scatter = ax.scatter(data.e*1e3, data.e/data.heater_p, c= data.env_temp, cmap = cmap)
    fig.colorbar(scatter, label = 'Ambient Temperature (°C)')
    ax.set_xlabel(r'$e-e_0\:\left(\mathrm{mV}\right)$')
    ax.set_ylabel(r'$\frac{e-e_0}{P_{dc}}\:\left(\frac{\mathrm{V}}{\mathrm{W}}\right)$')
    figs.append(fig)
    
    fig,ax = plt.subplots(1,1)
    # color by approximate unique powers
    scatter = ax.scatter(data.heater_p*1e3, data.e/data.heater_p, c= data.env_temp, cmap = cmap)
    fig.colorbar(scatter, label = 'Ambient Temperature (°C)')
    ax.set_xlabel(r'$P_{dc}\:\left(\mathrm{mW}\right)$')
    ax.set_ylabel(r'$k^{dc}\:\left(\frac{\mathrm{V}}{\mathrm{W}}\right)$')
    figs.append(fig)

    # if thermometer_monitor:
    #     fig,ax = plt.subplots(1,1)
    #     # color by approximate unique powers
    #     scatter = ax.scatter(data.e*1e3, data.therm_r, c= data.env_temp, cmap = plt.cm.jet)
    #     fig.colorbar(scatter, label = 'Ambient Temperature')
    #     ax.set_xlabel(r'e (mV)')
    #     ax.set_ylabel(r'$R_{thermometer}\:\left(\Omega\right)$')
    #     figs.append(fig)

    # fig,ax = plt.subplots(1,1)
    # # color by approximate unique powers
    # scatter = ax.scatter(data.e*1e3, data.e/data.heater_p, c= data.heater_p, cmap = plt.cm.jet)
    # fig.colorbar(scatter, label = r'$P_{dc}\:\left(\mathrm{mW}\right)$')
    # ax.set_xlabel(r'$e-e_0\:\left(\mathrm{mV}\right)$')
    # ax.set_ylabel(r'$\frac{e-e_0}{P_{dc}}$')
    # figs.append(fig)

    fig,ax = plt.subplots(1,1)
    # color by approximate unique powers
    scatter = ax.scatter(data.e*1e3, data.e/data.heater_p, c= data.adjust_time/3600, cmap = cmap)
    fig.colorbar(scatter, label = 'Sample Time (hrs)')
    ax.set_xlabel(r'$e-e_0\:\left(\mathrm{mV}\right)$')
    ax.set_ylabel(r'$\frac{e-e_0}{P_{dc}}\:\left(\frac{\mathrm{V}}{\mathrm{W}}\right)$')
    figs.append(fig)

    if thermometer_monitor:
        fig,ax = plt.subplots(1,1)
        # color by approximate unique powers
        scatter = ax.scatter(data.e*1e3, data.e/data.heater_p, c= data.therm_r/1000, cmap = cmap)
        fig.colorbar(scatter, label = r'$R_{\mathrm{T}}\:\left(\mathrm{k\Omega}\right)$')
        ax.set_xlabel(r'$e-e_0\:\left(\mathrm{mV}\right)$')
        ax.set_ylabel(r'$k_{dc}\:\left(\frac{\mathrm{V}}{\mathrm{W}}\right)$')
        figs.append(fig)
        
    # spec sheets
    def import_spec(name: str, class_name: str, fallback_model: str) -> type:
        print(f'Initializing specs for {name}')
        try:
            model = config['instruments'][name]['model']
            module = importlib.import_module( f"rminstr_specs.{model}")
            print(f"  specsheet is {model}.{class_name}")
        except (KeyError, ModuleNotFoundError) as e:
            print(f"  Failed to get specs for error : {type(e).__name__} : {e}")
            print(f"  Falling back to {fallback_model}")
            module = importlib.import_module( f"rminstr_specs.{fallback_model}")
        return getattr(module, class_name)
        

    heater_i_specs = import_spec(heater, 'DatasheetMeasureDCI', fallback_model = 'K2450')(
        'heater_i',serial = 'xxx',suppress_warnings = True
        )
    heater_v_specs = import_spec(heater, 'DatasheetMeasureDCV', fallback_model = 'K2450')(
        'heater_V',serial = 'xxx',suppress_warnings = True
        )

    e_specs = import_spec(thermopile_monitor, 'DatasheetDCV', fallback_model = 'HP34420A')(
       'thermopile_monitor',serial = 'xxx',suppress_warnings = True
       )
    
    therm_i_specs = None
    therm_v_specs = None
    if thermometer_monitor:
        therm_i_specs = import_spec(thermometer_monitor, 'DatasheetMeasureDCI', fallback_model = 'K2450')(
            'thermometer_monitor_i',serial = 'xxx',suppress_warnings = True
            )
        therm_v_specs =  import_spec(thermometer_monitor, 'DatasheetMeasureDCV', fallback_model = 'K2450')(
            'thermometer_monitor_v',serial = 'xxx',suppress_warnings = True
            )

    # put on values in rme meas objects
    on_step_groups = ons.groupby(ons.steps)
    on_means = on_step_groups.mean()
    on_stds = on_step_groups.std(ddof = 1)
    
    # off values should also be grouped by the
    # on value step numbers
    off_step_groups = offs.groupby(ons.steps)
    off_means = off_step_groups.mean()
    off_stds = off_step_groups.std(ddof = 1)

    measurements = {}
    spec_sheets = {
            'heater_v': heater_v_specs,
            'heater_i': heater_i_specs,
            'e': e_specs,
            'therm_i': therm_i_specs,
            'therm_v': therm_v_specs
        }
    
    prop = RMEProp(sensitivity = True)
    
    for var in ['heater_i','heater_v','e']:
        # print(var,'on')
        on = metered_to_linmeas(
            f'{var}_on',
            mean = on_means[var],
            std = on_stds[var],
            specs = spec_sheets[var],
            umech_prefix = 'DC Sweep',
            origin = 'DC Sweep'
            )
        # print(var,'off')
        off = metered_to_linmeas(
            f'{var}_off',
            mean = off_means[var],
            std = off_stds[var],
            specs = spec_sheets[var],
            umech_prefix = 'DC Sweep',
            origin = 'DC Sweep'
            )
        measurements[var] = on - off
    if thermometer_monitor:
        for var in ['therm_i','therm_v']:
            measurements[var] = metered_to_linmeas(
                var,
                mean = on_means[var],
                std = on_stds[var],
                specs = spec_sheets[var],
                umech_prefix = 'DC Sweep',
                origin = 'DC Sweep'
                )
    
    return measurements, figs
