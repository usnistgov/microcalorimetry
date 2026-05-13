from rmellipse.uobjects import RMEMeas
from rmellipse.propagators import RMEProp
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from rminstr.data_structures import ExistingRecord, ExptParameters, TimeSeries
import rminstr_specs.K2450 as kspecs
import rminstr_specs.HP34420A as nvmspecs
import rminstr_specs
from datetime import datetime
from typing import Protocol

from typing import Protocol
from typing import (
    Protocol,
    Sequence,
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


class TimeSeries[T](Protocol):
    shape: Tuple[int]
    dims: Tuple[Literal['time']]
    dtype: T

class DCCalibrationData(Protocol):
    """
    Protocol for a DCCalibratedData xarray.Dataset.
    
    All values should be samples alligned on the time coordinates.
    """
    # variables
    heater_v: TimeSeries[FloatType]
    heater_i: TimeSeries[FloatType]
    heater_p: TimeSeries[FloatType]
    heater_r: TimeSeries[FloatType]
    therm_v: TimeSeries[FloatType]
    therm_i: TimeSeries[FloatType]
    therm_p: TimeSeries[FloatType]
    therm_r: TimeSeries[FloatType]
    e: TimeSeries[FloatType]
    # coordinates
    pwr_setting: Sequence[FloatType]
    time: Sequence[FloatType]
    env_temp: Sequence[FloatType]
    

def read_experiment(metadata_path: str, settings: str, meas_list: str):
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
        umech_prefix: str = ''
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
    
    meas: RMEMeas = RMEMeas.from_nom(name, mean)
    
    if std is not None:
        for i,si in  enumerate(std):
            pert = mean.copy()
            pert[i] += si
            umech_i= f'{umech_prefix}_std_{i}'
            meas.add_umech(
                umech_i, pert,add_uid = True,category = {'Type':'A','Origin':'dc scweep noise'}
                )
    
    if specs is not None:
        unc_array = specs.all_manufacturer_errors(mean.values)
        pert = mean + unc_array
        umech_i= f'{umech_prefix}_spec_{specs.name}'
        meas.add_umech(
            umech_i, pert,add_uid = True,category = {'Type':'A','Origin':'dc sweep traceability'}
            )

    return meas


def legacy_to_parsed_dc(
    metadata_path: str,
    settings: str,
    meas_list: str,
    on_time_window: float = 300,
    off_time_window: float = 300,
    heater_instr_name: str = 'SMU',
    sensor_instr_name: str = 'NVM',
    zero_threshhold: float = 1e-6,
    transition_threshhold_watts: float = 0.1e-4,
    ) -> tuple[RMEMeas,tuple[plt.Figure]]:
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
    data, ep = read_experiment(metadata_path, settings, meas_list)
    
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
            f"Did not find the right number of source transitions (found {n_transitions} instead of {expected_n_transitions}). Try adjusting the transition_threshhold_watts.")
    
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
    sensor_step_numbers = heater_step_numbers.interp(time = e.time, method = 'nearest')\
        .shift(time = -2)


    # timestamps in hours
    te_hrs = (e.time-e.time[0])/3600
    h_hrs = (heater_v.time - heater_v.time[0])/3600

    # make a figure to start plotting analysis
    # this will get added to through the rest
    # of this function
    fig,ax= plt.subplots(2,1, sharex = True)
    for us in np.unique(sensor_step_numbers):
        ind_sensor = sensor_step_numbers == us
        ind_heater = heater_step_numbers == us
        ax[0].plot(e[ind_sensor].time, e[ind_sensor]*1e3)
        ax[1].plot(heater_p[ind_heater].time, heater_p[ind_heater]*1000)
    ax[1].set_xlabel('Time (s)')
    ax[0].set_ylabel(r'$e \left(\mathrm{mV}\right)$')
    ax[1].set_ylabel(r'$P_{heater} \left(\mathrm{mW}\right)$')
    heater_v = xr.DataArray(heater_v)
    
    
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
        ax[0].plot(sample.time, sample*1000,'k')
    
    ax[0].plot(e_ons.time, e_ons*1000, 'ko')
    
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
    
    
    ax[1].plot(heater_v_ons.time, heater_v_ons*heater_i_ons*1000,'k.')
    for v,i in zip(heater_v_ons_samples, heater_i_ons_samples):
        ax[1].plot(v.time, v*i*1000,'k')


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
        ax[0].plot(sample.time, sample*1000,'k.')

    ax[0].plot(e_offs_mn.time, e_offs_mn*1000,'k.')
    
    # interpolate to the on times
    e_offs = e_offs_mn.interp(time = e_ons.time)
    ax[0].plot(e_offs.time, e_offs*1000,'ko--')
    
    
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
        ax[1].plot(v.time, v*i*1000,'k')    
    
    for v,i in zip(heater_v_ons_samples, heater_i_ons_samples):
        ax[1].plot(v.time, v*i*1000,'k')
    
    heater_v_offs = heater_v_offs.interp(time = heater_v_ons.time)
    heater_i_offs = heater_i_offs.interp(time = heater_i_ons.time)


    # make spec sheets, assume worst case which is default
    heater_i_specs = kspecs.DatasheetMeasureDCI('K2450',serial = 'xxx',suppress_warnings = True)
    heater_v_specs = kspecs.DatasheetMeasureDCV('K2450',serial = 'xxx',suppress_warnings = True)
    e_specs = nvmspecs.DatasheetDCV('HP34420A',serial = 'xxx',suppress_warnings = True)
    
    # put in to RMEMEas objects with correlated uncertainties
    e_ons = metered_to_linmeas('e_on',e_ons,std = e_ons_std,specs = e_specs)
    e_offs = metered_to_linmeas('e_off',e_offs,std = e_offs_std,specs = e_specs)
    
    heater_v_ons = metered_to_linmeas(
        'heater_v_on',heater_v_ons,std = heater_v_ons_std,specs = heater_v_specs)
    heater_v_offs = metered_to_linmeas(
        'heater_v_off',heater_v_offs,std = heater_v_offs_std,specs = heater_v_specs)
    
    heater_i_ons = metered_to_linmeas(
        'heater_i_on',heater_i_ons,std = heater_i_ons_std,specs = heater_i_specs)
    heater_i_offs = metered_to_linmeas(
        'heater_i_off',heater_i_offs,std = heater_i_offs_std,specs = heater_i_specs)
    
    
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
    return e, heater_v, heater_i, fig
    