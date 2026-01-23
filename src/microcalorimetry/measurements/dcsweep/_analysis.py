from rmellipse.uobjects import RMEMeas
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from rminstr.data_structures import ExistingRecord, ExptParameters
import rminstr_specs.K2450 as kspecs
import rminstr_specs.HP34420A as nvmspecs
from datetime import datetime


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


def calculate_step_final_values(
    raw,
    exp_settings,
    sensor_id: str,
    overview_plots: bool = False,
    zero_limit: float = 0.1,
    source_threshhold: float = 0.1,
    avg_window_size_secs: float = None,
    avg_window_shiftback_secs: float = 0,
    fit_imm_step: bool = False,
    smu_volts_col='V_SMU (V)',
    smu_curr_col='I_SMU (A)',
    thermo_col='V_NVM (V)',
    test_settling_statistics: bool = True,
    force_equal_length_timeseries: bool = False,
    save_plot_path: str = None,
):
    """
    Calculate the final values of the steps on the staircase.

    Parameters
    ----------
    raw : dict
        Dictionary of tuples, output of read_experiment.
    exp_settings : parameter_tree
        Settings used in the staircase experiment. Output of read_experiment.
    sensor_id: str,
        Name of sensor to attach to uncertainty origin.
    overview_plots : bool, optional
        If True, makes plots of the experiment. The default is False.
    zero_limit : float, optional
        Treats anything below this as a zero. Somtimes the SMU is noisy when it
        tries to source zero, and that throws off the threshholding.
        The default is 0.1.
    source_threshhold : float, optional
        Value to look for in difference in source to detect steps.
        The default is 0.1.
    avg_window_size_secs : float, optional
        Time window to average samples over. The default is None.
    avg_window_shiftback_secs : float, optional
        Shift the averaging window starttime backwards by this amount.
        The default is 0.
    fit_imm_step : bool, optional
        If true, fits immediately after the step instead of relative to the
        end of a step. The default is False.
    smu_volts_col : TYPE, optional
        Key of SMU Volts data in raw. The default is 'V_SMU (V)'.
    smu_curr_col : TYPE, optional
        Key for SMU Amps. The default is 'I_SMU (A)'.
    thermo_col : TYPE, optional
        Key for Thermoelectric Volts Column in raw. The default is 'V_NVM (V)'.
    force_equal_length_timeseries: bool, optional
        If true, will force timeseries data to be of the same length, assuming
        that they are alligned at the first index. Useful for experiments that
        got shut down half way. The default is False.
    save_plot_path: str,
        If provided, saves overview plots
    "
    Returns
    -------
    RMEMeas
        RMEMeas object containing final voltage values.
    RMEMeas
        RMEMeas object containing final current values.
    RMEMeas
        RMEMeas object containing final thermoelectric values.
    fig
        overview of experiment data

    """
    # column names
    cnames = [smu_volts_col, smu_curr_col, thermo_col]

    # reduce smu and current lengths to equal grid
    clengths = []
    for cn in [smu_volts_col, smu_curr_col]:
        clengths.append(len(raw[cn][0]))
    length_test = clengths[0]
    clengths_equal = [cl == length_test for cl in clengths]
    if not all(clengths_equal) and force_equal_length_timeseries:
        new_length = min(clengths)
        new_raw = {}
        for k in [smu_volts_col, smu_curr_col]:
            new_raw[k] = (raw[k][0][0:new_length], raw[k][1][0:new_length])
        new_raw[thermo_col] = (
            raw[thermo_col][0][0:new_length],
            raw[thermo_col][1][0:new_length],
        )
        raw = new_raw
    elif not all(clengths_equal) and not force_equal_length_timeseries:
        raise Exception(
            'Data has missing lengths. Set force_equal_length_timeseries to avoid this issue.'
        )

    # remove weird points near boundaries
    indexes = []
    for cn in [smu_volts_col, smu_curr_col]:
        indexes.append(np.abs(raw[cn][1]) > 100)
    index = np.logical_not(np.logical_or(*indexes))

    new_raw = {}
    for k in [smu_volts_col, smu_curr_col]:
        new_raw[k] = (raw[k][0][index], raw[k][1][index])
    new_raw[thermo_col] = (raw[thermo_col][0], raw[thermo_col][1])
    raw = new_raw

    smu_serial = exp_settings['serials']['SMU']
    nvm_serial = exp_settings['serials']['NVM']
    specs = {
        smu_volts_col: kspecs.DatasheetMeasureDCV(
            'vsource',
            smu_serial,
            v_range='auto',
            t_ambient=26,
            time_zero=raw[smu_volts_col][0][0],
        ),
        smu_curr_col: kspecs.DatasheetMeasureDCI(
            'isource',
            smu_serial,
            i_range='auto',
            t_ambient=26,
            time_zero=raw[smu_curr_col][0][0],
        ),
        thermo_col: nvmspecs.DatasheetDCV(
            'nvm',
            serial=nvm_serial,
            v_range='auto',
            t_ambient=26,
            time_zero=raw[thermo_col][0][0],
            null=False,
        ),
    }

    # steps
    zero_ind = np.where(abs(raw[smu_volts_col][1]) < zero_limit)[0]
    zeros = raw[smu_volts_col][0][zero_ind]
    stair_ind = np.where(abs(raw[smu_volts_col][1]) > zero_limit)[0]
    steps_ind = np.where(abs(np.diff(raw[smu_volts_col][1])) > source_threshhold)[0]
    steps_ind = np.array([s for s in steps_ind if s not in zero_ind])
    # store steps as time stamps so
    steps = raw[smu_volts_col][0][steps_ind]
    # create uncertainty objects for each variable
    date = datetime.fromtimestamp(raw[smu_volts_col][0][0]).strftime('%Y%m%d')
    finals = {}
    finals_ind = {}
    for n in cnames:
        t = raw[n][0]
        d = raw[n][1]
        mu = []
        std = []
        nminus1 = []
        # get the average of the mean at end of step
        for si, s in enumerate(steps):
            # for fitting the end of steps
            # the first step is always at the end of the initial
            # zero cooldown period, so needs to be at the final value
            if not fit_imm_step or si == 0:
                t1 = s - avg_window_shiftback_secs
                t0 = t1 - avg_window_size_secs - avg_window_shiftback_secs
            else:
                # need to shift the first timestamp by one sample to not overlap with the previous
                # step
                t0 = (
                    raw[smu_volts_col][0][steps_ind[si - 1] + 1]
                    - avg_window_shiftback_secs
                )
                t1 = t0 + avg_window_size_secs - avg_window_shiftback_secs

            ind = np.logical_and(t >= t0, t < t1)
            vals = d[ind]
            mu.append(np.mean(vals))
            std.append(np.std(vals, ddof=1))  # / np.sqrt(len(vals)))
            nminus1.append(len(vals) - 1)
            try:
                finals_ind[n] = np.append(finals_ind[n], np.where(ind)[0])
            except KeyError:
                finals_ind[n] = np.where(ind)[0]

        # Add type A uncertainty to each final value
        covarr = np.expand_dims(np.array(mu), axis=0)
        pnames = ['nominal']
        for i, s in enumerate(std):
            cdi = np.expand_dims(np.array(mu), axis=0).copy()
            cdi[0, i] += s
            pnames.append(n + '_std_' + 'step' + str(i) + '_' + date)
            covarr = np.concatenate((covarr, cdi), axis=0)
        # Datasheet tings+
        bunc = covarr[0, :] + specs[n].all_manufacturer_errors(covarr[0, :])

        pnames.append('Datasheet_' + n + '_' + date)

        covarr = np.concatenate((covarr, np.expand_dims(bunc, 0)), axis=0)

        cov = xr.DataArray(
            covarr,
            dims=('umech_id', 'steps'),
            coords={'umech_id': pnames, 'steps': np.arange(covarr.shape[-1])},
        )

        # dofs
        dofs = xr.DataArray(
            np.ones(len(pnames[1:])), dims=('umech_id'), coords={'umech_id': pnames[1:]}
        )
        dofs.values[...] = np.inf
        dofs.values[:-1] = np.array(nminus1)

        # make a MUF meas object with data

        uobj = RMEMeas(name=n, cov=cov, covdofs=dofs)
        type_b_categories = {
            'Type': 'B',
            'Origin': f'{sensor_id} $k$ DC Traceability',
            'Experiment': 'Voltage Staircase',
        }
        type_a_categories = {
            'Type': 'A',
            'Origin': f'{sensor_id} $k$ Noise',
            'Experiment': 'Voltage Staircase',
        }
        for k, v in type_a_categories.items():
            uobj.assign_categories(
                pnames[1:-1], [k] * len(pnames[1:-1]), [v] * len(pnames[1:-1])
            )
        for k, v in type_b_categories.items():
            uobj.assign_categories([pnames[-1]], [k], [v])

        finals[n] = uobj
    fig = None
    if overview_plots:
        fig, ax = plt.subplots(2, 1, sharex=True)
        fig = fig
        ax = ax
        for s in steps:
            ax[0].axvline((s - raw[smu_volts_col][0][0]) / 3600, color='k')
        ax[0].plot(
            (raw[smu_volts_col][0] - raw[smu_volts_col][0][0]) / 3600,
            raw[smu_volts_col][1] * raw[smu_curr_col][1] * 1e3,
            '.',
        )

        ax[1].plot(
            (raw[thermo_col][0] - raw[thermo_col][0][0]) / 3600,
            raw[thermo_col][1] * 1e9,
            '.',
        )
        ax[1].set_xlabel('Time (Hours)')
        ax[0].set_ylabel('P_SMU (mW)')

        ax[1].set_ylabel(thermo_col.replace('(V)', ('(nV)')))

        ax[0].plot(
            (
                raw[smu_volts_col][0][finals_ind[smu_volts_col]]
                - raw[smu_volts_col][0][0]
            )
            / 3600,
            1e3
            * raw[smu_volts_col][1][finals_ind[smu_volts_col]]
            * raw[smu_curr_col][1][finals_ind[smu_curr_col]],
            'ro',
        )

        ax[1].plot(
            (raw[thermo_col][0][finals_ind[thermo_col]] - raw[thermo_col][0][0]) / 3600,
            raw[thermo_col][1][finals_ind[thermo_col]] * 1e9,
            'ro',
        )
        if save_plot_path:
            fig.savefig(save_plot_path)
    return finals[smu_volts_col], finals[smu_curr_col], finals[thermo_col], fig
