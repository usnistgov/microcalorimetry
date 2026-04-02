"""Parse, View, and Run functions for a microcalorimeter experiment."""

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

__all__ = ['run', 'parse', 'view', 'generate_settled_runlist', 'runlist_from_loss']


def view(
    metadata: Path,
    signal_config: configs.RFSweepSignalConfig = None,
    time_window: list[float] = None,
    plot_p_est: bool = True,
    plot_sensor_raw: bool = True,
    down_sample_n: int = 1,
    power_units: str = 'W',
    time_units: str = 'hrs',
) -> tuple[plt.Figure]:
    """
    View an ongoing rfsweep experiment.

    Parameters
    ----------
    metadata : Path
        Path to the metadata file of an active experiment
    signal_config : RFSweepSignalConfig, optional
        Path to a signal configuration of the measurement.
        Will attempt to read the signal config from the measurement
        files if not provided, if provided will overide what is in
        the metadata files.
    time_window : list[float], optional
        Time window to look at (in units of time_units),
        as [min, max]. If
        not provided will plot entire time series. by default None
    plot_p_est : bool, optional
        Plots the estimate of each power signal, by default True
    plot_sensor_raw : bool, optional
        Plots the raw data that composes each signal, by default True
    down_sample_n : int, optional
        Down sample time series by n, by default 1
    power_units : str, optional
        Plot units for power, by default 'W'
    time_units : str, optional
        Plot units for time, by default 'hrs'

    Returns
    -------
    figures : tuple[Figure]
        Tuple of output figures.
    """

    def down_sample(ts, n=down_sample_n):
        index = np.arange(0, len(ts[0]), down_sample_n)
        t = ts[0][index]
        y = ts[1][index]
        return t, y

    def zero_hour(ts, t0=None):
        return ts / 3600

    def punit(vals, origin: str = 'W'):
        if origin == power_units:
            return vals
        if origin == 'W':
            if power_units == 'mW':
                return vals * 1000
            elif power_units == 'dBm':
                return 10 * np.log10(vals) + 30
        elif origin == 'dBm':
            if power_units == 'mW':
                return 10 ** (vals / 10)
            elif power_units == 'W':
                return 10 ** (vals / 10) / 1000

    # check that powerlevelling is functioning properly
    # this is just for debugging inline

    # mpl.use('tkagg')
    # plt.close('all')
    # metadata = r"O:\67201\Power\24Calor\rawdata\C24N129\040\cstd_run_0_incomplete\20250115_metadata.csv"
    # it will break the app if run normally


    # if a folder is pointed to, use a filed with _metadata.csv
    # in the name as the metadata file, so folders can be pointed to
    # adjusted_metadata = []
    # for md in [metadata]:
    #     md = Path(md)
    #     if not md.exists():
    #         raise FileExistsError(f'{md} doesnt exist')
    #     elif md.is_file():
    #         adjusted_metadata.append(md)
    #     # assume folders contain a single run
    #     # 1 metadata file
    #     elif md.is_dir():
    #         new_md = [f for f in md.glob('*_metadata.csv')]
    #         if len(new_md) == 1:
    #             adjusted_metadata.append(new_md[0])
    #         else:
    #             raise ValueError(f'{md} must contain exactly 1 *_metadata files to be pointed to by parser.')
    # metadata = adjusted_metadata[0]

   # if power levelling is happening, plot a summary of that

    dr = ExistingRecord(metadata)

    d_full = dr.batch_read()


    try:
        ep = ExptParameters(dr.metadata['config_file'], dr.metadata['settings_file'])

    except FileNotFoundError:
        newdir = dirname(metadata)
        _config_file = join(newdir, basename(dr.metadata['config_file']))
        _run_settings_file = join(newdir, basename(dr.metadata['settings_file']))
        ep = ExptParameters(_config_file, _run_settings_file)

    if time_window is not None:
        raise NotImplementedError(
            "Haven't added time windowing, zoom in to full plot for now."
        )

    cmm = signal_config
    if cmm is None:
        cmm = ep.config['signal_config']

    time_coeffs = {'hrs': 1 / 3600, 'min': 1 / 60, 's': 1}
    tcoeff = time_coeffs[time_units]

    fig_ts = None
    if plot_p_est:
        fig_ts, ax_ts = plt.subplots(1, 1)
        for ai, sensor in enumerate(dict(cmm)):
            try:
                # the e mapping corresponds to the thermopile voltage, it
                # doesnt have a power estimate
                pest = d_full[microrunner.format_pmeter_est_column((sensor))]
                ax_ts.plot(
                    pest.t * tcoeff, punit(pest.values, origin='W'), 'o-', label=sensor
                )
            except KeyError:
                print(
                    f'Missing Estimated Signal power from {sensor}. Possibly from an older version of the runner.'
                )
        ax_ts.set_ylabel(f'Estimated Metered Power {power_units}')
        ax_ts.set_xlabel('Time (' + time_units + ')')
        ax_ts.legend(loc='best')
        fig_ts.suptitle('Estimated Metered Power of Sensors')
        fig_ts.tight_layout()
    # otherwise, plot each sensors raw time series in a seperate window
    sensor_figs = []
    if plot_sensor_raw:
        for ai, sensor in enumerate(dict(cmm)):
            sensor_map = cmm[sensor]
            sensor_cols = {}
            for k, v in dict(sensor_map).items():
                try:
                    sensor_cols.update({k: v['column']})
                except (TypeError, KeyError):
                    pass
            N = len(sensor_cols)
            fig_i, axs_i = plt.subplots(N, 1, sharex=True)
            if N == 1:
                axs_i = [axs_i]
            fig_i.suptitle(f'{sensor} Raw Data')
            for i, sc in enumerate(sensor_cols):
                try:
                    ts = d_full[sensor_cols[sc]]
                    axs_i[i].plot(ts.t * tcoeff, ts.values, 'o-', ds='steps-post')
                    axs_i[i].set_ylabel(sensor_cols[sc])
                except KeyError as e:
                    msg = f'{sensor_cols[sc]} not in data record, are the input_signals for {sensor} correct? maybe one of {list(d_full.keys())}'
                    raise KeyError(msg) from e
            axs_i[-1].set_xlabel('Time (' + time_units + ')')
            sensor_figs.append(fig_i)

    return fig_ts, *sensor_figs


def run_gui(
    output_dir: Folder,
    configs: list[Path],
    settings: list[Path],
    sensor_master_list: Path,
    name: str = 'rfsweep',
    no_confirm: bool = False,
    dry_run: bool = False,
    repeats: int = 1,
):
    """
    RF Sweep GUI runner.

    Parameters
    ----------
    output_dir : Folder
        Directory to output from. The default is the current working directory.
    configs : list[Path]
        Configuration powers to set up instruments and other measurement
        settings.
    settings : list[Path]
        Settings file(s) for frequency points and power levels.
    sensor_master_list : Path
        Master list of sensor information for sanity checking, should conform
        to RFSensorMasterList spec.
    name : str, optional
        Measurement name. The default is 'rfsweep'.
    no_confirm : bool, optional
        Skip confirming instrument ID strings. The default is False.
    dry_run : bool, optional
        Try to load the settings, and do some validation. The default is False.
    repeats : int, optional
        Number of repeats to perform on all settings files. The default is 1.

    Returns
    -------
    None.

    """

    commands = [
        'rfsweep',
        'run',
        f'"{Path(output_dir).resolve()}"',
    ]

    for c in configs:
        commands += ['--configs', f'"{Path(c).resolve()}"']

    for s in settings:
        commands += ['--settings', f'"{Path(s).resolve()}"']

    commands += ['--sensor-master-list', f'"{Path(sensor_master_list).resolve()}"']
    commands += ['--name', name]

    if no_confirm:
        commands.append('--no-confirm')
    if dry_run:
        commands.append('--dry-run')

    commands += ['--repeats', f'{repeats}']

    clitools.ucal_cli(commands)


@click.command(name='run')
@click.argument('output_dir', type=Path)
@click.option('--repeats', type=int, default=1)
@click.option('--configs', '-c', type=Path, required=True, multiple=True)
@click.option('--settings', '-s', type=Path, required=True, multiple=True)
@click.option('--sensor-master-list', '-m', type=Path, required=True)
@click.option('--name', '-n', type=str)
@click.option('--no-confirm', is_flag=True, default=False)
@click.option('--dry-run', is_flag=True, default=False)
def _run_cli(*args, **kwargs):
    """
    Interface for the RF sweep run function.

    """
    # click inputs tuples, i need lists here.
    kwargs['configs'] = list(kwargs['configs'])
    kwargs['settings'] = list(kwargs['settings'])
    print(args)
    print(kwargs)
    return run(*args, **kwargs)


def run(
    output_dir: str,
    repeats: int,
    configs: list[Path],
    settings: list[Path],
    sensor_master_list: configs.RFSensorMasterList,
    name: str = 'rf_sweep',
    no_confirm: bool = False,
    dry_run: bool = False,
):
    """
    Run a microcalorimetry RF Sweep experiment.

    Parameters
    ----------
    output_directory : Path, optional
        Directory to output from. The default is None.
    repeats : int, optional
        Number of repeats to perform on all settings files. The default is None.
    configs : list[Path], optional
        Config files for experiment settings / instruments. The default is None.
    settings : list[Path], optional
        Run settings files, looped over for experiment. The default is None. Each
        index i of a configs file corresponds to a index i of the configs file.
    sensor_master_list : configs.RFSensorMasterList,
        Master list of sensors and resitance/sensitivity values to use for
        sanity checking config files.
    name : str, optional
        Name of measurement. The default is None.
    no_confirm : bool, optional
        If True, skips any confirmations that may be asked when starting
        the run.
    dry_run : bool, optional
        If true, will try to load the configurations without actually running
        anything to do a dry-check - can be used to validate some basic type validation
        of the configuration.
    """
    priority = [0] * len(configs)
    if isinstance(settings, str):
        settings = [settings]

    for i in range(repeats):
        for setting in settings:
            if not dry_run:
                output_dir = new_dir(output_dir, name) + r'//'
                with microrunner.MicrocalorimeterRunner(
                    configs,
                    setting,
                    output_dir,
                    sensor_master_list,
                    priority,
                    no_confirm=no_confirm,
                ) as runner:
                    # opens visa resources for every instrument
                    runner.initialize_instruments()
                    runner.start_monitor_mode()
                    while not runner.done:
                        runner.iterate()
                        # print(time.time() - time_0)
                        runner.output()
            # for dry run, just try to initialize all the config files.
            # and don't do anythin with them.
            else:
                output_dir = new_dir(output_dir, name) + r'//'
                microrunner.MicrocalorimeterRunner(
                    configs,
                    setting,
                    output_dir,
                    sensor_master_list,
                    priority,
                    dry_run=True,
                )


_run_cli = clitools.format_from_npdoc(run)(_run_cli)


@click.command(name='parse')
@click.argument('metadata', type=Path, nargs=-1)
@click.option('--output-file', '-o', type=Path)
@click.option('--show-plots', is_flag=True)
@click.option('--save-plots', type=Path)
@click.option('--plot-ext', type=str)
def _parse_cli(
    *args,
    show_plots: bool = False,
    output_file: Path = None,
    save_plots: Path = None,
    plot_ext: str = '.png',
    **kwargs,
):
    """
    Interface for the command line for parsing DC sweeps.

    Parameters
    ----------
    show_plots : bool, optional
        If true, shows the plots in a gui and freezes the terminal, by default False.
    output_file : Path, optional
        If provided, outputs any saveable objects to an HDF5 file, by default None.
    save_plots : Path, optional
        If provided, saves plots to this directory.  The default is None.
    plot_ext : str, optional
        File extension to save plots as. The default is '.png.'.

    Returns
    -------
    _type_
        _description_
    """
    kwargs['make_plots'] = bool(show_plots or save_plots)
    print('Parse RFSWEEP from CLI:')
    full_output = dict(
        args=[str(a) for a in args],
        show_plots=show_plots,
        output_file=str(output_file),
        save_plots=str(save_plots),
        plot_ext=plot_ext,
        **{k: str(v) for k, v in kwargs.items()},
    )
    print(json.dumps(full_output, indent=True))
    outputs = clitools.run_and_show_plots(
        parse,
        *args,
        show_plots=show_plots,
        save_plots=save_plots,
        plot_ext=plot_ext,
        **kwargs,
    )
    if output_file:
        clitools.save_saveable_objects(outputs[0], output_file=output_file)
    return outputs


def parse(
    metadata: list[Path],
    analysis_config: configs.RFSweepParserConfig = None,
    verbose: bool = False,
    make_plots: bool = False,
    plot_segments_analysis: list[int] = [0,-1],
    plot_all_segments_analysis: bool = False,
    dataframe_results: Path = None,
    format_matlab: Path = None,
) -> tuple[dict[RMEMeas], list[plt.Figure]]:
    """
    Parse a microccalorimeter run to produce data with uncertainties.

    Parameters
    ----------
    metadata : list[Path]
        Path to metadata file for experient. Can be multiples.
    analysis_config : RFSweepParserConfig, optional
        Path to sensor configuration file, overrides any sensor configuration in the metadata if provided.
    verbose : bool, optional
        If True, prints info about the analysis. Default is True
    make_plots : bool, optional
        Makes plots if True.
    plot_segments_analysis : list[int]
        What segment of each run to plot.
    plot_all_segments_analysis : bool
        If True, when making plots plot every segments
        analysis.
    dataframe_results : Path, optional
        If provided, saves a csv of intermediate calculated values.
    format_matlab : Path, optional
        If provided, saves a matlab version of the output results.

    Returns
    -------
    parsed_rf : dict[RMEMeas]
        Dictionairy of RFSweep datasets in a parsed RF sweep configuration.
    figures : list[plt.Figure]
        Dictionairy of RFSweep datasets in a parsed RF sweep configuration.

    """
    # wrap propagator around functions
    basicprop = RMEProp(
        sensitivity=True,
        montecarlo_sims=0,
    )

    zeta_dcsub = basicprop.propagate(rfpower.zeta_dcsub)
    zeta_general = basicprop.propagate(rfpower.zeta_general)
    dc_sub = basicprop.propagate(rfpower.dc_substituted_power)
    openloope_te_power = basicprop.propagate(rfpower.openloop_thermoelectric_power)
    if analysis_config is None:
        analysis_config = {}
    else:
        analysis_config = configs.load_config(analysis_config)

    # set up parser
    metadata_dict = {}
    figures = []
    if type(metadata) is not list:
        metadata = [metadata]

    # if a folder is pointed to, use a filed with _metadata.csv
    # in the name as the metadata file, so folders can be pointed to
    adjusted_metadata = []
    for md in metadata:
        md = Path(md)
        if not md.exists():
            raise FileExistsError(f'{md} doesnt exist')
        elif md.is_file():
            adjusted_metadata.append(md)
        # assume folders contain a single run
        # 1 metadata file
        elif md.is_dir():
            new_md = [f for f in md.glob('*_metadata.csv')]
            if len(new_md) == 1:
                adjusted_metadata.append(new_md[0])
            else:
                raise ValueError(f'{md} must contain exactly 1 *_metadata files to be pointed to by parser.')

    metadata = adjusted_metadata

    runs = []
    for metadata_path in metadata:
        run_dir = Path(metadata_path).parent
        run_file = Path(metadata_path).name
        run = microparser.NewTypeRun(
            str(run_dir) + '/', run_dir / run_file, run_file, **analysis_config
        )

        run.load()
        run.analyze()
        metadata_dict.update({str(k): v for k, v in run.expt.config.items()})

        # make detailed analysis plots
        if make_plots:
            for signal, analyzer in run.analyzers.items():
                try:
                    analyzer.plot_analysis
                except AttributeError:
                    print(f"{signal} {analyzer} has no plot analysis method. Skipping" )
                    continue
                for si, segment in enumerate(run.segments):
                    called_out = si in plot_segments_analysis
                    is_end = si == len(run.segments)-1 and (-1 in plot_segments_analysis)
                    if called_out or is_end or plot_all_segments_analysis:
                        # plot the analysis for a single step
                        figure = analyzer.plot_analysis(segment)
                        figure.suptitle(f"{signal} signal \n run {Path(metadata_path).parent.name} ; segment {si}")
                        figures.append(figure)

        runs.append(run)

    # use the last signal config
    signal_config = configs.RFSweepSignalConfig(runs[-1].parsed_config['signal_config'])

    # format data output into rmellipse objects
    c = microparser.Campaign(runs, Path.cwd(), 'rfsweep')



    # generate noise plots
    if make_plots:
        # if make_plots:
        data_df = c.output_segments(fmt_for='pandas')
        for signal, sconfig in signal_config.items():
            input_signals = sconfig['input_signals']
            if isinstance(input_signals, str):
                input_signals = [input_signals]
            for ins in input_signals:
                # plot the deviation of the RF on value as a function
                # of the step number
                fig,ax = plt.subplots(2,1)
                try:
                    column = sconfig[ins]['column']
                    fig.suptitle(f'{column} RF On standard deviation')


                    for mode in ['on']:
                        dev = data_df[f'{column}_{mode}_dev']
                        mean = data_df[f'{column}_{mode}']
                        ppm = dev/mean*1e6
                        ax[0].plot(dev, 'o')
                        ax[1].set_xlabel('Step Number')
                        ax[1].plot(ppm, 'o')
                        ax[0].set_ylabel(f'Column RF On std ({sconfig[ins]['units']})')
                        ax[1].set_ylabel('Column RF On std (ppm)')
                    figures.append(fig)
                except Exception as e:
                    plt.close(fig)
                    if verbose:
                        print(f"Encountered error plotting signal {signal}:{ins} std \n {type(e)}: {e}")


                fig,ax = plt.subplots(2, 1, figsize = (8,8))
                try:
                    column = sconfig[ins]['column']
                    fig.suptitle(f'{signal} \n {column} standard deviation')
                    ax[1].set_xlabel('Before (left) and After (right)')
                    ax[0].set_ylabel(f'STD of {ins} ({sconfig[ins]['units']})')
                    ax[1].set_ylabel(f'STD of {ins} (ppm)')
                    for ir, run in enumerate(c.run_list):
                        for iseg, segment in enumerate(run.segments):
                            i_off=  segment.results[f'{column}_off_i']
                            f_off =  segment.results[f'{column}_off_f']
                            i_off_dev =  segment.results[f'{column}_off_i_dev']
                            f_off_dev =  segment.results[f'{column}_off_f_dev']
                            label = f'{run.name} : segment {iseg}'
                            ax[0].plot([0,1], [i_off_dev, f_off_dev],'o--', label = label)
                            ax[1].plot([0,1], [i_off_dev/i_off*1e6, f_off_dev/f_off*1e6],'o--')
                    ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
                    fig.tight_layout()
                    figures.append(fig)
                except Exception as e:
                    plt.close(fig)
                    if verbose:
                        print(f"Encountered error plotting  {signal}:{ins} segment off \n {type(e)}: {e}")

        # Plot if the source think it is in
        # compression

        compression = {}
        freq = np.array([])
        try:
            for run in c.run_list:
                for segment in run.segments:
                    if segment.results['complete']:
                        for step in segment.steps:
                            signals = [k for k in step.raw_data.keys() if fnmatch(k, '*power_signal*') and 'timestamp' not in k and 'calorimeter' not in k]
                            source = step.raw_data['RF_source_power_signal (W)']
                            source = source[2:].astype(float)
                            pow_signals = {s:step.raw_data[s] for s in signals if 'source' not in s}

                            # calculate compression
                            good_freq = False
                            for k,sig in pow_signals.items():
                                sig = sig[2:].astype(float)
                                max_i = np.argmax(sig)
                                sig_dB_change = 10*np.log10(sig[max_i]/sig)

                                #closest to 1dB of change
                                sig_1dB_closest_i = np.argmin(abs(sig_dB_change - 1))
                                sig_1dB_closest = sig_dB_change[sig_1dB_closest_i]

                                # if the smalles changes was < 0.4, skip it
                                # is that a good number? idk
                                if sig_1dB_closest < 0.5:
                                    continue

                                # estimate the compression
                                max_source = source[max_i]
                                source_1dB_closest = source[sig_1dB_closest_i]
                                compress_i = sig_1dB_closest - 10*np.log10(max_source/source_1dB_closest)
                                # print(k.split('_power_signal')[0], sig_1dB_closest)
                                try:
                                    compression[k] = np.append(compression[k], compress_i)
                                except KeyError:
                                    compression[k] = np.array([compress_i])

                                good_freq = True
                            if good_freq:
                                freq = np.append(freq, step.frequency)
                            # print(k, step.frequency, compress_i)
            fig,ax = plt.subplots(1,1)
            for k in compression:
                ax.plot(freq, compression[k],'o', label = f'Measured By: {k.split('_power_signal')[0]} Est.')

            ax.axhline(0.4, color = 'r', ls = '--', lw = 3, label = 'Limit')
            ax.legend(loc = 'best')
            ax.set_xlabel('Frequency (GHz)')
            ax.set_ylabel('Compression (dB)')
            ax.set_title('Source Compression Check')
            # plt.show()
            figures.append(fig)
        except Exception as e:
            print(f"Warning: Couldn't estimate compression for : {e}")

        ...

    # output a the dataframe results
    if dataframe_results:
        df = c.output_dataframe()
        df.to_csv(dataframe_results)

    # format fata for rmellipse calculataions
    data = c.output_segments(fmt_for='rmellipse', include_specs=True)


    ep = run.expt.config

    # variablize the column names to make it a bit easier
    assert signal_config['calorimeter_power']['type'] == 'thermoelectric'
    e_col = signal_config['calorimeter_power']['e']['column']

    # do a little bit of post processing to calculate power
    # This calculates the inferred power flowing through the thermopile
    outputs = {}

    cal_coeffs = configs.ThermoelectricFitCoefficients(
        signal_config['calorimeter_power']['coeffs']
    ).load()

    try:
        p_of_e_calorimeter = cal_coeffs.attrs['p_of_e']

        E_on = openloope_te_power(
            cal_coeffs, data.sel(col=e_col + '_on'), p_of_e=p_of_e_calorimeter
        )

        E_off = openloope_te_power(
            cal_coeffs, data.sel(col=e_col + '_off'), p_of_e=p_of_e_calorimeter
        )

        outputs.update({'E_on': E_on, 'E_off': E_off})

    except AttributeError:
        print("Warning: Can't compute power inferred by calorimeter, skipping. Likely a full data model of the sensitivity wasn't supplied.")



    outputs.update(
        {'e_on': data.sel(col=e_col + '_on'), 'e_off': data.sel(col=e_col + '_off')}
    )

    # this part of the code is trying to turn the voltage/current
    # measurements into power measurements of the sensor inside
    # calorimeter and/or on the sidearm.

    # this should be a class insttead of a bunch of if statements

    # if there is a voltage column and no thermoelectric column
    # it's a dc substitution sensor inside the calorimeter (port 2)
    # so use that.

    if signal_config['DUT_power']['type'] == 'bolometer':
        # possible no current was provided if measured on a type 4
        s_v_col = signal_config['DUT_power']['vdc']['column']
        if 'idc' in signal_config['DUT_power']:
            s_i_col = signal_config['DUT_power']['idc']['column']
            I_on = (data.sel(col=s_i_col + '_on'),)
            I_off_slow = (data.sel(col=s_i_col + '_off_slow'),)
            I_off_fast = (data.sel(col=s_i_col + '_off_fast'),)
            p2dc_on = data.sel(col=s_v_col + '_on') * I_on
            p2dc_off_slow = data.sel(col=s_v_col + '_off_slow') * I_off_slow
            p2dc_off_fast = data.sel(col=s_v_col + '_off_fast') * I_off_fast
        else:
            s_resistance = signal_config['DUT_power']['resistance']
            I_on = None
            I_off_slow = None
            I_off_fast = None
            p2dc_on = data.sel(col=s_v_col + '_on') ** 2 / s_resistance
            p2dc_off_slow = data.sel(col=s_v_col + '_off_slow') ** 2 / s_resistance
            p2dc_off_fast = data.sel(col=s_v_col + '_off_fast') ** 2 / s_resistance

        p2_fast = dc_sub(
            data.sel(col=s_v_col + '_on'),
            data.sel(col=s_v_col + '_off_fast'),
            I_on=I_on,
            I_off=I_off_fast,
            R=s_resistance,
        )

        p2_slow = dc_sub(
            data.sel(col=s_v_col + '_on'),
            data.sel(col=s_v_col + '_off_slow'),
            I_on=I_on,
            I_off=I_off_slow,
            R=s_resistance,
        )

        zeta = zeta_dcsub(
            data.sel(col=e_col + '_on'),
            data.sel(col=e_col + '_off'),
            data.sel(col=s_v_col + '_on'),
            data.sel(col=s_v_col + '_off_fast'),
            data.sel(col=s_v_col + '_off_slow'),
            I_on,
            I_off_fast,
            I_off_slow,
        )

        outputs.update(
            {
                'p2_fast': p2_fast,
                'p2_slow': p2_slow,
                'p2dc_on': p2dc_on,
                'p2dc_off_slow': p2dc_off_slow,
                'p2dc_off_fast': p2dc_off_fast,
                'zeta': zeta,
            }
        )

    # this is a thermoelectric power sensor, so lets do that
    elif signal_config['DUT_power']['type'] == 'thermoelectric':
        # get sensor coefficients
        # and the calorimeter coefficients
        s_coeffs = configs.ThermoelectricFitCoefficients(
            signal_config['DUT_power']['coeffs']
        ).load()
        s_e_col = signal_config['DUT_power']['e']['column']

        # check if the slope of the sensor equals the slope of the
        # coefficients, if not then the thermoelectric sensor's RF
        # side has a negative polarity to the srf side and the voltage
        # measured needs to be multiplied by -1.

        s_e_const = 1.0
        measured_slope_sign = np.sign(data.nom.sel(col=s_e_col + '_on')[0])
        coeff_sign = np.sign(s_coeffs.nom.sel(deg=1))
        if measured_slope_sign != coeff_sign:
            s_e_const = -1.0

        p2_on = openloope_te_power(
            s_coeffs,
            s_e_const * data.sel(col=s_e_col + '_on'),
            p_of_e=s_coeffs.attrs['p_of_e'],
        )

        p2_off = openloope_te_power(
            s_coeffs,
            data.sel(col=s_e_col + '_off'),
            p_of_e=cal_coeffs.attrs['p_of_e'],
        )

        p2_slow = p2_on - p2_off

        zeta = zeta_general(
            data.sel(col=e_col + '_on'),
            data.sel(col=e_col + '_off'),
            cal_coeffs,
            cal_coeffs.attrs['p_of_e'],
            p2_slow,
        )

        outputs.update(
            {
                'e_p2_off': data.sel(col=s_e_col + '_off'),
                'e_p2_on': data.sel(col=s_e_col + '_on'),
                'p2_fast': p2_slow,
                'p2_slow': p2_slow,
                'zeta': zeta,
            }
        )
    # dummy sensors don't do anything
    elif signal_config['DUT_power']['type'] == 'special':
        pass
    else:
        msg = f'{signal_config["DUT_power"]["type"]} not recognized'
        raise ValueError(msg)

    # now calculate the relevant powers for the sidearm
    # it may n ot be peresent, check for it in the signal config
    # first.
    try:
        signal_config['monitor_power']
        calculate_monitor = True
    except KeyError:
        calculate_monitor = False
    if calculate_monitor and signal_config['monitor_power']['type'] == 'bolometer':
        s3_v_col = signal_config['monitor_power']['columns']['v_col']
        if 'idc' in signal_config['monitor_power']:
            s3_i_col = signal_config['monitor_power']['columns']['i_col']
            I3_on = (data.sel(col=s3_i_col + '_on'),)
            I3_off_slow = (data.sel(col=s3_i_col + '_off_slow'),)
            I3_off_fast = (data.sel(col=s3_i_col + '_off_fast'),)
            p3_fast = dc_sub(
                data.sel(col=s3_v_col + '_on'),
                data.sel(col=s3_v_col + '_off_fast'),
                I_on=I3_on,
                I_off=I3_off_fast,
            )
            p3_slow = dc_sub(
                data.sel(col=s3_v_col + '_on'),
                data.sel(col=s3_v_col + '_off_slow'),
                I_on=I3_on,
                I_off=I3_off_slow,
            )
        else:
            s3_resistance = signal_config['monitor_power']['resistance']
            p3_fast = dc_sub(
                data.sel(col=s3_v_col + '_on'),
                data.sel(col=s3_v_col + '_off_fast'),
            )
            p3_slow = dc_sub(
                data.sel(col=s3_v_col + '_on'),
                data.sel(col=s3_v_col + '_off_slow'),
                R=s3_resistance,
            )

        outputs.update(
            {
                'p3_fast': p3_fast,
                'p3_slow': p3_slow,
            }
        )

    # no distinguishing between a fast and slow analysis
    # for a commercial power meter
    elif calculate_monitor and signal_config['monitor_power']['type'] == 'commercial':
        s3_p_col = signal_config['monitor_power']['power']['column']
        p3_slow = data.sel(col=s3_p_col + '_on')
        p3_fast = data.sel(col=s3_p_col + '_on')
        outputs.update(
            {
                'p3_fast': p3_fast,
                'p3_slow': p3_slow,
            }
        )
        outputs.update()

    elif calculate_monitor and signal_config['monitor_power']['type'] == 'special':
        pass

    elif calculate_monitor:
        msg = f'{signal_config["monitor_power"]["type"]} not recognized'
        raise ValueError(msg)

    # plot parsed components
    if make_plots:
        plot_outputs = {k:v for k,v in outputs.items() if k in ['zeta']}
        for k, v in plot_outputs.items():
            fig, ax = plt.subplots(1, 1)
            unc = v.stdunc().cov
            ax.errorbar(
                v.nom.frequency,
                v.nom,
                yerr = unc,
                capsize = 3,
                label = 'k=1',
                ls = '')
            ax.set_ylabel(k.replace('zeta','Uncorrected Eta'))
            ax.set_xlabel('Frequency (GHz)')

            sidearm_name = None
            try:
                sidearm_name = ep['measurement_description']['sidearm_name']
            except KeyError:
                pass

            try:
                fig.suptitle(
                    f'Internal mount: {ep["measurement_description"]["mount_name"]}, external mount: {sidearm_name} \n {k}'
                )

            except KeyError:
                fig.suptitle(f'Parsed RF Sweep:  {k}')
            fig.tight_layout()
            figures.append(fig)

    return outputs, figures


_parse_cli = clitools.format_from_npdoc(parse)(_parse_cli)


def mean_last_of_point_dBm(signal, points, i: int, n_samples: int):
    nearest = np.searchsorted(signal[0], points[0][i])
    avg = float(
        np.mean(
            signal[1][:nearest][
                abs(np.diff(signal[1][:nearest], append=signal[1][-1])) > 0
            ][-n_samples]
        )
    )
    return 10 * np.log10(avg) + 30


def runlist_from_loss(
    metadata: Path,
    output_dir: Path = Path('.'),
    level_to: str = 'DUT_power',
    DUT_power_max_dBm: float = 10,
    monitor_power_max_dBm: float = 0,
    max_source_dBm: float = 15,
    output_name: str = 'runlist.csv',
    n_samples: int = 1,
    frequencies: np.ndarray[float] = None,
    segment_size: int = 10,
    off_step_length: int = 2,
    safety_backoff_dBm: float = 3
) -> list[plt.Figure]:
    """
    Generate a runlist from the approximate RF Loss of measurement signals.

    Parameters
    ----------
    metadata : Path
        Path to measurement metadata.
    output_dir : Path, optional
        Directory to output runfiles. The default is Path('.').
    level_to : str, optional
        Which signal to level to. The default is 'DUT_power'.
    DUT_power_max_dBm : float, optional
        Maximum DUT power in dBm. The default is 10.
    monitor_power_max_dBm : float, optional
        Maxmium monitor power in dBm. The default is 0.
    max_source_dBm : float, optional
        Maxmimum allowed source value in dBm . The default is None.
    output_name : str, optional
        Name to output the file as The default is 'runlist.csv'.
    n_samples : int, optional
        Number of samples to average over for calculations. The default is 1.
    frequencies : np.ndarray[float], optional
        Frequency list to interpolate the RF loss values too and produce
        the runlist. If None, uses the frequencies in the provided
        measurement. The default is None.
    segment_size : int, optional
        Number of frequencie points per segment. The default is 5.
    off_step_length : int, optional
        How many steps each off period should be. Typically 2, the default
        it 2.
    safety_backoff_dBm : float, optional
        Back off the start value by this amount to avoid over sourcing.
        The levelling feature will converge to the correct value during a
        measurement. The default is 3.0.

    Returns
    -------
    figures : list[plt.Figure]
        List of generated figures.
    """
    dr = ExistingRecord(metadata)

    frequencies = np.sort(frequencies)

    # PARSE THE MEASURMENT
    max_powers = {
        'DUT_power_signal (W)': DUT_power_max_dBm,
        'monitor_power_signal (W)': monitor_power_max_dBm,
        'RF_source_power_signal (W)': max_source_dBm,
    }
    signal_columns = ['DUT_power_signal (W)']

    if 'monitor_power_signal (W)' in dr.columns:
        signal_columns.append('monitor_power_signal (W)')

    print('Present signals ', signal_columns)

    d_full = dr.batch_read(
        ['frequency', 'power_on', 'point_counter', 'RF_source_power_signal (W)']
        + signal_columns
    )

    src = d_full['RF_source_power_signal (W)']
    points_dr = d_full['point_counter']

    def frmt(*args):
        args = [float(a) for a in args]
        return '{:6.3f} | {:6.3f} | {:6.3f} | {:6.3f}'.format(*args)

    # read in settings file
    try:
        settings = pd.read_csv(dr.metadata['settings_file'])
    except FileNotFoundError:
        try_file = Path(metadata).parent / Path(dr.metadata['settings_file']).name
        settings = pd.read_csv(try_file)

    # the data record counts the initial warm up as a point.
    # experiment needs to be finished for this
    if len(settings) + 1 != len(points_dr[1]):
        print(
            'Incomplete measurement passed to metadata. Meas list may be incomplete or have bad source settings on last frequency.'
        )

    # for each completed measurment
    # change the initial source to the final source
    points = (points_dr[0][1:], points_dr[1][1:])
    new_list = {c: [] for c in settings.columns}
    read_frequencies = np.array([], float)
    losses = {s: np.array([], float) for s in signal_columns}

    # build an RF loss table
    for i, (tp, p) in enumerate(zip(points[0], points[1])):
        # copy row
        fset = settings.iloc[i]
        this_avg = {}
        this_loss = {}
        # leave the zero rows alone, otherwise update initial source
        if not all([fs == 0 for fs in fset]):
            # go up a point to find end of source
            if i < len(points[0]):
                # go up a point to find end of source
                src = mean_last_of_point_dBm(
                    d_full['RF_source_power_signal (W)'], points, i, n_samples
                )
                # this is silly, but finds the start of the next point
                # with nearest, then looks for where the source changes state
                # closest to that point, and averages the previous 7 samples
                # to get the source value right before the fast off
                # happened.
                for signal_name in signal_columns:
                    this_avg[signal_name] = mean_last_of_point_dBm(
                        d_full[signal_name], points, i, n_samples
                    )
                    this_loss[signal_name] = src - this_avg[signal_name]
                    losses[signal_name] = np.append(
                        losses[signal_name], this_loss[signal_name]
                    )
                read_frequencies = np.append(read_frequencies, fset.Frequency_GHz)

    # interpolate losses to the reqeusted frequency grid
    # use the measured frequencies by default
    if frequencies is None:
        frequencies = read_frequencies

    interp_losses = {}



    for signal_name in signal_columns:
        interp_losses[signal_name] = _smallest_neighbour(
            frequencies, read_frequencies, losses[signal_name]
        )

    figs = []

    # make some plots
    fig, ax = plt.subplots()
    figs.append(fig)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('RF Loss (dBm)')
    colors = cycle(['b', 'C1', 'm'])
    for signal_name in signal_columns:
        color = next(colors)
        print(signal_name, interp_losses[signal_name])
        ax.plot(
            read_frequencies,
            losses[signal_name],
            'o',
            color=color,
            label=signal_name.replace('(W)', 'Measured'),
        )
        ax.plot(
            frequencies,
            interp_losses[signal_name],
            '-',
            color=color,
            label=signal_name.replace('(W)', 'Interpolated'),
        )
    ax.legend(loc='best')

    # calculate source power to achieve target power
    for s in signal_columns:
        if level_to in s:
            level_to = s
    initial_source = interp_losses[level_to] + max_powers[level_to] - safety_backoff_dBm

    # calculate expected powers
    expected_powers = {'RF_source_power_signal (W)': initial_source}
    for signal_name in signal_columns:
        expected_powers[signal_name] = initial_source - interp_losses[signal_name]


    # plot the expected power levels
    fig, ax = plt.subplots()
    figs.append(fig)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Expected Initial Power (dBm)')
    for signal_name in signal_columns + ['RF_source_power_signal (W)']:
        color = next(colors)
        ax.plot(
            frequencies,
            expected_powers[signal_name],
            color=color,
            label=signal_name.replace('(W)', 'expected'),
        )
        ax.axhline(
            max_powers[signal_name],
            color=color,
            ls='--',
            label=signal_name.replace('(W)', 'limit'),
        )

    # signal_name = 'RF_source_power_signal (W)'
    # color = next(colors)
    # ax.plot(frequencies, initial_source, color = color, label = signal_name.replace('(W)','expected'))
    # ax.axhline(max_powers[signal_name], color = color,ls = '--', label = signal_name.replace('(W)','limit'))
    ax.legend(loc='best')

    # check no maximums are exceeded
    no_maximums = True
    for signal_name in signal_columns + ['RF_source_power_signal (W)']:
        if (expected_powers[signal_name] > max_powers[signal_name]).any():
            no_maximums = False
            print(
                f'Maximum power exceeded for {signal_name}, no runlist will be generated'
            )

    # if no maximums hit, build a run list
    # interleave the frequency points
    initial_source = _interleave(initial_source)
    frequencies = _interleave(frequencies)


    if no_maximums:
        output_df = {
            'Frequency_GHz': [],
            'Initial_source_power_dBm': [],
            'Target_source_power_dBm': [],
            'Source_power_limit_dBm': [],
        }

        for i, fi in enumerate(frequencies):
            # insert zero rows
            if i % segment_size == 0:
                for n in output_df:
                    output_df[n] += [0]*off_step_length
            output_df['Frequency_GHz'].append(fi)
            output_df['Initial_source_power_dBm'].append(initial_source[i])
            output_df['Target_source_power_dBm'].append(max_powers[level_to])
            output_df['Source_power_limit_dBm'].append(
                max_powers['RF_source_power_signal (W)']
            )

        # append with a zero row
        for n in output_df:
            output_df[n] += [0]*off_step_length
        output_df = pd.DataFrame(output_df)
        output_df.to_csv(Path(output_dir) / output_name, index = False)

    return figs

def _interleave(a):
    out = np.append(a[0:len(a):2], a[1:len(a):2])
    assert len(out) == len(a)
    return out

def _smallest_neighbour(x_interp: np.array, x: np.array, y: np.array):
    """
    Picks the smallest neighbour of y when interpolating x_interp to x.
    """
    # make sure input arrays are sorted
    # and unique
    sort_ind = np.argsort(x)
    x = x[sort_ind]
    y = y[sort_ind]
    x, uniq_index = np.unique(x, return_index = True)
    y = y[uniq_index]
    # pick neighbour with smalles y value
    out = np.zeros(len(x_interp))
    for i, xi in enumerate(x_interp):
        closest_i = np.argmin(abs(x_interp[i] - x))
        f_closest = x[closest_i]
        if f_closest < xi:
            other_neighbour = closest_i+1
        else:
            other_neighbour = closest_i-1
        ytest_1 = y[closest_i]
        try:
            ytest_2 = y[other_neighbour]
        except IndexError:
            ytest_2 = np.inf
        out[i] = min(ytest_1, ytest_2)
    return out

def generate_settled_runlist(
    metadata: Path,
    analysis_config: configs.RFSweepParserConfig = None,
    output_dir: Path = Path('.'),
    output_name: str = None,
    n_samples: int = 7,
):
    """
    Generate a run list with settled source powers.

    Copies the run settings (frequency list, target power, initial source
    power, and source limit) and replaces the initial source power
    with the settled value from the provided run.

    This function does NOT check if the source was actually
    settled, it just assumes it was right before power was turned off.
    Check that your self by inspecting the dashboard.

    Unfinshed runs can be provided, but they may provide bad settings for
    the final points if the experiment never actually settled, and the
    frequency list may be incomplete.

    Parameters
    ----------
    metadata : Path
        Metadata file of output.
    output_dir : Path, optional
        Folder to output new file in. The default is the current directory.
    output_name : str, optional
        What to name new file. The default is the provided metadata name
        + '_settled_runlist.csv'
    n_samples : int, optiona;
        Number of samples to average for final source value.
    """
    dr = ExistingRecord(metadata)

    d_full = dr.batch_read(
        ['rf_power_setting', 'frequency', 'power_on', 'point_counter']
    )

    # get columns I care about
    src = d_full['rf_power_setting']
    points_dr = d_full['point_counter']

    def frmt(*args):
        args = [float(a) for a in args]
        return '{:6.3f} | {:6.3f} | {:6.3f} | {:6.3f}'.format(*args)

    # read in settings file
    try:
        settings = pd.read_csv(dr.metadata['settings_file'])
    except FileNotFoundError:
        try_file = Path(metadata).parent / Path(dr.metadata['settings_file']).name
        settings = pd.read_csv(try_file)

    # the data record counts the initial warm up as a point.
    # experiment needs to be finished for this
    if len(settings) + 1 != len(points_dr[1]):
        print(
            'Incomplete measurement passed to metadata. Meas list may be incomplete or have bad source settings on last frequency.'
        )

    # for each completed measurment
    # change the initial source to the final source
    points = (points_dr[0][1:], points_dr[1][1:])
    new_list = {c: [] for c in settings.columns}

    for i, (tp, p) in enumerate(zip(points[0], points[1])):
        # copy row
        fset = settings.iloc[i]
        for c in settings.columns:
            new_list[c].append(fset[c])
        # leave the zero rows alone, otherwise update initial source
        if not all([fs == 0 for fs in fset]):
            # go up a point to find end of source
            if i < len(points[0]):
                # go up a point to find end of source
                nearest = np.searchsorted(src[0], points[0][i])
                # this is silly, but finds the start of the next point
                # with nearest, then looks for where the source changes state
                # closest to that point, and averages the previous 7 samples
                # to get the source value right before the fast off
                # happened.
                new = float(
                    np.mean(
                        src[1][:nearest][
                            abs(np.diff(src[1][:nearest], append=src[1][-1])) > 0
                        ][-n_samples]
                    )
                )

                new_list['Initial_source_power_dBm'][-1] = new
            pass
        line = frmt(*[fs[-1] for fs in new_list.values()])
        print(line)
        print('-' * len(line))

    # save new dataframe
    df = pd.DataFrame(new_list)
    output_dir = Path(output_dir)

    if output_name is None:
        output_name = os.path.basename(dr.metadata['settings_file']).split('.')[0]
        output_name += '_settled.csv'
    df.to_csv(output_dir / output_name, index=False)
