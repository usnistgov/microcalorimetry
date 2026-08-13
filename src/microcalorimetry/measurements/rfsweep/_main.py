"""Parse, View, and Run functions for a microcalorimeter experiment."""

from rminstr.utilities.path import new_dir
from microcalorimetry.math import rfpower
from rmellipse.uobjects import RMEMeas
from rmellipse.propagators import RMEProp

# from rminstr.data_structures import ExistingRecord, ExptParameters
from pathlib import Path

# from os.path import join, dirname, basename
import microcalorimetry.measurements.rfsweep._parser as microparser
import microcalorimetry.measurements.rfsweep._runner as microrunner
import microcalorimetry.configs as configs
import microcalorimetry._helpers._intf_tools as clitools
from microcalorimetry._tkquick.gui_dtypes import Folder, SaveAsPath
import matplotlib.pyplot as plt

# import os
import numpy as np
import pandas as pd
import json
import click
import time

# from itertools import cycle
# from decimal import Decimal
# from fnmatch import fnmatch
# from microcalorimetry.math.numbers import mean_unique_values
from matplotlib.cm import ScalarMappable

__all__ = [
    'run',
    'parse',
    'reduce_initial_power',
    'reorder_runlist',
    'review_runlist',
]


def run_gui(
    output_dir: Folder,
    configs: list[Path],
    settings: Path,
    sensor_master_list: Path,
    name: str = 'rfsweep',
    no_confirm: bool = False,
    dry_run: bool = False,
):
    """
    RF Sweep GUI runner.

    Parameters
    ----------
    output_dir : Folder
        Directory to output data to.
    configs : list[Path]
        List of configuration files use to setup the measurement.
    settings : list[Path]
        RFSweep run list.
    sensor_master_list : Path
        Master list of sensor information for sanity checking, should conform
        to RFSensorMasterList spec.
    name : str, optional
        Measurement name. The default is 'rfsweep'.
    no_confirm : bool, optional
        Skip confirming instrument ID strings. The default is False.
    dry_run : bool, optional
        Try to load the settings, and do some validation. The default is False.

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

    clitools.ucal_cli(commands)


@click.command(name='run')
@click.argument('output_dir', type=Path)
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
    configs: list[Path],
    settings: list[Path],
    sensor_master_list: configs.RFSensorMasterList,
    name: str = 'rf_sweep',
    no_confirm: bool = False,
    dry_run: bool = False,
    validate: bool = True,
):
    """
    Run a microcalorimetry RF Sweep experiment.

    Parameters
    ----------
    output_directory : Path
        Directory to output from. The default is None.
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
    validate : bool, optional
        If true, attempts to validate configuration files. The default is True.
    """
    priority = [0] * len(configs)
    if isinstance(settings, str) or isinstance(settings, Path):
        settings = [settings]

    if len(settings) > 1:
        raise NotImplementedError('Only on runlist can be supplied at a time.')
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
                validate=validate,
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
                validate=validate,
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
    data_records: list[Folder],
    DUT_power_analysis: dict,
    calorimeter_power_analysis: dict,
    make_plots: bool = True,
    plot_segments_indexes: list[int] = [0, -1],
    plot_all_segments_analysis: bool = False,
    dataframe_results: SaveAsPath = None,
    linear_bolo_eff: bool = False,
    include_time_std: bool = True,
    extra_configs: configs.SerialDictionary = None,
    monitor_power_analysis: dict = None,
    RF_source_power_analysis: dict = None,
) -> tuple[dict[RMEMeas], list[plt.Figure]]:
    """
    Parse an RF sweep.

    Parameters
    ----------
    data_records : list[Folder]
        Path(s) to metadata files (or directories their directories) containing rfsweep runs.
    DUT_power_analysis : dict, optional
        Set the analysis settings for the DUT.
        Format
        ------
        {
        instr_timing_tolerance : float
            Max expected misalignment of instrument's time column to determine
            when RF is turned off. Default is 5.
        stats_window_override : float
            Override the defined stats window to change the window of samples
            averaged over to determine off/on values. The default is None.
        fast_off_analysis : bool
            Whether or not to use fast analysis. Only matters for thermoelectric
            sensors power sensors (which may or may not use the fast off
            analysis). The default is False.
        coeffs : Path
            Provided a path to thermoelectric fit coefficients you want to use.
            Must be provided for thermoelectric power sensors. Overrides the coefficients
            defined in the signal config of the measurement.
        V_off_delay : float
            Relative time from RF being turned off to on to use for the V off
            measurement. Only relevant if fast analysis is being used.
        V_off_function : str
            What V Off fit function to use, only required if doing a fast fit.
            Options Format
            --------------
            {
            lin_plus_exp
                Fit fast off measurements to a line + an exponential.
            linear
                Fit fast off measurements to a line.
            single_sample
                Treat all fast off measurements in window as realizations of the same measurement and average over them.
            }
        V_off_fit_time_window : list[float]
            Relative time window from RF being turned on to fit to. For example,
            [1, 2] would fit between 1 and 2 seconds after RF power is turned
            off.
        }
    calorimeter_power_analysis : dict
        Set the analysis settings for the calorimeter.
        Format
        ------
        {
        instr_timing_tolerance : float
            Max expected misalignment of instrument's time column to determine
            when RF is turned off. Default is 5.
        stats_window_override : float
            Override the defined stats window to change the window of samples
            averaged over to determine off/on values. The default is None.
        coeffs : Path
            Provided a path to thermoelectric fit coefficients you want to use.
            Must be provided for thermoelectric power sensors. Overrides the coefficients
            defined in the signal config of the measurement.
        }
    make_plots : bool, optional
        Makes review plots if True. The default is True.
    plot_segments_indexes : list[int]
        What segment of each run to plot. Each interger represents the index
        of every segment for every run in order of runs. For example,
        index 0 is the first segment of the first run Index -1 is the last
        segment of the last run. The default is [0, -1].
    plot_all_segments_analysis : bool
        If True, plot every single segment's review charts and ignore
        plot_segment_indexes.
    dataframe_results : SaveAsPath, optional
        If provided, saves a csv of intermediate calculated values to the
        specified path.
    linear_bolo_eff : bool, optional
        If true, will utilize the lineraized version of the effective efficiency
        of a bolometer sensor regardless of whether or not a microcalorimeter's
        sensitivity model is present.
    include_time_std : bool, optional
        It True, includes standard deviation of stable samples used to
        calculate the value of a particular column as an uncertainty mechanism.
        The default is True.
    extra_configs : Path, optional
        Path to a .csv file that can be used to overload fields in an existing config file.
    monitor_power_analysis : dict, optional
        Set the analysis settings for the monitor.
        Format
        ------
        {
        instr_timing_tolerance : float
            Max expected misalignment of instrument's time column to determine
            when RF is turned off. Default is 5.
        stats_window_override : float
            Override the defined stats window to change the window of samples
            averaged over to determine off/on values. The default is None.
        fast_off_analysis : bool
            Whether or not to use fast analysis. Only matters for thermoelectric
            sensors power sensors (which may or may not use the fast off
            analysis). The default is False.
        coeffs : Path
            Provided a path to thermoelectric fit coefficients you want to use.
            Must be provided for thermoelectric power sensors. Overrides the coefficients
            defined in the signal config of the measurement.
        V_off_delay : float
            Relative time from RF being turned off to on to use for the V off
            measurement. Only relevant if fast analysis is being used.
        V_off_function : str
            What V Off fit function to use, only required if doing a fast fit.
            Options Format
            --------------
            {
            lin_plus_exp
                Fit fast off measurements to a line + an exponential.
            linear
                Fit fast off measurements to a line.
            single_sample
                Treat all fast off measurements in window as realizations of the same measurement and average over them.
            }
        V_off_fit_time_window : list[float]
            Relative time window from RF being turned on to fit to. For example,
            [1, 2] would fit between 1 and 2 seconds after RF power is turned
            off.
        }
    RF_source_power_analysis : dict, optional
        Set the analysis settings for the RF Source.
        Format
        ------
        {
        instr_timing_tolerance : float
            Max expected misalignment of instrument's time column to determine
            when RF is turned off. Default is 5.
        stats_window_override : float
            Override the defined stats window to change the window of samples
            averaged over to determine off/on values. The default is None.
        }


    Returns
    -------
    parsed_rf : dict[RMEMeas]
        Dictionary of RFSweep datasets in a parsed RF sweep configuration.
    figures : list[plt.Figure]
        Dictionary of RFSweep datasets in a parsed RF sweep configuration.

    """
    verbose = False
    metadata = data_records
    # wrap propagator around functions
    basicprop = RMEProp(
        sensitivity=True,
        montecarlo_sims=0,
    )

    zeta_dcsub = basicprop.propagate(rfpower.zeta_dcsub)
    zeta_general = basicprop.propagate(rfpower.zeta_general)
    dc_sub = basicprop.propagate(rfpower.dc_substituted_power)
    openloope_te_power = basicprop.propagate(rfpower.openloop_thermoelectric_power)

    # if an analysis config fils is provided, load that
    if extra_configs is None:
        analysis_config = {}
    else:
        analysis_config = configs.load_config(analysis_config)

    def overload_analysis_config_from_fvalues(signame, d):
        if d is None:
            return
        # pre fill dictionairys that might be missing
        if 'analysis_config' not in analysis_config:
            analysis_config['analysis_config'] = {}

        if signame not in analysis_config['analysis_config']:
            analysis_config['analysis_config'][signame] = {}

        # over load values that aren't None
        for ckey, value in d.items():
            # there fit coeffs were specified, put those in the
            # right place.
            if ckey == 'coeffs' and value is not None:
                if 'signal_config' not in analysis_config:
                    analysis_config['signal_config'] = {}

                if signame not in analysis_config['signal_config']:
                    analysis_config['signal_config'][signame] = {}

                analysis_config['signal_config'][signame]['coeffs'] = value

            # other rise pass it in
            elif value is not None:
                analysis_config['analysis_config'][signame][ckey] = value

    # use any dictionaries passed in directly to modify the
    # configuration file at run time. This provides the GUI an
    # easier way to modify these config values.
    overload_analysis_config_from_fvalues('DUT_power', DUT_power_analysis)
    overload_analysis_config_from_fvalues(
        'calorimeter_power', calorimeter_power_analysis
    )
    overload_analysis_config_from_fvalues('monitor_power', monitor_power_analysis)
    overload_analysis_config_from_fvalues('RF_source_power', RF_source_power_analysis)

    # import json
    # print(json.dumps(analysis_config, indent = True))

    # DUT, RF_source, and calorimeter_power should always be there
    for always_present in ['DUT_power', 'calorimeter_power', 'RF_source_power']:
        if always_present not in analysis_config['analysis_config']:
            analysis_config['analysis_config'][always_present] = {}

    # print(analysis_config)

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
                raise ValueError(
                    f'{md} must contain exactly 1 *_metadata files to be pointed to by parser.'
                )

    metadata = adjusted_metadata

    runs = []
    # print(metadata)
    ts = time.time()
    for metadata_path in metadata:
        run_dir = Path(metadata_path).parent
        run_file = Path(metadata_path).name
        # fill path to metadata
        run_name = Path(run_file).resolve().as_posix()
        print('run_name', run_name)
        run = microparser.NewTypeRun(
            working_folder=str(run_dir) + '/',
            data_file=run_dir / run_file,
            name=run_file,
            **analysis_config,
        )

        run.load()
        run.analyze()
        metadata_dict.update({str(k): v for k, v in run.expt.config.items()})

        runs.append(run)

    print('\n parse time = ', time.time() - ts)
    print(' Making Noise Plots...')
    print(' ---------------------')
    # make detailed analysis plots
    # there are so many
    ts = time.time()
    if make_plots:
        for run in runs:
            for signal, analyzer in run.analyzers.items():
                try:
                    analyzer.plot_analysis
                except AttributeError:
                    print(f'{signal} {analyzer} has no plot analysis method. Skipping')
                    continue
                for si, segment in enumerate(run.segments):
                    called_out = si in plot_segments_indexes
                    is_end = si == len(run.segments) - 1 and (
                        -1 in plot_segments_indexes
                    )
                    if called_out or is_end or plot_all_segments_analysis:
                        # plot the analysis for a single step
                        # may get multiple plots for step
                        print(
                            f'Making review figure segment: {si} : {signal} : {type(analyzer)}'
                        )
                        new_figures = None
                        try:
                            new_figures = analyzer.plot_analysis(segment)
                        except NotImplementedError:
                            print('  Not implemented')

                        if not isinstance(new_figures, list):
                            new_figures = [new_figures]
                        for figure in new_figures:
                            if figure is not None:
                                figure.suptitle(
                                    f'{signal} signal \n run {Path(metadata_path).parent.name} ; segment {si}'
                                )
                                figures += new_figures

    # use the last signal config
    signal_config = configs.RFSweepSignalConfig(runs[-1].parsed_config['signal_config'])

    # format data output into rmellipse objects
    c = microparser.Campaign(runs, Path.cwd(), 'rfsweep')

    # generate noise plots
    if make_plots:
        # if make_plots:
        data_df = c.output_segments(fmt_for='pandas')
        for signal, sconfig in signal_config.items():
            # don't make noise plots for RF Source power
            # because it doesn't make sense
            if signal == 'RF_source_power':
                continue
            if sconfig['type'] == 'special':
                continue
            input_signals = sconfig['input_signals']
            if isinstance(input_signals, str):
                input_signals = [input_signals]
            for ins in input_signals:
                # plot the deviation of the RF on value as a function
                # of the step number
                fig, ax = plt.subplots(2, 1)
                try:
                    column = sconfig[ins]['column']
                    fig.suptitle(f'{column} RF On standard deviation')
                    cmap = plt.get_cmap('viridis')
                    norm = plt.Normalize(0, len(data_df))
                    for mode in ['on']:
                        freq = data_df['frequency']
                        points = np.arange(len(freq))
                        dev = data_df[f'{column}_{mode}_dev']
                        mean = data_df[f'{column}_{mode}']
                        ppm = dev / mean * 1e6
                        sc = ax[0].scatter(freq, dev, marker='o', c=points)
                        ax[1].set_xlabel('Frequency (GHz)')
                        ax[1].scatter(freq, ppm, marker='o', c=points)
                        ax[0].set_ylabel(f'Column RF On std ({sconfig[ins]["units"]})')
                        ax[1].set_ylabel('Column RF On std (ppm)')
                    sm = ScalarMappable(norm=norm, cmap=cmap)
                    sm.set_array([])
                    cbar = fig.colorbar(sm, ax=ax)
                    cbar.ax.set_title('Point Number')
                    figures.append(fig)
                except Exception as e:
                    plt.close(fig)
                    print(
                        f'Encountered error plotting signal {signal}:{ins} std \n {type(e)}: {e}'
                    )
                plt.show()
                fig, ax = plt.subplots(2, 1)
                try:
                    column = sconfig[ins]['column']
                    fig.suptitle(f'{signal} \n {column} RF Off standard deviation')
                    ax[1].set_xlabel('Before (left) and After (right)')
                    ax[0].set_ylabel(f'STD of {ins} ({sconfig[ins]["units"]})')
                    ax[1].set_ylabel(f'STD of {ins} (ppm)')
                    for ir, run in enumerate(c.run_list):
                        for iseg, segment in enumerate(run.segments):
                            i_off = segment.results[f'{column}_off_i']
                            f_off = segment.results[f'{column}_off_f']
                            i_off_dev = segment.results[f'{column}_off_i_dev']
                            f_off_dev = segment.results[f'{column}_off_f_dev']
                            rname = Path(run.working_folder)
                            # label relative to the CWD if possible,
                            # otherwise use the full path
                            try:
                                rname = (
                                    rname.absolute().relative_to(Path.cwd()).as_posix()
                                )
                                rname = './' + rname
                            except ValueError:
                                rname = '...' + rname.as_posix()[-15:]
                            label = f'{rname} : segment {iseg}'
                            ax[0].plot(
                                [0, 1], [i_off_dev, f_off_dev], 'o--', label=label
                            )
                            ax[1].plot(
                                [0, 1],
                                [i_off_dev / i_off * 1e6, f_off_dev / f_off * 1e6],
                                'o--',
                            )
                    ax[0].legend(
                        bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0
                    )
                    fig.tight_layout()
                    figures.append(fig)
                except Exception as e:
                    plt.close(fig)
                    print(
                        f'Encountered error plotting  {signal}:{ins} segment off \n {type(e)}: {e}'
                    )

    print('plot time = ', time.time() - ts)

    # output a the dataframe results
    df = c.output_dataframe()
    if dataframe_results:
        print(' Outputing intermediate csv results')
        print(' ----------------------------------')
        df.to_csv(dataframe_results)

    # format fata for rmellipse calculataions
    print('generating RMEMeas of raw data...')
    print('---------------------------------')
    ts = time.time()
    data = c.output_segments(
        fmt_for='rmellipse', include_specs=True, include_time_std=include_time_std
    )

    ep = run.expt.config
    print('rme time = ', time.time() - ts)

    # variablize the column names to make it a bit easier
    assert signal_config['calorimeter_power']['type'] == 'thermoelectric'
    e_col = signal_config['calorimeter_power']['e']['column']
    source_col = signal_config['RF_source_power']['power']['column']

    # do a little bit of post processing to calculate power
    # This calculates the inferred power flowing through the thermopile
    print('calculating sensor powers...')
    print('----------------------------')
    outputs = {}

    cal_coeffs = configs.ThermoelectricFitCoefficients(
        signal_config['calorimeter_power']['coeffs']
    ).load()

    try:
        p_of_e_calorimeter = cal_coeffs.attrs['p_of_e']

        E = openloope_te_power(
            cal_coeffs,
            data.sel(col=e_col + '_on') - data.sel(col=e_col + '_off_slow'),
            p_of_e=p_of_e_calorimeter,
        )

        outputs.update({'E': E})

    except AttributeError:
        print(
            "Warning: Can't compute power inferred by calorimeter, skipping. Likely a full data model of the sensitivity wasn't supplied."
        )

    outputs.update(
        {
            'e_on': data.sel(col=e_col + '_on'),
            'e_off': data.sel(col=e_col + '_off_slow'),
        }
    )

    outputs.update({'RF_source_on': data.sel(col=source_col + '_on')})

    # this part of the code is trying to turn the voltage/current
    # measurements into power measurements of the sensor inside
    # calorimeter and/or on the sidearm.

    # this should be a class insttead of a bunch of if statements

    # analysis follows the type of power sensor being used.
    if signal_config['DUT_power']['type'] == 'bolometer':
        # possible no current was provided if measured on a type 4

        s_v_col = signal_config['DUT_power']['vdc']['column']

        # If a current column is present, use that to calculate
        # the dc powers
        if 'idc' in signal_config['DUT_power']:
            s_i_col = signal_config['DUT_power']['idc']['column']
            I_on = (data.sel(col=s_i_col + '_on'),)
            I_off_slow = (data.sel(col=s_i_col + '_off_slow'),)
            I_off_fast = (data.sel(col=s_i_col + '_off_fast'),)
            p2dc_on = data.sel(col=s_v_col + '_on') * I_on
            p2dc_off_slow = data.sel(col=s_v_col + '_off_slow') * I_off_slow
            p2dc_off_fast = data.sel(col=s_v_col + '_off_fast') * I_off_fast
        # if no current column was specified, estimate the off DC power using
        # calorimeter sensitivity and
        # use that to calculate the dc power in the ON states.
        else:
            s_resistance = signal_config['DUT_power']['resistance']
            I_on = None
            I_off_slow = None
            I_off_fast = None

            try:
                E_off = openloope_te_power(
                    cal_coeffs,
                    data.sel(col=e_col + '_off_slow'),
                    cal_coeffs.attrs['p_of_e'],
                )
                s_resistance = data.sel(col=s_v_col + '_off_slow') ** 2 / E_off
            except AttributeError as e:
                print(
                    'failed to estimate microcalorimeter power in off state for :\n{e}'
                )
                print(
                    f'Using provided value of the setpoint resitance instead of calcuating setpoint resistance ({s_resistance} Ohms)'
                )

            p2dc_on = data.sel(col=s_v_col + '_on') ** 2 / s_resistance
            p2dc_off_slow = data.sel(col=s_v_col + '_off_slow') ** 2 / s_resistance
            p2dc_off_fast = data.sel(col=s_v_col + '_off_fast') ** 2 / s_resistance
            outputs.update({'s_resistance': s_resistance})

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

        # linearized approximation of effective efficiency
        # for bolometer sensors
        if linear_bolo_eff:
            zeta = zeta_dcsub(
                data.sel(col=e_col + '_on'),
                data.sel(col=e_col + '_off_slow'),
                data.sel(col=s_v_col + '_on'),
                data.sel(col=s_v_col + '_off_fast'),
                data.sel(col=s_v_col + '_off_slow'),
                I_on,
                I_off_fast,
                I_off_slow,
            )
        # non linear corrected effective efficiency
        else:
            try:
                zeta = zeta_general(
                    e_on=data.sel(col=e_col + '_on'),
                    e_off=data.sel(col=e_col + '_off_slow'),
                    e_0=0,
                    cal_k=cal_coeffs,
                    p_of_e=cal_coeffs.attrs['p_of_e'],
                    P2=p2_fast,
                    P_dc_on_slow=p2dc_on,
                    P_dc_off_slow=p2dc_off_fast,
                )
            except AttributeError as e:
                msg = f'''
                Failed to calculate uncorrected eta for "{e}". Failed to access sensitivity
                coefficients of the microcaloriemter. Either provide coeffs to the microcalorimeter
                power signal or use the linear_bolo_eff flag of the parser.'''
                msg = ' '.join([f.strip() for f in msg.splitlines() if f != ''])
                raise ValueError(msg)

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
        dut_signals = signal_config['DUT_power']
        s_e_col = dut_signals['e']['column']

        # thermometer model do a temperature correction
        if 'therm_v' in dut_signals and 'col' in s_coeffs.dims:
            s_e_const = 1.0
            therm_v_col = dut_signals['therm_v']['column']
            therm_i_col = dut_signals['therm_i']['column']
            therm_v = data.sel(col=therm_v_col + '_on')
            therm_i = data.sel(col=therm_i_col + '_on')
            temperature = therm_v / therm_i
            p2dc_on = therm_v * therm_i
            outputs.update({'temperature_p2': temperature})
            outputs.update({'p2dc_on': p2dc_on})
            sensor_p_of_e = None

        # this is a polyomial fit
        # check if the slope of the sensors measured voltage
        # equals the slope of the sensitivity
        # coefficients, if not then the thermoelectric sensor's RF
        # side has a negative polarity to the dc side and the voltage
        # measured needs to be multiplied by -1.\
        else:
            temperature = None
            p2dc_on = 0
            s_e_const = 1.0
            measured_slope_sign = np.sign(data.nom.sel(col=s_e_col + '_on')[0])
            coeff_sign = np.sign(s_coeffs.nom.sel(deg=0))
            sensor_p_of_e = s_coeffs.attrs['p_of_e']
            if measured_slope_sign != coeff_sign:
                s_e_const = -1.0

        p2_slow = openloope_te_power(
            s_coeffs,
            s_e_const
            * (data.sel(col=s_e_col + '_on') - data.sel(col=s_e_col + '_off_slow')),
            p_of_e=sensor_p_of_e,
            temperature=temperature,
        )

        # if a fast off is available, use that
        # using the thermometer in this calculation was causing me problems, why?
        # Edit from Cole:
        # It was because the thermometer is already implicitly accounted for in the
        # dc fit of k_dc (it is subtrated off by the e_0) so it doesn't
        # need to be subtracted off again.
        try:
            p2_fast = openloope_te_power(
                s_coeffs,
                s_e_const
                * (data.sel(col=s_e_col + '_on') - data.sel(col=s_e_col + '_off_fast')),
                p_of_e=sensor_p_of_e,
                temperature=temperature,
            )
            outputs.update({'e_p2_off_fast': data.sel(col=s_e_col + '_off_fast')})
            zeta = zeta_general(
                e_on=data.sel(col=e_col + '_on'),
                e_0=data.sel(col=e_col + '_off_slow'),
                e_off=None,
                cal_k=cal_coeffs,
                p_of_e=cal_coeffs.attrs['p_of_e'],
                P2=p2_fast,
                # P_dc_on_slow = p2dc_on
            )

        except KeyError:
            print('No fast off analysis for DUT_power, equating to slow off')
            p2_fast = p2_slow
            outputs.update({'e_p2_off_fast': data.sel(col=s_e_col + '_off_slow')})
            zeta = zeta_general(
                e_on=data.sel(col=e_col + '_on'),
                e_0=data.sel(col=e_col + '_off_slow'),
                e_off=None,
                cal_k=cal_coeffs,
                p_of_e=cal_coeffs.attrs['p_of_e'],
                P2=p2_slow,
                # P_dc_on_slow = p2dc_on
            )

        outputs.update(
            {
                'e_p2_on': data.sel(col=s_e_col + '_on'),
                'e_p2_off_slow': data.sel(col=s_e_col + '_off_slow'),
                'p2_fast': p2_fast,
                'p2_slow': p2_slow,
                'zeta': zeta,
            }
        )
    # dummy sensors don't do anything, skip em
    elif signal_config['DUT_power']['type'] == 'special':
        pass

    # sensor type, bail out
    else:
        msg = f'{signal_config["DUT_power"]["type"]} not recognized'
        raise ValueError(msg)

    # now calculate the relevant powers for the sidearm
    # it may n ot be present, check for it in the signal config
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

    # plot uncorrected effective efficiency if it  was calculated
    if make_plots:
        plot_outputs = {k: v for k, v in outputs.items() if k in ['zeta']}
        for k, v in plot_outputs.items():
            fig, ax = plt.subplots(1, 1)
            unc = v.stdunc(k=2).cov
            ax.errorbar(
                v.nom.frequency,
                v.nom,
                yerr=unc,
                capsize=3,
                label=r'Nominal $\pm$ Uncertainty (k=2)',
                ls='',
                elinewidth=1.5,
                marker='o',
                mfc='k',
            )
            ax.set_ylabel(k.replace('zeta', r'$\eta^{\left(unc\right)}$'))
            ax.set_xlabel('Frequency (GHz)')
            ax.legend(loc='best')
            sidearm_name = None
            try:
                sidearm_name = ep['measurement_description']['sidearm_name']
            except KeyError:
                pass

            try:
                fig.suptitle(
                    rf'Internal mount: {ep["measurement_description"]["mount_name"]}, external mount: {sidearm_name} \n Uncorrected $\eta$'
                )

            except KeyError:
                fig.suptitle(r'Parsed RF Sweep:  Uncorrected $\eta$')
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


def reduce_initial_power(
    runlist: Path, output_path: SaveAsPath, reduce_by_dB: float, decimals: int = 4
):
    """
    Reduce a runlist's initial power setting.

    Parameters
    ----------
    runlist : Path
        Path to runlist to modify.
    reduce_by_dB : float
        How much to reduce the initial power by. Negative
        values will increase the initial power.
    output_path : SaveAsPath, optional
        If None, uses the same name as input file with
        '_min{num}dB.csv'
    decimals : int, optional
        Number of decimals to use for frequency points. Default is 4.
    """
    runlist = Path(runlist)
    data = pd.read_csv(runlist)
    ind = data.Frequency_GHz > 0
    new_init = data.Initial_source_power_dBm[ind] - reduce_by_dB
    data.loc[ind, 'Initial_source_power_dBm'] = np.round(new_init, decimals)
    if output_path is None:
        output_path = runlist.parent / (runlist.stem + f'_min{reduce_by_dB}dB.csv')
    data.to_csv(output_path, index=False)


def reorder_runlist(
    runlist: Path,
    output_path: SaveAsPath,
    segment_size: int = 10,
    off_step_length: int = 2,
    freq_ordering: str = 'interleave',
    make_plots: bool = True,
):
    """
    Reorder a runlist.

    Can modify the number of and size of segments, and reorder frequencies.

    Parameters
    ----------
    runlist : Path
        Frequency list to reorder.
    output_path : SaveAsPath, optional
        Directory to output runfile. The default is the same path as the input
        file with '_reordered' appended to the name.
    segment_size : int, optional
        Number of steps per segment (maximum). The default is 10.
    off_step_length : int, optional
        How many steps each off period should be. Typically 2, the default
        is 2.
    freq_ordering : str, optional
        Determines how frequencies are sorted before being split into segments.
        Options Format
        --------------
        {
        monotonic_increasing
            Sort frequencies monitonically increasing (e.g. 1,2,3,5,6).
        interleave_increasing
            Sort frequencies into 2 monitonically increasing, interleaved lists. For example: 1,2,3,4,5,6 becomes 1,3,5,2,4,6.
        }
    make_plots : bool, optional
        Generate review plots of runlist. The default is True.

    """

    # get runlist,  remove off points, sort by
    # frequency
    runlist = Path(runlist)
    if output_path is None:
        output_path = runlist.parent / (runlist.stem + '_reordered.csv')
    data = pd.read_csv(runlist)
    ind = data.Frequency_GHz > 0
    new = data[ind].sort_values('Frequency_GHz')

    # seperate out columns
    frequency = new['Frequency_GHz']
    starting_powers = new['Initial_source_power_dBm']
    target_power = new['Target_source_power_dBm']
    limit_powers = new['Source_power_limit_dBm']

    # interleave if asked to
    match freq_ordering:
        case 'monotonic_increasing':
            initial_source = starting_powers.values.copy()
            frequencies = frequency.values.copy()
            targets = target_power.values.copy()
            limits = limit_powers.values.copy()
        case 'interleave_increasing':
            initial_source = _interleave(starting_powers.values)
            frequencies = _interleave(frequency.values)
            targets = _interleave(target_power.values)
            limits = _interleave(limit_powers.values)
        case _:
            raise ValueError('freq_ordering not a recognized value')

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
                output_df[n] += [0] * off_step_length
        output_df['Frequency_GHz'].append(fi)
        output_df['Initial_source_power_dBm'].append(initial_source[i])
        output_df['Target_source_power_dBm'].append(targets[i])
        output_df['Source_power_limit_dBm'].append(limits[i])

    # append with a zero row
    for n in output_df:
        output_df[n] += [0] * off_step_length
    output_df = pd.DataFrame(output_df)
    output_df.to_csv(output_path, index=False)
    figures = []
    if make_plots:
        figures = review_runlist(output_path)
    return figures


def review_runlist(runlist: Path) -> list[plt.Figure]:
    """
    Review a runlist for an rf sweep measurement.

    Parameters
    ----------
    runlist : Path
        Path to runlist.

    Returns
    -------
    plots : list[plt.Figures]
        List of review charts.

    """
    runlist = Path(runlist)
    data = pd.read_csv(runlist)
    # srtd with 0 rows removed
    ind_on = data.Frequency_GHz > 0
    ind_off = data.Frequency_GHz == 0
    srtd = data[ind_on].sort_values('Frequency_GHz')

    fig1, ax = plt.subplots(1, 1)
    ax.plot(srtd.Frequency_GHz, srtd.Initial_source_power_dBm, label='Initial')
    ax.plot(srtd.Frequency_GHz, srtd.Target_source_power_dBm, label='Target')
    ax.plot(srtd.Frequency_GHz, srtd.Source_power_limit_dBm, label='Limit')
    ax.set_xlabel('Frequency GHz')
    ax.set_ylabel('Power (dBm)')
    ax.set_title(f'Runlist Review: \n {str(runlist)}')
    ax.legend(loc='best')

    fig2, ax = plt.subplots(1, 1)
    ax.plot(data.Frequency_GHz[ind_on], 'o', label='Power On')
    ax.plot(data.Frequency_GHz[ind_off], 'o', label='Power Off')
    ax.set_xlabel('Step Number')
    ax.set_ylabel('Frequency (GHz)')
    ax.set_title(f'Runlist Review: \n {str(runlist)}')
    return [fig1, fig2]


def _interleave(a):
    out = np.append(a[0 : len(a) : 2], a[1 : len(a) : 2])
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
    x, uniq_index = np.unique(x, return_index=True)
    y = y[uniq_index]
    # pick neighbour with smalles y value
    out = np.zeros(len(x_interp))
    for i, xi in enumerate(x_interp):
        closest_i = np.argmin(abs(x_interp[i] - x))
        f_closest = x[closest_i]
        if f_closest < xi:
            other_neighbour = closest_i + 1
        else:
            other_neighbour = closest_i - 1
        ytest_1 = y[closest_i]
        try:
            ytest_2 = y[other_neighbour]
        except IndexError:
            ytest_2 = np.inf
        out[i] = min(ytest_1, ytest_2)
    return out
