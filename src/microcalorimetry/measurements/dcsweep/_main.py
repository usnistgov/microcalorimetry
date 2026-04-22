"""
This script performs a voltage staircase.

"""

from pathlib import Path
from rmellipse.propagators import RMEProp
from rmellipse.uobjects import RMEMeas
from rminstr.utilities import importer, path  # , timer
from rminstr.data_structures import ExptParameters, ActiveRecord, ExistingRecord
from rminstr.instruments.communications import GPIBInterface
from numpy import ceil
from os.path import join
import matplotlib.pyplot as plt
import shutil
import time
import microcalorimetry.measurements.dcsweep._analysis as staircase_analysis
import click
import json
import inspect
from microcalorimetry._tkquick.dtypes import Folder
from typing import TYPE_CHECKING
from datetime import timedelta
if TYPE_CHECKING:
    pass
import microcalorimetry.configs as configs
import microcalorimetry._helpers._intf_tools as clitools

__all__ = ['run', 'parse']


def time_delta(t, t0, unit: str):
    tdel = t - t0
    if unit == 'hrs':
        return tdel / 3600
    if unit == 'min':
        return tdel / 60
    if unit == 's':
        return tdel



class MeasurementManager:
    def __init__(self, smu, *instr):
        self.smu = smu
        self.instruments = [smu] + [i for i in instr]

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        # clean up data outputs
        print('shutting down')
        self.smu.setup(source='off')
        for i in self.instruments:
            i.close()


@click.command(name='parse')
@click.argument('metadata', type=Path)
@click.option('--output-file', '-o', type=Path)
@click.option('--settings', type=Path)
@click.option('--measlist', type=Path)
@click.option('--zero-limit', type=float)
@click.option('--source-threshhold', type=float)
@click.option('--avg-window-size-secs', type=float)
@click.option('--avg-window-shiftback_secs', type=float)
@click.option('--imm-step', type=float)
@click.option('--show-plots', is_flag=True)
@click.option('--save-plots', type=Path)
@click.option('--plot-ext', type=str)
@click.option('--repeatability-id', type=str)
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
    print('Parse DCSWEEP from CLI:')
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
    metadata: Path,
    settings: Path = None,
    measlist: Path = None,
    zero_limit: float = 0,
    source_threshhold: float = 0.01,
    avg_window_size_secs: float = 3000,
    avg_window_shiftback_secs: float = 0,
    imm_step: bool = False,
    make_plots: bool = True,
    repeatability_id: str = 'dc_sweep',
) -> tuple[configs.ParsedDCSweep, list[plt.Figure]]:
    r"""
    Load and analyze a sensitivity run.

    Takes in a sensitivty experiment and generate sensitivity coefficients
    and other relevant data.

    Parameters
    ----------
    metadata : Path
        Path to datarecord metadata
    settings : Path, optional
        Path to experiment settings. Assumed to be a file called settings.csv in metadata directory if not provided.
    measlist : Path, optional
        Path to experiment meas list. Assumed to be a file called measlist.csv in metadata directory if not provided.
    zero_limit : float, optional
        Treats anything below this as a zero for threshholding.
        Somtimes the SMU is noisy when it tries to source zero,
        and that throws off the threshholding.
        The default is 0.1.
    source_threshhold : float, optional
        Value to look for in difference in source to detect steps.
        The default is 0.1.
    avg_window_size_secs : float, optional
        Time window to average samples over in seconds.
        The default is 3000.
    avg_window_shiftback_secs : float, optional
        Shift back the averaging window start time by this amount.
        Averaging start time is end of step - avg_window_size_secs - avg_window_shiftback_secs.
        The default is 0.
    imm_step : bool, optional
        If true, fits immediately after the step instead of relative to the
        end of a step. The default is False.
    make_plots : bool, optional
        If true, generates plots. The default is True.
    repeatability_id : str, optional
        If provided, used as id for uncertainty origin in calculation.
        The default is 'dc_sweep'.

    Returns
    -------
    parsed_dc : configs.ParsedDCSweep
        ParsedCalibration object and any figures generated
    figures : list[plt.Figure]
        Generated figures.

    """

    # do the analysis and save things
    metadata = Path(metadata)
    if metadata.is_dir():
        metadata = [p for p in metadata.glob('*metadata*')][0]
    meta_dir = metadata.parents[0]
    if settings is None:
        settings = meta_dir / 'settings.csv'
    if measlist is None:
        measlist = meta_dir / 'measlist.csv'

    raw, ep = staircase_analysis.read_experiment(
        metadata_path=str(Path(metadata)),
        settings=str(Path(settings)),
        meas_list=str(Path(measlist)),
    )
    try:
        nvm_names_temp = ep['nvm_names']
        if isinstance(nvm_names_temp, str):
            nvm_names_temp = [nvm_names_temp]
        nvm_names = [f'V_{n} (V)' for n in nvm_names_temp]
    except KeyError:
        nvm_names = ['V_NVM (V)']

    # attach metadata to group
    e_list = []
    fig_list = []
    # v, and i never chaneg here, i should make the final values
    # function loop ove reverything, I am being lazy here
    parsed = {'v': {}, 'i': []}
    for nvm_name in nvm_names:
        v, i, e, fig_overview = staircase_analysis.calculate_step_final_values(
            raw,
            ep,
            repeatability_id,
            overview_plots=make_plots,
            zero_limit=zero_limit,
            source_threshhold=source_threshhold,
            avg_window_size_secs=avg_window_size_secs,
            avg_window_shiftback_secs=avg_window_shiftback_secs,
            fit_imm_step=imm_step,
            thermo_col=nvm_name,
            force_equal_length_timeseries=True,
        )
        e_list.append(e)
        fig_list.append(fig_overview)
        parsed[nvm_name] = e
    parsed['v'] = v
    parsed['i'] = i
    v.name = 'V_SMU (V)'
    i.name = 'I_SMU (A)'
    for e, name in zip(e_list, nvm_names):
        e.name = name
        parsed[name] = e

    prop = RMEProp(sensitivity=True)

    # generate  noise vs power plots
    if make_plots:
        power = parsed['v'] * parsed['i']
        for nvm_name in nvm_names:
            fig, ax = plt.subplots(1, 1)
            ax.set_xlabel('Power (mW)')
            ax.set_ylabel(f'Uncertainty in  {nvm_name.replace("(V)", "(nV)")} (k = 1)')
            ax.plot(power.nom*1000, parsed[nvm_name].stdunc().cov * 1e9, 'ko')
            fig_list.append(fig)

    # attatch metadata
    settings_path = settings
    for output in parsed.values():
        output.attrs['metadata'] = str(metadata)
        output.attrs['settings'] = str(settings_path)
        output.attrs['measlist'] = str(measlist)

    return parsed, fig_list


_parse_cli = clitools.format_from_npdoc(parse)(_parse_cli)


def run_gui(
        settings: Path,
        measlist: Path,
        output_dir: Folder,
        name: str = 'dcsweep',
        dry_run: bool = False
        ):
    """
    dcsweep runner GUI.

    Parameters
    ----------
    settings : Path
        Path to the settings config file.
    measlist : Path
        Path to the measurement list file.
    output_dir : Folder
        Directory to output data to.
    name : str, optional
        Name of measurement. Default is 'dcsweep'.
    dry_run : bool, optional
        Does everything up to but not including run the
        experiment. Can be used to validate settings. The default is False.
    """

    commands = [
        'dcsweep',
        'run',
        f'"{str(Path(settings))}"',
        f'"{str(Path(measlist))}"',
        f'"{str(Path(output_dir))}"',
    ]

    if dry_run:
        commands.append('--dry-run')

    clitools.ucal_cli(commands)


def run(
    settings: str,
    measlist: str,
    output_dir: str = '.',
    name: str = 'dcsweep',
    dry_run: bool = False,
    validate: bool = True,
) -> str:
    """
    DCSweep experiment for microcalorimeter.

    Source a series of voltages while monitoring the thermopile.

    Parameters
    ----------
    settings : str
        Settings files for experiment.
    measlist : str
        Voltage measurment list for experiment.
    output_dir : str
        Directory to output to, creats a new folder inside of to store
        csv files.
    name : str, optional
        Name of measurement. Default is 'dcsweep'.
    dry_run : bool, optional
        If True, attempts to validate the measurement.
    validate : bool, optional
        Validate settings against a schema.

    Returns
    -------
    str.
        path to metadata location of output file.

    """

    
    def update_smu_dr(dr: ActiveRecord, data: dict, timestamp: float, name: str = 'SMU'):
        # update fast data while smu is running slow
        dr.update(
            f'V_{name} (V)', data['Voltage (V)'][0], timestamp
        )
        dr.update(
            f'I_{name} (A)', data['Current (A)'][0], timestamp
        )


    # read run settings
    ep = ExptParameters(settings, measlist)

    # validate the config file
    if validate:
        ep2 = ExptParameters(settings)
        ep2 = configs.DCSweepConfiguration(ep2.config)


    # set up an interface
    interface_number = [i.split(':')[0][-1] for i in ep.config['addresses'].values()][0]
    gpib_intfc = GPIBInterface(f'GPIB{interface_number}::INTFC')

    print(f"Using GPIB{interface_number} interface")



    # setup a data record
    columns = ['V_SMU (V)', 'I_SMU (A)']

    # create a column for each nvm
    nvm_names = ep.config['nvm_names']
    if type(nvm_names) is str:
        nvm_names = [nvm_names]

    monitor_thermometer = ep.config['monitor_thermometer']
    # create a column for each nvm
    thermometer_names = ep.config['thermometer_names']
    if type(thermometer_names) is str:
        thermometer_names = [thermometer_names]
    # for thermometer_name in thermometer_names:
    #     v_column = ep.config['']

    # add extra column for thermometers and such
    columns += [f'V_{name} (V)' for name in nvm_names]
    columns += [f'I_{name} (A)' for name in thermometer_names]
    columns += [f'V_{name} (V)' for name in thermometer_names]



    output_dir = path.new_dir(output_dir, name)

    # copy settings files to output directory
    copied_settings_path = join(output_dir, 'settings.csv')
    copied_measlist_path = join(output_dir, 'measlist.csv')
    shutil.copyfile(settings, copied_settings_path)
    shutil.copyfile(measlist, copied_measlist_path)

    # if its a dry run, bounce out once the configs have been validated
    # and before any instruments have been connected to
    if dry_run:
        return

    # active record will output data if measurement stops for whatever reason
    # and knows to make local backups if something goes wrong
    with ActiveRecord(
        columns, output_dir=output_dir + '//', **ep.config['record_settings']
    ) as dr:
        # import instruments
        smu = importer.import_instrument(ep.config['models']['SMU'], 'SMUSourceSweep')
        smu = smu(ep.config['addresses']['SMU'])
        
        thermometers = []

        nvms = []
        if ep.config['monitor_thermopile']:
            for nvm_name in nvm_names:
                NVM = importer.import_instrument(
                    ep.config['models'][nvm_name], 'Voltmeter'
                )
                nvms.append(NVM(ep.config['addresses'][nvm_name]))
                nvms[-1].initial_setup()
                nvms[-1].setup(**ep.config['setup']['NVM'])

        if monitor_thermometer:
            for thermometer_name in thermometer_names:
                _thermometer = importer.import_instrument(
                    ep.config['models'][thermometer_name], 'SMUSourceSweep'
                )
                thermometers.append(_thermometer(ep.config['addresses'][thermometer_name]))
                thermometers[-1].initial_setup(**ep.config['initial_setup'][thermometer_name])
                thermometers[-1].setup(**ep.config['setup'][thermometer_name])
                # set to 2 values so that it is the same as when the SMU
                # sweeps
                print(ep.config['thermometer_source'])
                thermometers[-1].setup(source_trigger_levels = [ep.config['thermometer_source']])
        
        # Here is the measurement loop
        # measurement manager shuts things down if measurmeent
        # stops for whatever reason
        t0 = time.time()
        with MeasurementManager(smu, *nvms, *thermometers) as mm:
            # i'm initializing the SMU inside the context manager
            # so if something goes wrong the context manager can shut it off
            smu.initial_setup(**ep.config['initial_setup']['SMU'])
            smu.setup(**ep.config['setup']['SMU'])

            # initialize measurement loop
            ep.advance()
            source_last = ep.config['V_SOURCE_SETTING (V)']
            samples= {}
            # start measurement
            while not ep.complete():
                source_new = ep.config['V_SOURCE_SETTING (V)']

                # start fast measurements
                smu.setup(
                    source_trigger_levels=[source_new],
                    level = source_new
                    )

                step_start = time.time()

                # sample until the step is done
                while time.time() - step_start < ep.config['step_duration']:
                    # arm everything
                    smu.arm()
                    trigger_instruments = [smu]

                    # arm and add any thermometers or nvms
                    # to the trigger list
                    for thermometer in thermometers:
                        thermometer.arm()
                        trigger_instruments.append(thermometer)

                    for nvm in nvms:
                        nvm.arm()
                        trigger_instruments.append(nvm)

                    # trigger everything
                    t_trigger = gpib_intfc.group_trigger(*trigger_instruments)
          
                    # collect the samples and add to record
                    for nvm, nvm_name in zip(nvms,nvm_names):
                        nvm.wait_until_data_available(timeout = 10)
                        data = nvm.fetch_data()
                        dr.update(
                            f'V_{nvm_name} (V)',
                            data['Voltage (V)'][0],
                            t_trigger
                        )

                    smu.wait_until_data_available(timeout = 10)
                    heater_data = smu.fetch_data()
                    update_smu_dr(dr, heater_data,timestamp= t_trigger, name = 'SMU')

                    for thermometer_name, thermometer in zip(thermometer_names, thermometers):
                        thermometer.wait_until_data_available(timeout = 10)
                        update_smu_dr(
                            dr, 
                            thermometer.fetch_data(),
                            timestamp= t_trigger, 
                            name = thermometer_name
                        )


                    # print record state results:
                    time_left_in_step = max(ep.config['step_duration'] - (time.time() - step_start),0)
                    print('')
                    print(f'Status')
                    print( '======')
                    print(' ','time left in step: ', timedelta(seconds=time_left_in_step))
                    print(' ',f'V_SMU (V): ', dr[f'V_SMU (V)'])
                    print(' ',f'I_SMU (A): ', dr[f'I_SMU (A)'])
                    print(' ',f'P SMU (mW): ', dr[f'V_SMU (V)']*dr[f'I_SMU (A)']*1000)
                    print(' ',f'R SMU (kOhms): ', dr[f'V_SMU (V)']/dr[f'I_SMU (A)']/1000)
                    for thermometer_name in thermometer_names:
                        print('----------------')
                        print(' ',f'R {thermometer_name} (kOhms): ', dr[f'V_{thermometer_name} (V)']/dr[f'I_{thermometer_name} (A)']/1000)
                    for nvm_name in nvm_names:
                        print('----------------')
                        print(' ',f'V_{nvm_name} (V): ', dr[f'V_{nvm_name} (V)'])

                # update fast data while smu is running slow
                dr.batch_update()


                # iterate
                source_last = source_new
                ep.advance()

    return join(output_dir, dr.session_str + '_metadata.csv')


@click.command(name='run')
@click.argument('settings', type=Path)
@click.argument('measlist', type=Path)
@click.argument('output_dir', type=Path)
@click.option('--name',type = str, default = 'dcsweep')
@click.option('--dry-run', is_flag=True, default=False)
def _run_cli(*args, **kwargs):
    print(args)
    return run(*args, **kwargs)


_run_cli = clitools.format_from_npdoc(run)(_run_cli)


def plot_experiment(metadata: str):
    data = ExistingRecord(metadata).batch_read()

    fig, ax = plt.subplots(1, 1)
    ax.plot(data['V_SMU (V)'][0], data['V_SMU (V)'][1], 'k-o')
    ax.set_xlabel('Timestamp (s)')
    ax.set_ylabel('SMU Voltage (V)')
    fig.tight_layout()

    fig, ax = plt.subplots(1, 1)
    ax.plot(data['V_SMU (V)'][0], data['V_SMU (V)'][1] / data['I_SMU (A)'][1], 'k-o')
    ax.set_xlabel('Timestamp (s)')
    ax.set_ylabel(r'SMU Resistance ($\Omega$)')
    fig.tight_layout()

    fig, ax = plt.subplots(1, 1)
    ax.plot(
        data['V_SMU (V)'][0], 1e3 * data['V_SMU (V)'][1] * data['I_SMU (A)'][1], 'k-o'
    )
    ax.set_xlabel('Timestamp (s)')
    ax.set_ylabel(r'SMU Power ($mW$)')
    fig.tight_layout()

    fig, ax = plt.subplots(1, 1)
    ax.plot(data['V_SMU (V)'][0], data['I_SMU (A)'][1], 'k-o')
    ax.set_xlabel('Timestamp (s)')
    ax.set_ylabel(r'SMU Current (A)')
    fig.tight_layout()

    fig, ax = plt.subplots(1, 1)
    ax.plot(data['V_NVM (V)'][0], data['V_NVM (V)'][1], 'k-o')
    ax.set_xlabel('Timestamp (s)')
    ax.set_ylabel('NVM Voltage (V)')
    fig.tight_layout()
