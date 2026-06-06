"""
This script performs a voltage staircase.

"""

from pathlib import Path
from rmellipse.propagators import RMEProp
from rminstr.utilities import importer, path  # , timer
from rminstr.data_structures import ExptParameters, ActiveRecord
from rminstr.instruments.communications import GPIBInterface, Instrument
from os.path import join
import matplotlib.pyplot as plt
import shutil
import json
import time
import microcalorimetry.measurements.dcsweep._analysis as staircase_analysis
import click
from microcalorimetry._tkquick.dtypes import Folder
from typing import TYPE_CHECKING, Mapping
from datetime import timedelta

if TYPE_CHECKING:
    pass
import microcalorimetry.configs as configs
import microcalorimetry._helpers._intf_tools as clitools

__all__ = ['run', 'parse_v0', 'parse_v1']


def time_delta(t, t0, unit: str):
    tdel = t - t0
    if unit == 'hrs':
        return tdel / 3600
    if unit == 'min':
        return tdel / 60
    if unit == 's':
        return tdel

class MeasurementManager:
    def __init__(
        self,
        smus: Mapping[str, Instrument],
        thermometers: Mapping[str, Instrument],
        nvms: Mapping[str, Instrument],
        interface: GPIBInterface,
    ):
        self.smus: Mapping[str, Instrument] = smus
        self.thermometers: Mapping[str, Instrument] = thermometers
        self.nvms: Mapping[str, Instrument] = nvms
        self.instruments: Mapping[Instrument] = smus | thermometers | nvms
        self.interface: GPIBInterface = interface

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        # clean up data outputs
        print('shutting down')
        # turn off things sourceing
        for smu in list(self.smus.values()) + list(self.thermometers.values()):
            smu.setup(source='off')
        for instrument in self.instruments.values():
            instrument.close()

    def update_smu_dr(
        self, dr: ActiveRecord, data: dict, timestamp: float, name: str = 'SMU'
    ):
        # update fast data while smu is running slow
        dr.update(f'V_{name} (V)', data['Voltage (V)'][0], timestamp)
        dr.update(f'I_{name} (A)', data['Current (A)'][0], timestamp)

    def arm_all(self):
        for i in self.instruments.values():
            i.arm()

    def measure(
        self,
        dr: ActiveRecord,
        source_value: float,
        first_sample_delay: float,
        step_duration: float,
        print_label: str = 'Normal Mesurement',
    ):
        step_count = 0
        step_start = time.time()  # sample until the step is done
        while (time.time() - step_start < step_duration) or (step_count < 1):
            # First sample of step delay
            # by a bit
            if step_count > 0:
                self.arm_all()
                t_trigger = self.interface.group_trigger(*self.instruments.values())

            # smu needs to be triggered first when the source
            # is adjusted so that it has time to level out?
            # idk why but it needs to work like this :(
            else:
                for smu in self.smus.values():
                    smu.setup(
                        source_level=source_value,
                    )
                self.arm_all()
                for smu in self.smus.values():
                    smu.trigger()
                t_adjust = time.time()
                time.sleep(first_sample_delay)
                t_trigger = self.interface.group_trigger(
                    *self.nvms.values(), *self.thermometers.values()
                )

            step_count += 1

            # collect the samples and add to record
            for nvm_name, nvm in self.nvms.items():
                nvm.wait_until_data_available(timeout=10)
                data = nvm.fetch_data()
                dr.update(f'V_{nvm_name} (V)', data['Voltage (V)'][0], t_trigger)

            for smu_name, smu in self.smus.items():
                smu.wait_until_data_available(timeout=10)
                heater_data = smu.fetch_data()
                self.update_smu_dr(dr, heater_data, timestamp=t_trigger, name=smu_name)

            for thermometer_name, thermometer in self.thermometers.items():
                thermometer.wait_until_data_available(timeout=10)
                self.update_smu_dr(
                    dr,
                    thermometer.fetch_data(),
                    timestamp=t_trigger,
                    name=thermometer_name,
                )

            dr.update('SOURCE_SETTING (A)', source_value, t_trigger)
            dr.update('time_since_source_adjust (s)', t_trigger - t_adjust, t_trigger)

            # print record state results:
            time_left_in_step = max(step_duration - (time.time() - step_start), 0)
            print('')
            print(f'Status {print_label}')
            print('====================')
            print(' ', 'time left in step: ', timedelta(seconds=time_left_in_step))
            print(' ', 'time since adjust: ', dr['time_since_source_adjust (s)'])
            print('Heater')
            print('------')
            print(' ', f'SOURCE_SETTING (A): ', dr[f'SOURCE_SETTING (A)'])
            print(' ', f'V_SMU (V): ', dr[f'V_SMU (V)'])
            print(' ', f'I_SMU (A): ', dr[f'I_SMU (A)'])
            print(' ', f'P SMU (mW): ', dr[f'V_SMU (V)'] * dr[f'I_SMU (A)'] * 1000)
            print(' ', f'R SMU (kOhms): ', dr[f'V_SMU (V)'] / dr[f'I_SMU (A)'] / 1000)
            for thermometer_name in self.thermometers:
                print('----------------')
                print(
                    ' ',
                    f'R {thermometer_name} (kOhms): ',
                    dr[f'V_{thermometer_name} (V)']
                    / dr[f'I_{thermometer_name} (A)']
                    / 1000,
                )
            for nvm_name in self.nvms:
                print('----------------')
                print(' ', f'V_{nvm_name} (V): ', dr[f'V_{nvm_name} (V)'])
            # input('pause...:')


def run(
    config_files: list[Path],
    runlist: Path,
    output_dir: str = '.',
    meas_name: str = 'dcsweep',
    dry_run: bool = False,
    validate: bool = True,
) -> str:
    """
    DCSweep experiment for microcalorimeter.

    Source a series of voltages while monitoring the thermopile.

    Parameters
    ----------
    config_files : list[Path]
        Settings files for experiment.
    runlist : Path
        Voltage measurment list for experiment.
    output_dir : str
        Directory to output to, creats a new folder inside of to store
        csv files.
    meas_name : str, optional
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
    if isinstance(config_files, tuple):
        config_files = list(config_files)

    # read run settings
    ep = ExptParameters(list(config_files), runlist)

    # make easy to access iterables and
    # dictionaries to use later in the script,
    # and also so I don't have to rewrite the script logic.
    # This is so I can use the same instrument config files
    # as the rf sweep measuremenw as written with to make it easier to
    # maintain measurement systems.

    # smu means heater, sorry about that
    nvm_names = []
    smu_names = []
    thermometer_names = []
    models = {}
    addresses = {}
    setup = {}
    print(ep.config['instruments'])
    for name, instrument in ep.config['instruments'].items():
        # somtimes there is a field called names that lists
        # the keys, just skip that.
        if name == 'names':
            continue
        print('Defining Instrument')
        print(name, json.dumps(instrument, indent = True))
        models[name] = instrument['model']
        addresses[name] = instrument['GPIB_address']
        setup[name] = instrument['initial_settings']
        match instrument['role']:
            case 'heater':
                smu_names.append(name)
                if name != 'SMU':
                    raise ValueError('Heater needs to be called SMU.')
            case 'thermopile_monitor':
                nvm_names.append(name)
            case 'thermometer_monitor':
                thermometer_names.append(name)
            case _:
                raise ValueError(f'Instrument role {instrument['role']} of {name}:{models[name]} not supported')

    print(json.dumps(setup, indent = True))
    if len(smu_names) > 1:
        raise ValueError("Only 1 heater allowed.")
    elif len(smu_names) == 0:
        raise ValueError("No heater supplied.")

    monitor_thermometer = bool(len(thermometer_names))
    monitor_thermopile = bool(len(nvm_names))

    # validate the config file against the schema
    if validate:
        ep2 = ExptParameters(config_files)
        ep2 = configs.DCSweepConfiguration(ep2.config)

    # set up an interface
    gpib_interface_numbers = [i.split(':')[0][-1] for i in addresses.values() if 'GPIB' in i]
    if not all([gpib_interface_numbers[0] == gi for gi in gpib_interface_numbers]):
        raise ValueError(f'All gpib interface numbers should be the same got {gpib_interface_numbers} ')
    interface_number = gpib_interface_numbers[0]
    print(f'Using GPIB{interface_number} interface')

    # setup a data record
    # base columns I'm assuming are always there are the heater (SMU),
    # source setting, and a time since arouce adjustment
    columns = [
        'V_SMU (V)',
        'I_SMU (A)',
        'SOURCE_SETTING (A)',
        'time_since_source_adjust (s)',
    ]



    # add extra column for thermometers and such
    columns += [f'V_{name} (V)' for name in nvm_names]
    columns += [f'I_{name} (A)' for name in thermometer_names]
    columns += [f'V_{name} (V)' for name in thermometer_names]

    output_dir = path.new_dir(output_dir, meas_name)

    # copy settings files to output directory
    copied_settings_path = join(output_dir, 'settings.csv')
    copied_measlist_path = join(output_dir, 'measlist.csv')
    ep.save_config(copied_settings_path)
    shutil.copyfile(runlist, copied_measlist_path)

    # if its a dry run, bounce out once the configs have been validated
    # and before any instruments have been connected to
    if dry_run:
        return

    gpib_intfc = GPIBInterface(f'GPIB{interface_number}::INTFC')
    # active record will output data if measurement stops for whatever reason
    # and knows to make local backups if something goes wrong
    with ActiveRecord(
        columns, output_dir=output_dir + '//', **ep.config['record_settings']
    ) as dr:
        # import instruments
        smu = importer.import_instrument(models['SMU'], 'SMUSourceSweep')
        smu = smu(addresses['SMU'])

        thermometers = []

        nvms = []
        if monitor_thermopile:
            for nvm_name in nvm_names:
                NVM = importer.import_instrument(
                    models[nvm_name], 'Voltmeter'
                )
                nvms.append(NVM(addresses[nvm_name]))
                nvms[-1].initial_setup(**setup[nvm_name])
                nvms[-1].setup(**setup[nvm_name])
        if monitor_thermometer:
            for thermometer_name in thermometer_names:
                _thermometer = importer.import_instrument(
                    models[thermometer_name], 'SMUSourceSweep'
                )
                thermometers.append(
                    _thermometer(addresses[thermometer_name])
                )
                thermometers[-1].initial_setup(**setup[thermometer_name]
                )
                thermometers[-1].setup(**setup[thermometer_name])

        # Here is the measurement loop
        # measurement manager shuts things down if measurmeent
        # stops for whatever reason
        t0 = time.time()
        with MeasurementManager(
            smus={'SMU': smu},
            thermometers={
                therm_name: therm
                for therm_name, therm in zip(thermometer_names, thermometers)
            },
            nvms={nvm_name: nvm for nvm_name, nvm in zip(nvm_names, nvms)},
            interface=gpib_intfc,
        ) as mm:
            # i'm initializing the SMU inside the context manager
            # so if something goes wrong the context manager can shut it off
            # print('heater settings defined')
            # print(json.dumps(setup['SMU'], indent = True))
            smu.initial_setup(**setup['SMU'])
            smu.setup(**setup['SMU'])
            smu.setup(source='on')
            # print('heater settings actual')
            # print(json.dumps(smu.setup_settings, indent = True))
            # initialize measurement loop
            ep.advance()

            # start measurement
            count = 0
            while not ep.complete():
                if ep.config['off_duration'] > 0:
                    mm.measure(
                        dr,
                        source_value=0.0,
                        first_sample_delay=ep.config['first_sample_delay'],
                        step_duration=ep.config['off_duration'],
                        print_label='Zero Measurement',
                    )
                # measure sample
                mm.measure(
                    dr,
                    source_value=ep.config['SOURCE_SETTING (A)'],
                    first_sample_delay=ep.config['first_sample_delay'],
                    step_duration=ep.config['step_duration'],
                    print_label='Normal Sample',
                )

                dr.batch_update()
                ep.advance()

    return join(output_dir, dr.session_str + '_metadata.csv')



def _get_metadata(path: Path):
    """
    If a folder, get the metadata file. Otherwise assume metadata file.

    Parameters
    ----------
    path : Path
        DESCRIPTION.

    Returns
    -------
    Path
        Path to '*metadata.csv' file.
    """
    metadata = Path(path)
    if metadata.is_dir():
        metadata = [p for p in metadata.glob('*metadata*')][0]
    return metadata


def parse_v0(
    metadata: Path,
    settings: Path = None,
    measlist: Path = None,
    on_time_window: float = 300,
    off_time_window: float = 300,
    heater_instr_name: str = 'SMU',
    sensor_instr_name: str = 'NVM',
    thermometer_instr_name: str = 'Thermometer',
    zero_threshhold: float = 1e-6,
    transition_threshhold_watts: float = 0.1e-4,
    make_plots: bool = True,
) -> tuple[configs.ParsedDCSweep, list[plt.Figure]]:
    r"""
    Load and analyze the initial draft of a DC sweep run.

    Takes in a sensitivity experiment and generate sensitivity coefficients
    and other relevant data.

    Parameters
    ----------
    metadata : list[Path]
        Path to datarecord metadata
    settings : Path, optional
        Path to experiment settings. Assumed to be a file called settings.csv in metadata directory if not provided.
    measlist : Path, optional
        Path to experiment meas list. Assumed to be a file called measlist.csv in metadata directory if not provided.
    on_time_window : float, optional
        Time window for selecting on samples. The default is 300.
    off_time_window : float, optional
        Time window for selecting off samples. The default is 300.
    heater_instr_name : str, optional
        Name of heater instrument. The default is 'SMU'.
    sensor_instr_name : str, optional
        Name of sensor instrument. The default is 'NVM'.
    thermometer_instr_name : str, optional
        Name of the thermometer instrument. The default is 'Thermometer'
    zero_threshhold : float, optional
        Threshhold (in Watts) of samples below this where the heater
        is believed to be turned off. The default is 1e-6.
    transition_threshhold_watts : float, optional
        Thresh hold (in Watts) where changes in the applied power to the heater
        constitutes a new step in the sweep. The default is 0.1e-4.
    make_plots : bool, optional
        If true,make plots

    Returns
    -------
    parsed_dc : configs.ParsedDCSweep
        ParsedCalibration object and any figures generated
    sensitivity : configs.ThermolectricFitCoefficients
        Thermoelectric fit coefficients
    figures : list[plt.Figure]
        Generated figures.

    """
    # look at first meatadata file and check what's in it

    # do the analysis and save things
    metadata = Path(metadata)
    if metadata.is_dir():
        metadata = [p for p in metadata.glob('*metadata*')][0]
    meta_dir = metadata.parents[0]
    if settings is None:
        settings = meta_dir / 'settings.csv'
    if measlist is None:
        measlist = meta_dir / 'measlist.csv'

    # original draft of the measurement

    e, heater_v,heater_i, fig = staircase_analysis.parse_v0(
        metadata_path=str(Path(metadata)),
        settings=str(Path(settings)),
        meas_list=str(Path(measlist)),
        on_time_window=on_time_window,
        off_time_window=off_time_window,
        heater_instr_name=heater_instr_name,
        sensor_instr_name=sensor_instr_name,
        zero_threshhold=zero_threshhold,
        transition_threshhold_watts=transition_threshhold_watts,
    )

    parsed = {}
    parsed['e'] = e
    parsed['heater_v'] = heater_v
    parsed['heater_i'] = heater_i

    prop = RMEProp(sensitivity=True)

    figs = []
    if make_plots:
        figs.append(fig)

    # attatch metadata
    settings_path = settings
    for output in parsed.values():
        output.attrs['metadata'] = str(metadata)
        output.attrs['settings'] = str(settings_path)
        output.attrs['measlist'] = str(measlist)

    return parsed, figs


def parse_v1(
    metadata: list[Path],
    thermopile_monitor: str,
    on_min_wait_time: float = 0.0,
    on_max_wait_time: float = 1e99,
    off_min_wait_time: float = 0.0,
    off_max_wait_time: float = 1e99,
    min_pwr_setting: float = 0.0,
    throw_away_min_time: float = 0.,
    heater: str = 'SMU',
    thermometer_monitor: str = None,
    make_plots: bool = True
) -> tuple[configs.ParsedDCSweep, list[plt.Figure]]:
    """
    Parse version 1 of a DC sweep calibration measurement.

    Parameters
    ----------
    metadata : list[Path]
        List of paths to metadata files.
    theropile_monitor : str
        Name of thermopile monitor.
    on_min_wait_time : float
        Minimum time to wait before an on measurment should be included as part of the
        fit.
    on_max_wait_time : float
        Maximum time to wait before an on measurement should be included as
        part of the fit.
    off_min_wait_time : float
        Minimum time to wait before an off measurement should be included as
        part of the fit..
    off_max_wait_time : float
        Maximum time to wait before an off measurement should be included as
        part of the fit.
    min_pwr_setting : float
        Exclude and power levels below this value.
    throw_away_min_time : float
        Throw away samples taken before this time since source adjustment.
    heater : str, optional
        Name of heater. Default is 'SMU'.
    thermometer_monitor : str, optional
        Name of thermomter_monitor. Default is None.
    make_plots : bool, optional
        Generates plots if asked. Default is True.

    Returns
    -------
    parsed_dc : configs.ParsedDCSweep
        Parsed DC measurements.
    figs : list[plt.Figure]
        Analysis Figures

    """

    # distinguish between list of paths and single path
    if isinstance(metadata, str) or isinstance(metadata, Path):
        metadata = [metadata]

    # run through parser to extract parameters from the timeseries
    parsed, figs = staircase_analysis.parse_v1(
        metadata=metadata,
        heater = heater,
        thermometer_monitor =thermometer_monitor,
        thermopile_monitor = thermopile_monitor,
        throw_away_min_time=throw_away_min_time,
        on_min_wait_time=on_min_wait_time,
        on_max_wait_time=on_max_wait_time,
        off_min_wait_time=off_min_wait_time,
        off_max_wait_time=off_max_wait_time,
        min_pwr_setting=min_pwr_setting,
    )

    # attatch metadata
    for output in parsed.values():
        output.attrs['metadata'] = str([str(p) for p in metadata])
    return parsed, figs

def run_gui(
    config_files: list[Path],
    runlist: Path,
    output_dir: Folder,
    meas_name: str = 'dcsweep',
    dry_run: bool = False,
):
    """
    dcsweep runner GUI.

    Parameters
    ----------
    config_files : list[Path]
        Path to the settings config file.
    runlist : Path
        Path to the measurement list file.
    output_dir : Folder
        Directory to output data to.
    meas_name : str, optional
        Name of measurement. Default is 'dcsweep'.
    dry_run : bool, optional
        Does everything up to but not including run the
        experiment. Can be used to validate settings. The default is False.
    """

    commands = [
        'dcsweep',
        'run',
        f'"{str(Path(output_dir))}"',
        ]
    commands += ['--runlist',f'"{str(Path(runlist))}"']

    for c in config_files:
        commands += ['--config-files', f'"{Path(c).resolve()}"']

    commands += ['--meas-name',f'{meas_name}']

    if dry_run:
        commands.append('--dry-run')

    clitools.ucal_cli(commands)


@click.command(name='run')
@click.argument('output_dir', type=Path)
@click.option('--config-files', '-c', type=Path, required=True, multiple=True)
@click.option('--runlist', '-s', type=Path)
@click.option('--meas-name', type=str, default='dcsweep')
@click.option('--dry-run', is_flag=True, default=False)
def _run_cli(*args, **kwargs):
    print(args)
    return run(*args, **kwargs)


_run_cli = clitools.format_from_npdoc(run)(_run_cli)
