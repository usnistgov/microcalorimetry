"""microcalorimeter_runner module."""

import time
import matplotlib
import pylab as plt
import pyvisa as visa
import numpy as np

from pathlib import Path
from datetime import timedelta, datetime
from rmellipse.uobjects import RMEMeas

import microcalorimetry.configs as configs
from microcalorimetry._helpers._intf_tools import ConsoleManager
from microcalorimetry._helpers._collections import get_git_info, get_version

# import any instrument that you might want here.
from rminstr.instruments.Anritsu_MG362x1A import SignalGenerator as Anritsu_MG362x1A
from rminstr.instruments.Anritsu_MG3696A import SignalGenerator as Anritsu_MG3696A
from rminstr.instruments.RS_SMA100B import ArmedSignalGenerator as RS_SMA100B
from rminstr.instruments.HP34420A import Voltmeter as HP34420A_Voltmeter
from rminstr.instruments.HP3458A import Voltmeter as HP3458A_Voltmeter
from rminstr.instruments.K2450 import (
    DCSubPowerMeter,
    SMUSourceSweep as K2450_SMUSourceSweep,
)
from rminstr.instruments.KS_E8257D import SignalGenerator as KS_E8257D
from rminstr.instruments.DP8200 import VoltageGenerator as DP8200
from rminstr.instruments.Fluke_5720A import (
    VoltageGenerator as Fluke_5720A_VoltageGenerator,
)
from rminstr.instruments.RS_NRP75TWG import RFPowerMeter as RS_NRP75TWG_RFPowerMeter
import rminstr.instruments.RS_NRPxxTn as RS_NRPxxTn
from rminstr.instruments.communications import GPIBInterface
from rminstr.data_structures import (
    ExptParameters,
    ActiveRecord,
    kendall_p,
    runs_statistic,
)


# not expected to change very often
def format_pmeter_est_column(port_name: str):
    return f'{port_name}_signal (W)'


def format_voff_slow_column(port_name: str):
    return f'V_off_slow_{port_name}'


# the different names of the sensors ports will go here for
# easy access
SENSOR_PORTS = []

# dBm. power levelling does not occur if the readings are too small to be
# accurate
HARD_MIN_POWER_LEVELLING = -30.0

# dBm. The RF source turns off and the program stops if dc substituted power is
# larger than this value.
HARD_MAX_dBm = {}


HARD_AM_MAX = 0.5  # 0.99999 # hard coded so you won't change it by accident

# indexed by model, role
INSTRUMENT_CLASSES = {
    'Anritsu_MG362x1A': {'RF_source': Anritsu_MG362x1A},
    'Anritsu_MG3696A': {'RF_source': Anritsu_MG3696A},
    'RS_SMA100B': {'RF_source': RS_SMA100B},
    'KS_E8257D': {'RF_source': KS_E8257D},
    'Fluke_5720A': {'RF_amplitude_adjuster': Fluke_5720A_VoltageGenerator},
    'HP34420A': {
        'bias_monitor': HP34420A_Voltmeter,
        'thermopile_monitor': HP34420A_Voltmeter,
        'voltage_monitor': HP34420A_Voltmeter,
    },
    'HP3458A': {
        'bias_monitor': HP3458A_Voltmeter,
        'thermopile_monitor': HP3458A_Voltmeter,
        'voltage_monitor': HP3458A_Voltmeter,
    },
    'K2450': {
        'SMU_power_meter': DCSubPowerMeter,
        'thermometer_monitor': K2450_SMUSourceSweep,
    },
    #   "RS_ZVA67": {"VNA_source": RS_ZVA67_VNA},
    'DP8200': {'RF_amplitude_adjuster': DP8200},
    'RS_NRP75TWG': {
        'power_meter': RS_NRP75TWG_RFPowerMeter,
    },
    'RS_NRPxxTn': {'power_meter': RS_NRPxxTn.RFPowerMeter},
}

# these are the keys use to identify the physical meaning
# of specific data columns in relation to the different
# ports in the measurement. They are used in the column model
# mapping (CMM).
APPLIED_VOLTAGE_CMMKEY = 'vdc'
APPLIED_CURRENT_CMMKEY = 'idc'
CMRCL_POWER_CMMKEY = 'power'
THERMOPILE_VOLTS_CMMKEY = 'e'
SRC_LEVEL_CMMKEY = 'source_level'

NECESSARY_COLUMNS = [
    'step_counter',
    'point_counter',
    'power_on',
    'rf_power_setting',
    'AM_voltage',
    'frequency',
    'stable_samples',
]

# These are used for the flow control algorithm, but not necessary for parsing
# output data. They can be printed out to provide debug information
STATUS_COLUMNS = [
    'point_start_time',
    'last_stats_update_time',
    'last_plot_update_time',
    'timestamp',
    'kendall_p',
    'runs_Z',
]  # "loss_estimate", "compression_estimate", "target_adjustment"]

# these are just logical groupings of the status columns
# so that I can format the print logs a little nicer
TIME_STATUS_COLUMNS = [
    'timestamp',
    'point_start_time',
    'last_stats_update_time',
    'last_plot_update_time',
]
POSITION_STATUS_COLUMNS = ['step_counter', 'point_counter']
METERING_STATUS_COLUMNS = []
SOURCE_STATUS_COLUMNS = ['rf_power_setting', 'AM_voltage', 'frequency', 'power_on']
STABILITY_STATUS_COLUMNS = ['stable_samples', 'kendall_p', 'runs_Z']

# These are all supplied at runntime by a sensor master list,
# initializing here
SPECIAL_MOUNTS = []
THIN_FILM_MOUNTS = []  # for these mounts, use a PTC feedback loop
THERMISTOR_MOUNTS = []  # for these mounts, use a NTC feedback loop
COMMERCIAL_MOUNTS = []
KEYSIGHT_THERMOPILE_BALANCE_MOUNTS = []  # Keysight Thermopile Balance Sensors
ALL_MOUNTS = []
EXPECTED_LINEAR_TERM_BOUNDS = {}
EXPECTED_RESISTANCE = {}
RFSOURCES = []

# THese are different types of instrument roles, relevant to their
# behavior during the course of a measurement
# all voltage monitor roles do the same thing.
# the different names are to help make the config file more readable
CMRCL_POWER_METER_ROLES = ['power_meter']
VOLTAGE_MONITOR_ROLES = ['voltage_monitor', 'bias_monitor', 'thermopile_monitor']
SOURCE_ROLES = ['RF_source', 'VNA_source']
RF_AMPLITUDE_ADJUSTER_ROLES = ['RF_amplitude_adjuster']
SMU_POWER_METERS = ['PTC_SMU', 'NTC_SMU']
THERMOMETER_ROLES = ['thermometer_monitor']
THIN_FILM_DC_SOURCE_TYPES = ['PTC_SMU', 'PTC_TYPE_IV', 'DC_VOLTAGE']
THERMISTOR_DC_SOURCE_TYPES = ['NTC_SMU', 'NTC_TYPE_IV', 'DC_CURRENT']

# path to schema for a config file
SCHEMA = configs.load_config(configs.SCHEMA_DIR / 'RFSweepConfiguration.json')

SIGNAL_CONFIG_KEY = 'signal_config'

# manages the console output
# by tracking new lines, and logs
# to a txt file over time.
CONSOLE_MANAGER = ConsoleManager()
print = CONSOLE_MANAGER.cprint


class MissingColumnError(Exception):
    """Error for when a column is missing."""

    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class MicrocalorimeterRunner:
    """
    The microcalorimeter_runner class runs microcalorimeter measurements.

    A measurement proceedure is defined by two files: a config file, and a
    run_settings file. Please see "test_data" folder for examples.
    """

    def __init__(
        self,
        config_files: str | list[str],
        run_settings_file: str,
        output_dir: str,
        sensor_master_list: str | Path,
        config_file_priority: list[int] = None,
        no_confirm: bool = False,
        dry_run: bool = False,
        validate: bool = True,
    ):
        """
        Initialize a microcalorimeter_runner object.

        Parameters
        ----------
        config_files : Union[str, list[str]]
            If the value is a string, it is interpreted as the path to the
            config file.

            If as list of strings is given, each entry is interpreted as
            the path to a config file. Each config file is read in the order
            given in the list. If the same setting exists in multiple config
            files, use config_file_priority to determine which setting to use.

        config_file_priority : list[int], optional
            Indicates the priority of the corresponding (by order) config
            file. Higher priority corresponds to lower numbers.If None,
            all files have equal priority. The default is None.

        config_file : str
            Path to the config file.

        run_settings_file : str
            Path to the run settings file.

        output_dir : str
            Directory where data will be stored.

        no_confirm : bool
            If True, skip confirmation inputs and just run the experiment.

        dry_run : bool, optional
            Does not connect to any intruments and bounces out after
            validation the configurations (if True). The default is False.

        Returns
        -------
        None.

        """
        # so that plotting will work
        matplotlib.use('agg')

        sensor_master_list = configs.RFSensorMasterList(Path(sensor_master_list))

        # assign the master list variables to global constants
        global SPECIAL_MOUNTS, THIN_FILM_MOUNTS
        global THERMISTOR_MOUNTS, KEYSIGHT_THERMOPILE_BALANCE_MOUNTS
        global EXPECTED_RESISTANCE, ALL_MOUNTS, EXPECTED_LINEAR_TERM_BOUNDS
        global CALORIMETERS

        SPECIAL_MOUNTS = sensor_master_list['SPECIAL_MOUNTS']
        print(SPECIAL_MOUNTS)
        THIN_FILM_MOUNTS = sensor_master_list['THIN_FILM_MOUNTS']
        THERMISTOR_MOUNTS = sensor_master_list['THERMISTOR_MOUNTS']
        COMMERCIAL_MOUNTS = sensor_master_list['COMMERCIAL_MOUNTS']
        KEYSIGHT_THERMOPILE_BALANCE_MOUNTS = sensor_master_list[
            'KEYSIGHT_THERMOPILE_BALANCE_MOUNTS'
        ]
        EXPECTED_RESISTANCE = sensor_master_list['EXPECTED_RESISTANCE']
        EXPECTED_LINEAR_TERM_BOUNDS = sensor_master_list['EXPECTED_LINEAR_TERM_BOUNDS']
        CALORIMETERS = sensor_master_list['CALORIMETERS']
        RFSOURCES = sensor_master_list['RFSOURCES']
        ALL_MOUNTS = (
            COMMERCIAL_MOUNTS
            + SPECIAL_MOUNTS
            + THIN_FILM_MOUNTS
            + THERMISTOR_MOUNTS
            + KEYSIGHT_THERMOPILE_BALANCE_MOUNTS
            + CALORIMETERS
            + RFSOURCES
        )

        self.no_confirm = no_confirm
        self.output_dir = output_dir
        self.parameters = ExptParameters(
            config_files,
            run_settings_file=run_settings_file,
            config_file_priority=config_file_priority,
        )
        #  this is a little silly, but the presence of the run settings columns
        # in the config dictionary causes the validations to fail, and this is a
        # quick solution in the mean time.
        if validate:
            self._parameters_no_runlist = ExptParameters(
                config_files, config_file_priority=config_file_priority
            )
            configs.RFSweepConfiguration(self._parameters_no_runlist.config)

        # use the column model mapping to create metering status columns
        # for each sensor that's been mapped to an instrument column
        # the exact names are enforced by the schema for the config file,
        # but doing it this way will let us add more arbitrary sensors
        # in the future
        global STATUS_COLUMNS, METERING_STATUS_COLUMNS, SENSOR_PORTS
        global HARD_MAX_dBm
        for pname, mapping in dict(self.parameters[SIGNAL_CONFIG_KEY]).items():
            cname = format_pmeter_est_column(pname)
            SENSOR_PORTS.append(pname)
            STATUS_COLUMNS.append(cname)
            METERING_STATUS_COLUMNS.append(cname)
            HARD_MAX_dBm[pname] = self.parameters['levelling_settings']['HARD_MAX_dBm'][
                pname
            ]
            if mapping['type'] == 'bolometer':
                vslow_cname = format_voff_slow_column(pname)
                # SENSOR_PORTS.append(vslow_cname)
                METERING_STATUS_COLUMNS.append(vslow_cname)
                STATUS_COLUMNS.append(vslow_cname)
        # print(SENSOR_PORTS)
        # add extra output columns
        extra_output_columns = None
        try:
            extra_output_columns = self.parameters['output_settings']['columns']

        except KeyError:
            pass

        if extra_output_columns is None:
            extra_output_columns = []

        if type(extra_output_columns) is str:
            extra_output_columns = [extra_output_columns]

        if type(extra_output_columns) is not list:
            raise ValueError('unexpected data in output_settings, columns.')

        # linear term of sensitivity coefficients for sensors
        # these will get set during the validation
        # stage if they match the expected master
        # list, set them as empty here so they throw errors
        # if I try and do math with them later on
        # and they arent set.
        self.sensitivity_linear_term = {}

        # print("extra_columns", extra_output_columns)
        self.record_columns = list(
            set(NECESSARY_COLUMNS + STATUS_COLUMNS + extra_output_columns)
        )

        # parameter validation section
        # TODO: add validation checks for other parameters
        # validate mount resistance
        self._validate_sensor_settings()

        # stats window needs to be < minimum wait if not using traditional stats
        # so that it doesn't creep into more than 1 step when you go to do 
        # analysis
        traditional_stats = self.parameters['stats_settings']['use_traditional_stats']
        stats_window = self.parameters['stats_settings']['stats_window']
        min_wait = self.parameters['stats_settings']['minimum_wait']
        if not traditional_stats and stats_window >= min_wait:
            raise ValueError("stats_window must be less than minimum_wait if not use_traditional_stats.")


        # initializes an active data record
        self.record = ActiveRecord(
            self.record_columns,
            maxlen=self.parameters['output_settings']['maxlen'],
            output_dir=output_dir,
            minlen=self.parameters['output_settings']['minlen'],
            stage_everything=True,
            meas_name='rfsweep',
        )
        self.output_dir = output_dir

        # manages console log output
        self.console_log_file = Path(
            self.record.metadata['local backups'] / 'console-log.txt'
        )
        CONSOLE_MANAGER.fio = open(self.console_log_file, 'w')

        # save a copu of the configuration files to the target directory
        config_file = str(Path(output_dir) / (self.record.session_str + '_config.csv'))
        settings_file = str(
            Path(output_dir) / (self.record.session_str + '_settings.csv')
        )

        self.record.metadata['config_file'] = config_file
        self.record.metadata['settings_file'] = settings_file
        self.record.metadata['microcalorimetry_version'] = get_version('microcalorimetry')
        
        # add any git infor available about the
        # repository the source code lives in
        microcalorimetry_git_info = get_git_info(__file__)
        for k,v in microcalorimetry_git_info.items():
            self.record.metadata[k] = v
        

        self.parameters.save_config(config_file)
        self.parameters.save_run_settings(settings_file)

        # convenience variables for easier access
        self.signal_configs = self.parameters['signal_config']

        if dry_run:
            return
        self.resource_manager = visa.ResourceManager()
        self.gpib_interface = GPIBInterface(
            self.parameters['gpib_interface'], resource_manager=self.resource_manager
        )
        self.instruments = {}
        self.source_name = None
        self.source_type = None
        self.power_meter_name = None
        self.bias_monitor_name = None
        self.voltage_monitor_names = None
        self.commercial_power_meter_names = None
        self.rf_amplitude_adjuster_name = None
        self.thermometer_monitor_names = None

        self.source = None
        self.power_meter = None
        self.rf_amplitude_adjuster = None
        self.bias_monitor = None
        self.voltage_monitors = None
        self.commercial_power_meters = None
        self.thermometer_monitors = None

        # log variabel
        self.log_line_count = 0

        # intialize state variables
        self.index = 0
        self.done = False
        self.record['step_counter'] = self.index
        self.record['point_counter'] = self.parameters.index
        self.record['power_on'] = False
        # self.record["rf_power_setting"] = None
        # self.record["AM_voltage"] = None
        # self.record["frequency"] = None
        # self.record["DVM_volts"] = None
        # self.record["NVM_volts"] = None
        self.record['stable_samples'] = 0
        # self.record["compression_estimate"] = None
        # self.record["loss_estimate"] = None
        # self.record["target_adjustment"] = 0

        # self.record["V_off_slow"] = None
        # self.record["point_start_time"] = None
        self.record['last_stats_update_time'] = time.time() - self.record.time_zero
        self.record['last_plot_update_time'] = time.time() - self.record.time_zero

        # this is so the cleanup function gets called at the end of an experiment
        # and the context manager knows not to call it again.
        self.closed = False
        CONSOLE_MANAGER.set_origin()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.final_cleanup()

    def _validate_single_signal(self, port_name, device_name):
        sensor_type = self.parameters[SIGNAL_CONFIG_KEY][port_name]['type']
        all_instruments = self.parameters['instruments']['names']
        # check that any physical quantities have
        # columns and instruments that actually exists,
        # and that the columns actually do belong to that
        # instrument
        for item_name, item in dict(
            self.parameters[SIGNAL_CONFIG_KEY][port_name]
        ).items():
            # if the signal has any physical quantities
            # then check the columns exist
            column = None
            try:
                column = item['column']
                if column not in self.record_columns:
                    raise MissingColumnError(
                        f'{item["column"]} was mapped to {port_name} : {item_name} but is not in the record {self.record_columns}.'
                    )
            except (KeyError, TypeError):
                pass
            # check that the instrument associated with each column
            # also eists
            instrument = None
            try:
                instrument = item['instrument']
                if instrument not in all_instruments:
                    raise MissingColumnError(
                        f'{item["instrument"]} was mapped to {port_name} : {item_name} but is not a defined instrument {all_instruments}.'
                    )
            except (KeyError, TypeError):
                pass

            # check that the column belongs to instrument
            if instrument is None and column is None:
                pass
            # if one is none and the other isn't then something
            # is missing a defintiomn
            elif (instrument is None) ^ (column is None):
                raise ValueError(
                    'Both instrument and column must be defined {port_name}:{item_name}'
                )
            else:
                matches = False
                for key, item in dict(
                    self.parameters['instruments'][instrument]
                ).items():
                    if 'column' in key:
                        if item == column:
                            matches = True
                if not matches:
                    raise Exception(
                        f'column {column}  does not belong to instrument {instrument} hint: did you pick the right column/instrument for the {port_name} signal?'
                    )

        # check the resistance matches what is expected
        if device_name not in ALL_MOUNTS:
            raise ValueError(
                f'{device_name} not recognized, add it to the master list.'
            )

        # thermistor or thin_film mounts need to have a matching resistance
        # to what is expected, or if it just has an expected resistance.
        if (
            device_name in THERMISTOR_MOUNTS
            or device_name in THIN_FILM_MOUNTS
            or device_name in EXPECTED_RESISTANCE
        ):
            expected_R = EXPECTED_RESISTANCE[device_name]
            R = self.parameters[SIGNAL_CONFIG_KEY][port_name]['resistance']
            if R != expected_R:
                msg = 'You are using mount {} with resistance {} Ohms. \n'.format(
                    self.mount, R
                )
                msg += 'Expected {} Ohms.\n'.format(expected_R)
                msg += 'Incorrect resistance values can lead to incorrect power levelling, and equipment damage.'
                raise ValueError(msg)

            # make sure it's a bolometer or special
            # some of the special sensor's can be biased, but not power levelled
            if sensor_type != 'bolometer' and sensor_type != 'special':
                raise ValueError(
                    f'Expected sensor_type = bolometer for {port_name} {device_name}'
                )

        if (
            device_name in KEYSIGHT_THERMOPILE_BALANCE_MOUNTS
            or device_name in EXPECTED_LINEAR_TERM_BOUNDS
            or device_name in CALORIMETERS
        ):
            if sensor_type != 'thermoelectric':
                raise ValueError(
                    f'Expected sensor_type = thermoelectric for {port_name} {device_name} because sensor has an expected linear sensitivity term in master list'
                )

            try:
                expected_range = EXPECTED_LINEAR_TERM_BOUNDS[device_name]
            except KeyError as e:
                raise KeyError(
                    f'No expected linear coefficient bounds defined for {device_name} in master list.'
                )

            # i could infer the min/max, but I want the
            # person writing the ranges to be explicit
            # so I am forcing them to properly order their
            # list (I infer it anyways after because the code
            # is easier bu still dont come after me.)
            if expected_range[0] > expected_range[1]:
                raise ValueError(
                    f'expected linear term range of {
                        device_name
                    } should be formatted (min,max), first item <= second item.'
                )

            coeffs = configs.KDCLike(
                self.parameters[SIGNAL_CONFIG_KEY][port_name]['coeffs']
            ).load()

            # if it was a path, then load that bad boy in and check the linear term
            # if it's a simple polynomial model, then sensitivity is
            # the 0th order term
            if isinstance(coeffs, RMEMeas) and 'deg' in coeffs.dims:
                # if p_of_e, then W/V, expecting V/W
                if coeffs.attrs['p_of_e']:
                    linear_term = 1 / float(coeffs.nom.sel(deg=1))
                else:
                    linear_term = float(coeffs.nom.sel(deg=1))

            # if a dimension called column is present its a temperature
            # dependent model, so the linear approximation is the
            # constant c
            elif isinstance(coeffs, RMEMeas) and 'col' in coeffs.dims:
                linear_term = 1 / float(coeffs.nom.sel(col='c'))

            # you could load in a float too, that's cool.
            elif isinstance(coeffs, float):
                linear_term = coeffs

            else:
                raise TypeError(
                    rf'unexpected type of {
                        coeffs
                    }, expected RMEMeas or float (can provide the linear term as a float in V/W).'
                )

            # okedoke check it
            if linear_term < min(expected_range) or linear_term > max(expected_range):
                raise ValueError(
                    f"The provided coefficients of {
                        device_name
                    } don't match the expected linar term bounds."
                )

            # reassign the linear term to the dictionairy.
            self.sensitivity_linear_term[port_name] = linear_term

        # do some more specific error checking
        use_GPIB_levelling = self.parameters['levelling_settings']['use_GPIB_levelling']
        use_AM_levelling = self.parameters['levelling_settings']['use_AM_levelling']
        DC_source_type = self.parameters['measurement_description']['DC_source_type']
        # check that the DC source type selected is acceptable
        if (
            device_name in THIN_FILM_MOUNTS
            and DC_source_type not in THIN_FILM_DC_SOURCE_TYPES
        ):
            raise (
                ValueError(
                    device_name
                    + ' is a thin film mount, which is not compatible with '
                    + DC_source_type
                    + ' as a DC source'
                )
            )
        if (
            device_name in THERMISTOR_MOUNTS
            and DC_source_type not in THERMISTOR_DC_SOURCE_TYPES
        ):
            raise (
                ValueError(
                    device_name
                    + ' is a thermistor mount, which is not compatible with '
                    + DC_source_type
                    + ' as a DC source'
                )
            )

        # only do levelling to the dut
        if port_name == self.parameters['levelling_settings']['level_to']:
            if device_name in SPECIAL_MOUNTS and (
                use_GPIB_levelling or use_AM_levelling
            ):
                raise (
                    ValueError(
                        device_name
                        + ' is a special mount, so power levelling is impossible'
                    )
                )

    def _validate_sensor_settings(self):
        """
        Check if mount name and related settings are compatible.

        Helper function for __init__.

        Raises
        ------
        ValueError
            If there are inconsistant settings.

        Returns
        -------
        None.

        """
        # validate configuration against the defined json schema.
        # which will help keep the documentation up to date.
        # the runlist can be validated independently

        level_to = self.parameters['levelling_settings']['level_to']
        if level_to not in SENSOR_PORTS:
            raise ValueError(
                f'level_to setting {level_to} not a defined port: {SENSOR_PORTS}.'
            )
        for port_name in SENSOR_PORTS:
            # the Voff slow isn't a signal but gets treated like one?
            if 'V_off_slow' not in port_name:
                device_name = self.parameters['measurement_description'][
                    port_name.replace('_power', '_name')
                ]
                self._validate_single_signal(port_name, device_name)

    def _init_single_instrument(self, name: str):
        """
        Initialize individual instruments that are configured.

        Parameters
        ----------
        name : str
            Instrument name as defined by the configuration files.
        """
        print(f'initializing, {name}')

        # model determines which constructor is called
        model = self.parameters['instruments'][name]['model']
        role = self.parameters['instruments'][name]['role']
        address = self.parameters['instruments'][name]['GPIB_address']

        instrument = INSTRUMENT_CLASSES[model][role](
            address, resource_manager=self.resource_manager
        )
        self.instruments[name] = instrument
        # print("visa resource open")

        instrument_info = instrument.get_info()

        if 'serial' not in instrument_info:
            serial = None
            try:
                serial = self.parameters['instruments'][name]['tag']

            except KeyError:
                print(f'Warning: {name} does not have a serial number')

            instrument_info['tag'] = serial

        for key, value in instrument_info.items():
            new_key = name + '_' + key
            self.record.metadata[new_key] = value

        # call intial_setup for everything
        try:
            initial_setup_kwargs = self.parameters['instruments'][name][
                'initial_settings'
            ]
            instrument.initial_setup(**initial_setup_kwargs)

        except KeyError:
            instrument.initial_setup()
        except visa.errors.VisaIOError as e:
            msg = f"Failed to setup {name} which is a {model}, caught VisaIOError: {str(e)}"
            raise Exception(msg) from e

        # some instruments have special names
        if role == 'SMU_power_meter':
            self.power_meter_name = name
            self.power_meter = instrument

        if role in SOURCE_ROLES:
            self.source_name = name
            self.source_type = self.parameters['measurement_description'][
                'RF_source_type'
            ]
            self.source = instrument

        if role in VOLTAGE_MONITOR_ROLES:
            self.voltage_monitor_names.append(name)
            self.voltage_monitors.append(instrument)

        if role in CMRCL_POWER_METER_ROLES:
            self.commercial_power_meter_names.append(name)
            self.commercial_power_meters.append(instrument)

        if role in RF_AMPLITUDE_ADJUSTER_ROLES:
            self.rf_amplitude_adjuster_name = name
            self.rf_amplitude_adjuster = instrument

        if role in THERMOMETER_ROLES:
            self.thermometer_monitor_names.append(name)
            self.thermometer_monitors.append(instrument)

    def initialize_instruments(self):
        """
        Initialize instrument objects

        Raises
        ------
        ValueError
            If there are inconsistant settings.

        Returns
        -------
        None.

        """
        # the names of the instruments are arbitrary. They just serve as keys
        names = self.parameters['instruments']['names']

        # this block needs to go here so the instruments
        # have a chance to initialize properly into their roles
        # if they don't respond well to an IDN string query
        failed = 0
        for i, name in enumerate(names):
            print('')
            print(f'Validating Instrument *IDN? : {name}')
            print(f'    serial : {self.parameters["instruments"][name]["serial"]}')
            try:
                expected = self.parameters['instruments'][name]['*IDN?']
            except KeyError:
                print('    No *IDN? specified. Skipping validation.')
                continue
            visa_address = self.parameters['instruments'][name]['GPIB_address']
            rm = visa.ResourceManager()
            try:
                connection = rm.open_resource(visa_address)
                idn = str(connection.query('*IDN?')).strip()
            except Exception as e:
                idn = f'error - {e}'
            connection.close()
            print(f'  expected : {expected}')
            print(f'     *IDN? : {idn}')
            if idn != expected:
                print('    match? : No')
                failed += 1
            else:
                print('    match? : Yes')

        if failed > 0 and not self.no_confirm:
            print('-------------------------------------------------------------')
            input(
                'WARNING: Some *IDN? dont match the expected value. \n press anything to continue or ctrl + c to cancel.>>'
            )

        # Initialize instruments into their correct interfaces
        # now that we've confirmed they are correct
        self.voltage_monitor_names = []
        self.voltage_monitors = []
        self.commercial_power_meter_names = []
        self.commercial_power_meters = []
        self.thermometer_monitors = []
        self.thermometer_monitor_names = []
        for name in names:
            # role determines which constructor is called
            role = self.parameters['instruments'][name]['role']
            print('Checking config', name, role)
            # instruments that act individually
            self._init_single_instrument(name)

        # if there is a power meter, pass the resistance setting
        DC_source_type = self.parameters['measurement_description']['DC_source_type']
        if DC_source_type in SMU_POWER_METERS:
            sensor_type = self.parameters['instruments'][self.power_meter_name][
                'initial_settings'
            ]['sensor_type']

            # we also need to check if the mount is a thin-film mount
            # this check happens in _validate_sensor_settings, which is
            # called by __init__

            if (
                sensor_type == 'platinum_thinfilm'
                and DC_source_type not in THIN_FILM_DC_SOURCE_TYPES
            ):
                raise ValueError(
                    'SMU set up as platinum_thinfilm while DC source type was'
                    + DC_source_type
                )

            if (
                sensor_type == 'thermistor'
                and DC_source_type not in THERMISTOR_DC_SOURCE_TYPES
            ):
                raise ValueError(
                    'SMU set up as thermistor while DC source type was '
                    + DC_source_type
                )

            R = self.parameters['levelling_settings']['resistance']

            # fetch data before sending settings to the the power meter.
            # this shouldn't do anything
            self._fetch_data(all_power_data=False)
            power_meter = self.power_meter
            power_meter.setup(resistance_setpoint=R)

        # there may also be commands given to the setup method at the start
        # of the measurement
        self._load_instrument_settings('initial_settings')

        CONSOLE_MANAGER.set_origin()

    def _calculate_signals(self):
        """
        Estimate the metered power (in Watts) of each sensor and add it to the DataRecord.

        Estimated power is immediatley updated, and does not requied a batch update.
        """
        sensors = list(dict(self.signal_configs).keys())
        for sensor in sensors:
            mapping = self.signal_configs[sensor]
            metered_power_column = format_pmeter_est_column(sensor)
            # get an estimate of the metered power at the sensor
            # based on it's mapping, using the most recent samples available
            # commercial sensors should just be reporting power in W
            if mapping['type'] == 'commercial':
                quantity = mapping[CMRCL_POWER_CMMKEY]
                estimated_power = self.record[quantity['column']]
            # bolometers could be a Type IV with a voltage,
            # or could be an SMU with V and I
            elif mapping['type'] == 'bolometer':
                voltage_quantity = mapping[APPLIED_VOLTAGE_CMMKEY]
                try:
                    current_quantity = mapping[APPLIED_CURRENT_CMMKEY]
                except KeyError:
                    current_quantity = None
                if current_quantity is None:
                    vslow_cname = format_voff_slow_column(sensor)
                    slow_off = self.record[vslow_cname] ** 2 / mapping['resistance']
                    current = (
                        self.record[voltage_quantity['column']] ** 2
                        / mapping['resistance']
                    )
                    estimated_power = slow_off - current
                else:
                    raise NotImplementedError('Signal for SMU needs to be added')

            # thermoelectric signals just produce a thermoelectric voltage
            elif mapping['type'] == 'thermoelectric':
                e_quant = mapping[THERMOPILE_VOLTS_CMMKEY]
                column = e_quant['column']
                # print(self.sensitivity_linear_term[sensor])
                estimated_power = (
                    self.record[column] / self.sensitivity_linear_term[sensor]
                )
                # sometimes the linear term is negative, I am ignoring
                # the offsets here
                estimated_power = abs(estimated_power)

            # RF sources have a setting and potentially a voltage
            # for amplitufe adjustment
            elif mapping['type'] == 'RF_source':
                pow_quantity = mapping[CMRCL_POWER_CMMKEY]
                try:
                    vdc_quantity = mapping[APPLIED_VOLTAGE_CMMKEY]
                except KeyError:
                    vdc_quantity = None

                estimated_power_dBm = self.record[pow_quantity['column']]
                estimated_power = abs(10 ** (estimated_power_dBm / 10) / 1000)
                # if not none, then try and estimate the power with the AM
                if vdc_quantity is not None:
                    instr = self.instruments[pow_quantity['instrument']]
                    AM_per_key = 'AM_ext_sensitivity_percent_per_volt'
                    try:
                        percent_per_volt = instr.setup_settings[AM_per_key]
                    except KeyError:
                        percent_per_volt = instr.initial_setup_settings[AM_per_key]
                    v_am = self.record[vdc_quantity['column']]
                    estimated_power += estimated_power * percent_per_volt / 100 * v_am

            # special sensors dont report power
            elif mapping['type'] == 'special':
                estimated_power = np.nan

            # otherwise this type of signal hasnt been implemented
            else:
                raise NotImplementedError(
                    f'Signal type {mapping["type"]} hasnt been added.'
                )
            # add these to the data record
            self.record.update(metered_power_column, estimated_power)

    def output(self):
        """
        Print out state of the experiment.

        Returns
        -------
        None.

        """
        if self.done:
            return

        # CONSOLE_MANAGER.wipe_to_origin()
        print('-' * 60)
        RJ = 30

        def format_column(name, value):
            return f'{name} | '.rjust(RJ) + f'{value}'

        # output minimum time left in a step
        wait_time = self._get_min_wait_time_left()
        if wait_time > 0:
            step_time_left = str(timedelta(seconds=self._get_min_wait_time_left()))
        else:
            step_time_left = 'waiting for stability'

        unreported = [c for c in self.record.columns]

        for cname in METERING_STATUS_COLUMNS:
            unreported.pop(unreported.index(cname))
            value = self.record[cname]
            if isinstance(value, float):
                print(f'{cname} | '.rjust(RJ) + f'{self.record[cname]:.3e}')
            else:
                print(f'{cname} | '.rjust(RJ) + f'{self.record[cname]}')
        print('')

        for cname in SOURCE_STATUS_COLUMNS:
            unreported.pop(unreported.index(cname))
            print(format_column(cname, self.record[cname]))
        if self.record['power_on']:
            print(
                format_column(
                    'Target Power Level (dBm)',
                    self.parameters['Target_source_power_dBm'],
                )
            )
            level_to = self.parameters['levelling_settings']['level_to']
            levelling_pow = self.record[format_pmeter_est_column(level_to)]
            levelling_pow_dBm = 10.0 * np.log10(1000 * levelling_pow)
            print(format_column('Feedback Power (dBm)', levelling_pow_dBm))
            print(
                format_column(
                    'Levelling To', self.parameters['levelling_settings']['level_to']
                )
            )

        print('')
        # something to estaimte the end time of the experiment should go here
        print(format_column('min time left in point', step_time_left))
        for cname in TIME_STATUS_COLUMNS:
            unreported.pop(unreported.index(cname))
            print(format_column(cname, self.record[cname]))
        print('')

        for cname in STABILITY_STATUS_COLUMNS:
            unreported.pop(unreported.index(cname))
            print(format_column(cname, self.record[cname]))
        print('')

        for cname in POSITION_STATUS_COLUMNS:
            unreported.pop(unreported.index(cname))
            print(format_column(cname, self.record[cname]))
        print('')

        # print every other column that hasnt been reported
        for cname in unreported:
            print(format_column(cname, self.record[cname]))

        # try to plot what is stored in memory and
        # isnt yet accesible in saved data_record
        short_plot_window = self.parameters['output_settings']['short_plot_time_window']
        plot_interval = self.parameters['output_settings']['plot_interval']
        last_plot_update_time = self.record['last_plot_update_time']
        current_time = self.record['timestamp']

        if current_time - last_plot_update_time < plot_interval:
            return
        dvm_volts_present = True
        try:
            tV, V = self.record.get_time_series(
                'DVM_volts', t_max=current_time, t_min=current_time - short_plot_window
            )
        except (KeyError, IndexError):
            dvm_volts_present = False
            print('DVM_volts not present or failed to read.')



        try:
            te, e = self.record.get_time_series(
                'NVM_volts', t_max=current_time, t_min=current_time - short_plot_window
            )
            
            font = {'size': 10}

            plt.rc('font', **font)
            plt.rc('figure', dpi=300)
            fig_width = 6
            fig_height = 6

            fig, ax = plt.subplots(2, 1)
            fig.set_size_inches(fig_width, fig_height)

            if len(te) > 0:
                ax[0].plot(te, 1.0e6 * (e - e[-1]))
                ax[0].set_ylabel('Delta NVM voltage (uV)')

            if dvm_volts_present:
                if len(tV) > 0:
                    ax[1].plot(tV, 1000.0 * (V - V[-1]))
                    ax[1].set_xlabel('Time (s)')
                    ax[1].set_ylabel('Delta DVM voltage (mV)')

            fig.tight_layout()
            fig.savefig(self.output_dir + self.record.session_str + '_monitor.png')
            plt.close()

        except PermissionError:
            print('Please close the output graph files so that they can be updated.')

        except FileNotFoundError:
            print('Tried to output graph, file path not valid')
        except KeyError:
            print("No NVM Volts column, not plotting.")
        plt.close('all')

        self.record['last_plot_update_time'] = current_time

    def _get_min_wait_time_left(self):
        """
        Get the time left to wait in a step.

        Returns
        -------
        time_left : float
            Time stamp of amount of time left in step (at minimum)

        """
        # Determine if warmup is complete
        j = self.record['point_counter']
        current_time = self.record['timestamp']
        initial_wait = self.parameters['stats_settings']['initial_wait']
        minimum_wait = self.parameters['stats_settings']['minimum_wait']

        point_start_time = self.record['point_start_time']
        if self.parameters['stats_settings']['use_traditional_stats']:
            stats_window = self.parameters['stats_settings']['stats_window']
            minimum_wait = max(stats_window, minimum_wait)

        # i dont think this is true
        if j is None:
            wait_time = initial_wait

        else:
            wait_time = minimum_wait
        return wait_time - (current_time - point_start_time)

    def update_statistics(self):
        """
        Update statistics used to determine stability.

        Returns
        -------
        None.

        """
        measurement_interval = self.parameters['stats_settings']['measurement_interval']
        stats_window = self.parameters['stats_settings']['stats_window']

        last_stats_update_time = self.record['last_stats_update_time']
        point_start_time = self.record['point_start_time']
        current_time = self.record['timestamp']

        # we don't want to update the statistics too often
        if current_time - last_stats_update_time < measurement_interval:
            return

        # Determine if warmup is complete
        j = self.record['point_counter']
        initial_wait = self.parameters['stats_settings']['initial_wait']
        minimum_wait = self.parameters['stats_settings']['minimum_wait']

        if j is None:
            wait_time = initial_wait

        else:
            wait_time = minimum_wait

        if current_time - point_start_time < wait_time:
            return

        use_traditional_stats = self.parameters['stats_settings'][
            'use_traditional_stats'
        ]
        # don't compute stats if not enough data
        if (current_time - point_start_time < stats_window) and use_traditional_stats:
            return

        traditional_stats_good = False

        # If power is off and we are using a bolometer
        # update V_off_slow for each bolometer so it can.
        # be usd to estimate the dc power.
        if not self.record['power_on']:
            for sensor in SENSOR_PORTS:
                sensor_type = self.parameters['signal_config'][sensor]['type']
                if sensor_type == 'bolometer':
                    # THIS NEEDS TO MAKE SURE ITS ONLY USING SAMPLES FROM THE CURRENT
                    # STEP
                    column =  self.signal_configs[sensor][APPLIED_VOLTAGE_CMMKEY]['column']
                    # tV, V = self.record.get_time_series(
                    #     column,
                    #     delta_t=measurement_interval,
                    #     t_max=current_time,
                    #     t_min=current_time - stats_window,
                    # )
                    # V_mean = np.mean(V)
                    V_mean = self.record[column]
                    V_big_enough = (
                        V_mean > self.parameters['levelling_settings']['V_off_slow_min']
                    )
                    V_small_enough = (
                        V_mean < self.parameters['levelling_settings']['V_off_slow_max']
                    )
                    V_good = V_big_enough and V_small_enough
                    if not V_good:
                        self.final_cleanup()
                        raise (Exception(f'V_off = {V_mean} not in expected range'))
                    self.record[f'V_off_slow_{sensor}'] = V_mean

        if use_traditional_stats:
            te, e = self.record.get_time_series(
                self.signal_configs['calorimeter_power'][THERMOPILE_VOLTS_CMMKEY][
                    'column'
                ],
                delta_t=measurement_interval,
                t_max=current_time,
                t_min=current_time - stats_window,
            )
            self.record['kendall_p'] = kendall_p(e)
            self.record['runs_Z'] = runs_statistic(e)

            stats_Tcv = self.parameters['stats_settings']['stats_Tcv']
            stats_Rcv = self.parameters['stats_settings']['stats_Rcv']
            kendall_p_val = self.record['kendall_p']
            runs_Z_val = self.record['runs_Z']

            kendall_p_test = (
                kendall_p_val > stats_Tcv and kendall_p_val < 1.0 - stats_Tcv
            )
            runs_Z_test = runs_Z_val > stats_Rcv
            traditional_stats_good = kendall_p_test and runs_Z_test

        stats_stable = not use_traditional_stats or traditional_stats_good

        if stats_stable:
            self.record['stable_samples'] = self.record['stable_samples'] + 1

        self.record['last_stats_update_time'] = current_time

        return stats_stable

    def _fetch_data(
        self,
        all_voltage_data: bool = False,
        all_power_data: bool = False,
    ):
        """
        Wait for power meter measurements to complete, fetches data, add
        data to record.

        Parameters
        ----------

        all_power_data : bool, optional
            If True, add entire timeseries fetched from power sensor. If False,
            add only the last measurement. The default is False.

        Returns
        -------
        None.

        """
        # timestamp represents the approximate time of the voltage reading
        self.record['timestamp'] = time.time() - self.record.time_zero

        # fetch power meter data
        if self.power_meter is not None:
            state = self.power_meter.query_state()
            if state in ['measuring', 'data_available']:
                # start_wait = time.time()
                self.power_meter.wait_until_data_available()
                # end_wait = time.time()

                # print("waited {}s for power meter".format(end_wait - start_wait))
                out_data = self.power_meter.fetch_data()

                voltage_column = self.parameters['instruments'][self.power_meter_name][
                    'voltage_output_column'
                ]
                current_column = self.parameters['instruments'][self.power_meter_name][
                    'current_output_column'
                ]

                timestamps = out_data['timestamp']
                voltages = out_data['Voltage (V)']
                currents = out_data['Current (A)']

                if all_power_data:
                    self.record.stage_update(voltage_column, voltages, timestamps)
                    self.record.stage_update(current_column, currents, timestamps)

                else:
                    timestamp = self.record['timestamp']
                    self.record.stage_update(
                        voltage_column, [voltages[-1]], [timestamps[-1]]
                    )
                    self.record.stage_update(
                        current_column, [currents[-1]], [timestamps[-1]]
                    )

        if self.source_type == 'VNA':
            state = self.source.query_state()
            if state in ['measuring', 'data_available']:
                num_params = self.source.getNumParams()
                self.source.wait_until_data_available()
                frequencies, times, waves = self.source.fetch_data()

                for k in range(1, num_params + 1):
                    real_column = self.parameters['instruments'][self.source_name][
                        'output_column_{}r'.format(k)
                    ]
                    imag_column = self.parameters['instruments'][self.source_name][
                        'output_column_{}i'.format(k)
                    ]

                    param_data = waves[-1, (k - 1) * 2 : k * 2]
                    timestamp = times[-1]
                    self.record.stage_update(real_column, [param_data[0]], [timestamp])
                    self.record.stage_update(imag_column, [param_data[1]], [timestamp])

        # fetch data from Voltmeters
        for name in self.voltage_monitor_names:
            instrument = self.instruments[name]

            state = instrument.query_state()
            if state in ['measuring', 'data_available']:
                try:
                    instrument.wait_until_data_available()
                except Exception as e:
                    msg = f'Caught waiting for {name} : {e}'
                    raise type(e)(msg) from e
                out_data = instrument.fetch_data()
                timestamps = out_data['timestamp']
                voltages = out_data['Voltage (V)']
                column = self.parameters['instruments'][name]['output_column']
                self.record.stage_update(column, voltages, timestamps)

        # fetch data from commercial power meters
        for name in self.commercial_power_meter_names:
            instrument = self.instruments[name]
            state = instrument.query_state()
            if state in ['measuring', 'data_available']:
                # print(name, 'in _fetch_data', instrument.query_state())
                instrument.wait_until_data_available()
                out_data = instrument.fetch_data()
                timestamps = out_data['timestamp']
                powers = out_data['Power (W)']
                column = self.parameters['instruments'][name]['output_column']
                self.record.stage_update(column, powers, timestamps)
                # print(name, 'in _fetch_data', instrument.query_state())

        for name in self.thermometer_monitor_names:
            instrument = self.instruments[name]
            state = instrument.query_state()
            if state in ['measuring', 'data_available']:
                # print(name, 'in _fetch_data', instrument.query_state())
                try:
                    instrument.wait_until_data_available()
                except Exception as e:
                    msg = f'Caught waiting for {name} : {e}'
                    raise type(e)(msg) from e
                out_data = instrument.fetch_data()

                voltage = out_data['Voltage (V)']
                current = out_data['Current (A)']
                voltage_column = self.parameters['instruments'][name][
                    'voltage_output_column'
                ]
                current_column = self.parameters['instruments'][name][
                    'current_output_column'
                ]
                if all_power_data:
                    timestamps = out_data['timestamp']
                    timestamps = timestamps - timestamps[0] + self.record['timestamp']
                    self.record.stage_update(voltage_column, voltage, timestamps)
                    self.record.stage_update(current_column, current, timestamps)

                else:
                    timestamp = self.record['timestamp']
                    self.record.stage_update(
                        voltage_column, [voltage[-1]], [timestamps[-1]]
                    )
                    self.record.stage_update(
                        current_column, [current[-1]], [timestamps[-1]]
                    )

    def _end_of_iteration(self):
        """
        Called by iterate at the end of each iteration

        Returns
        -------
        None.

        """
        # determine if ready to advance
        self._batch_arm()
        self._batch_trigger()
        self.update_statistics()
        self.advance_if_ready()
        self.index += 1
        self.record.stage_update('step_counter', [self.index], [self.record.get_time()])

    def _batch_arm(self):
        """
        Call arm() on voltage monitors and power meter.

        Returns
        -------
        None.

        """

        # arm Voltmeters
        for instrument in self.voltage_monitors:
            enable = True
            try:
                enable = instrument.setup_settings['enable']

            except KeyError:
                pass
            if enable:
                instrument.arm()

        if self.source_type == 'VNA':
            enable = True
            try:
                enable = self.source.setup_settings['enable']

            except KeyError:
                pass

            if enable:
                self.source.arm()

        if self.power_meter is not None:
            self.power_meter.arm()

        for pm in self.commercial_power_meters:
            enable = True
            try:
                enable = pm.setup_settings['enable']

            except KeyError:
                pass
            pm.arm()

        for thermometer in self.thermometer_monitors:
            enable = True
            try:
                enable = thermometer.setup_settings['enable']
            except KeyError:
                pass
            thermometer.arm()

    def _batch_trigger(self):
        """
        Call trigger() on voltage monitors and power meter.

        Parameters
        ----------
        sub_trigger_instrmanagers: bool,
            If True, runs the trigger for instrument
            managers that need to do individually trigger
            their component instruments at this time.

        Returns
        -------
        None.

        """
        # tigger Voltmeters

        # VNA goes first because power levelling fails if volmeters read
        # value prior to VNA turn on

        # if it is a VNA source
        if self.source_type == 'VNA':
            enable = True
            try:
                enable = self.source.setup_settings['enable']

            except KeyError:
                pass

            if enable:
                self.source.trigger()

        for instrument in self.voltage_monitors:
            enable = True

            try:
                enable = instrument.setup_settings['enable']

            except KeyError:
                pass

            if enable:
                instrument.trigger()

        # if there is an SMU power meter
        if self.power_meter is not None:
            self.power_meter.trigger()

        #
        for pm in self.commercial_power_meters:
            pm.trigger()

        for thermometer in self.thermometer_monitors:
            thermometer.trigger()

    def iterate(self):
        """
        Poll instruments, record data, advance to next measurement if ready.

        Returns
        -------
        None.

        """
        self._fetch_data(all_power_data=False)
        self.record.batch_update()

        #######################################################################
        # RF power mangagement
        #######################################################################

        # if in pre-measurement initial wait, we don't need power levelling
        if self.parameters.index is None:
            self._end_of_iteration()
            return

        # now that we know we are not in initial wait, check the run settings
        # to figure out what we supposed to be doing.

        # estimate the metered power for each sensor
        # if we've made it this far, there should be measurements
        # in the record of our power sensors
        self._calculate_signals()

        Frequency_GHz = self.parameters['Frequency_GHz']
        Initial_source_power_dBm = self.parameters['Initial_source_power_dBm']
        # - self.record["target_adjustment"]
        Target_power_dBm = self.parameters['Target_source_power_dBm']

        Target_power_mW = 0
        if Target_power_dBm is not None:
            Target_power_mW = 10.0 ** (Target_power_dBm / 10.0)

        Source_power_limit_dBm = self.parameters['Source_power_limit_dBm']

        # First, figure out if RF power should be on. Frequency > 0 means we
        # want to apply RF power. If we do not want RF power, we are done.
        if Frequency_GHz is None or not Frequency_GHz > 0:
            # if we are here, it means that we are at a zero power point.
            # some instruments may need to zero out while there is no
            # RF power incoming. Now is the chance

            self._end_of_iteration()
            return

        # now, we know we want to have power on. so, figure out what the
        # settings currently are
        rf_power_setting = self.record['rf_power_setting']
        power_on = self.record['power_on']

        # if power is not on, all we have to do this iteration is turn it on.
        if not power_on:
            rf_power_setting = Initial_source_power_dBm

            # NOTE: validating Initial_source_power_dBm should happen when the
            # measurement starts

            self.change_source_state(
                source_on=True, dBm=rf_power_setting, f_GHz=Frequency_GHz
            )
            # fast measurements (like for powertables) sometimes need
            # to add a delay here so everything has time to respond
            # to the change in signal.
            try:
                time.sleep(self.parameters['levelling_settings']['on_trigger_delay'])
            except KeyError:
                print(
                    "Cant find 'on trigger delay' in 'levelling_settings'. Defaulting to 2 seconds."
                )
                time.sleep(0)
            self._end_of_iteration()
            return

        # Update power levelling.
        use_GPIB_levelling = self.parameters['levelling_settings']['use_GPIB_levelling']
        GPIB_levelling_time = self.parameters['levelling_settings'][
            'GPIB_levelling_time'
        ]

        use_AM_levelling = self.parameters['levelling_settings']['use_AM_levelling']
        AM_levelling_time = self.parameters['levelling_settings']['AM_levelling_time']

        # at this point the power is on, and has just been turned on
        # At this point, we have established that power should be on and power
        # is on. So, we just need to adjust the power. First, determine how
        # much power we are sourcing by reading the Voltmeter. We have
        # established at this point that V_off_slow has a reasonable value.
        # Otherwise, update_statistics would not have let us advance
        too_little_power = {}
        power_mW = {}
        power_dBm = {}
        level_to = self.parameters['levelling_settings']['level_to']
        for port_name in SENSOR_PORTS:
            metered_power_column = format_pmeter_est_column(port_name)
            power_mW[port_name] = 1000.0 * self.record[metered_power_column]
            power_dBm[port_name] = 10.0 * np.log10(power_mW[port_name])

            # check if DC substituted power in range
            # too much power can damage equipment
            # too little power means power levelling may not be predictable
            too_much_power = power_dBm[port_name] > HARD_MAX_dBm[port_name]
            too_little_power[port_name] = (
                power_dBm[port_name] < HARD_MIN_POWER_LEVELLING
            )
            # during a special run, we do not expect the mount to respond to
            # applied RF power. So it is ok if the DC substitued power is
            # nan because that is what it's assigned druing the estaimte
            # power function. In that case, do nothing.

            device_name = self.parameters['measurement_description'][
                port_name.replace('_power', '_name')
            ]
            power_is_invalid = (
                np.isnan(power_dBm[port_name])
                or not np.can_cast(power_dBm[port_name], float)
            ) and (device_name not in SPECIAL_MOUNTS)

            # too much power can damage equipment
            if too_much_power:
                # turns off source, shut down experiment
                self.final_cleanup()
                raise (
                    Exception(
                        f'Measurement aborted. Metered power {port_name} was too high {
                            round(power_dBm[port_name], 3)
                        } dBm > {HARD_MAX_dBm[port_name]} dBm '
                    )
                )

            if power_is_invalid and (use_GPIB_levelling or use_AM_levelling):
                self.final_cleanup()
                raise (
                    Exception(
                        f'Measurement aborted. Metered power {
                            port_name
                        } not a valid number:  {power_dBm[port_name]} dBm, {
                            power_mW[port_name]
                        } mW  for use of power levelling'
                    )
                )

        # if the levelling sensor power is too small, end the iteration.
        # I think it will be trivial to add a setting that lets you set
        # which sensor is being levelled, but will add that later.
        if too_little_power[level_to]:
            self._end_of_iteration()
            return

        point_start_time = self.record['point_start_time']
        current_time = self.record['timestamp']
        elapsed_time = current_time - point_start_time

        if elapsed_time <= GPIB_levelling_time and use_GPIB_levelling:
            GPIB_levelling_C = self.parameters['levelling_settings']['GPIB_levelling_C']
            max_source_power_change_dB = self.parameters['levelling_settings'][
                'max_source_power_change_dB'
            ]

            # so that if something weird happens, new_power will have a safe value
            new_power = Initial_source_power_dBm
            power_change = (Target_power_dBm - power_dBm[level_to]) * GPIB_levelling_C

            if power_change > max_source_power_change_dB:
                power_change = max_source_power_change_dB

            if power_change < -max_source_power_change_dB:
                power_change = -max_source_power_change_dB

            if rf_power_setting + power_change < Source_power_limit_dBm:
                new_power = rf_power_setting + power_change

            elif rf_power_setting + power_change >= Source_power_limit_dBm:
                new_power = Source_power_limit_dBm

            self.change_source_state(dBm=new_power)
            self._end_of_iteration()
            return

        if (
            elapsed_time > GPIB_levelling_time
            and elapsed_time < GPIB_levelling_time + AM_levelling_time
            and use_AM_levelling
        ):
            AM_voltage = self.record['AM_voltage']
            AM_levelling_C = self.parameters['levelling_settings']['AM_levelling_C']

            new_AM_voltage = (
                AM_voltage
                + AM_levelling_C
                * (Target_power_mW - power_mW[level_to])
                / power_mW[level_to]
            )

            AM_max = HARD_AM_MAX
            if new_AM_voltage > AM_max:
                new_AM_voltage = AM_max

            if new_AM_voltage < -AM_max:
                new_AM_voltage = -AM_max

            self.record['AM_voltage'] = new_AM_voltage

            self.output_AM_voltage(AM_voltage)

        self._end_of_iteration()
        return

    def output_AM_voltage(self, new_AM_voltage):
        # print(new_AM_voltage)
        self.rf_amplitude_adjuster.setup(source_level=new_AM_voltage)
        pass

    def _load_instrument_settings(self, mode: str):
        """
        Load a settings group for all instruments.

        Parameters
        ----------
        mode : str
            A setting group in the config file
            (initial_settings, monitor_mode_settings, fast_off_mode_settings)

        Returns
        -------
        None.

        """
        names = self.parameters['instruments']['names']
        for name in names:
            try:
                settings = self.parameters['instruments'][name][mode]

            except KeyError:
                continue

            if name == self.source_name:
                self.change_source_state(**dict(settings))

            else:
                # putting this here as a conditional breakpoint
                if self.instruments[name].query_state() != 'unarmed':
                    # print(name, ' in ', mode, ' _load_instr', self.instruments[name].query_state())
                    # print(dict(settings))
                    pass
                self.instruments[name].setup(**dict(settings))

    def advance_if_ready(self):
        """
        Advance to the next row in the run_settings file.

        Returns
        -------
        None.

        """
        if self.record['stable_samples'] < 1:
            return

        # do a fast off measurement if RF power was on
        power_was_on = self.record['power_on']

        # first finish up any measurements that might still be ongoing
        self._fetch_data(all_power_data=False)

        if power_was_on:
            # Do a fast off measurement
            self._load_instrument_settings('fast_off_mode_settings')
            self._batch_arm()
            self._batch_trigger()

            # sleep to ensure that the Voltmeters capture the turn off
            try:
                time.sleep(self.parameters['levelling_settings']['off_trigger_delay'])
            except KeyError:
                print(
                    "Cant find 'off trigger delay' in 'levelling_settings'. Defaulting to 2 seconds."
                )
                time.sleep(2)
            self.change_source_state(source_on=False)
            # sleep to ensure sensor catches the turn off
            try:
                self._fetch_data(all_power_data=True)
            except Exception as e:
                msg = f'Caught on fast off :{e}'
                raise type(e)(msg) from e
        else:
            # this means we are at the end of a no power on row in settings file
            # this is a good point to zero out the commercial power meters
            for name in self.commercial_power_meter_names:
                ...
                # I think we don't need to do this
                # self.instruments[name].setup(zero_once=True)

        # advance
        self.parameters.advance()
        self.done = self.parameters.complete()
        self.record['point_counter'] = self.parameters.index
        if not self.done:
            # put instruments into monitor mode
            self.start_monitor_mode()

        # the runner class is now a context manager, so this gets called
        # automatically once the "with" loop exits
        else:
            self.final_cleanup()

    def start_monitor_mode(self):
        """
        Load monitor mode settings.

        Returns
        -------
        None.

        """
        self._load_instrument_settings('monitor_mode_settings')
        self.record['point_start_time'] = time.time() - self.record.time_zero
        # self.record["rf_power_setting"] = None
        self.record['stable_samples'] = 0
        self.record['AM_voltage'] = 0
        # reset AM to 0
        use_AM_levelling = self.parameters['levelling_settings']['use_AM_levelling']
        if use_AM_levelling:
            self.output_AM_voltage(0)
        # self.record["compression_estimate"] = None
        # self.record["loss_estimate"] = None
        # self.record["target_adjustment"] = 0

    def change_source_state(
        self,
        source_on: bool = None,
        dBm: float = None,
        f_GHz: float = None,
        **extra_kwargs,
    ):
        """
        Changes source settings while updating record appropriately

        Parameters
        ----------
        power_on : bool, optional
            If True, turn power on. If False, turn power off. If None,
            do nothing. The default is None.

        dBm : float, optional
            Power level in dBm. If None, do nothing. The default is None.


        **extra_kwargs : dict
            Other settings for the source. These are passed to the source
            without generating any record.

        Returns
        -------
        None.

        """
        source = self.instruments[self.source_name]
        update_settings = {}

        for key, value in extra_kwargs.items():
            update_settings[key] = value

        if dBm is not None:
            self.record['rf_power_setting'] = dBm
            update_settings['dBm'] = dBm

        if f_GHz is not None:
            update_settings['f_GHz'] = f_GHz
            self.record['frequency'] = f_GHz

        if source_on is not None:
            self.record['power_on'] = source_on
            update_settings['source_on'] = source_on

        source.setup(**update_settings)

    def final_cleanup(self):
        """
        Put instruments in safe state, write data.

        Returns
        -------
        None.

        """
        # Turn off power meter, if we are exiting because of an error
        # if we are not exiting because of an error, we probably want it to
        # stay on so that we don't need to wait for the calorimeter to warm up.
        if not self.closed:
            # close the console output
            if CONSOLE_MANAGER.fio is not None:
                CONSOLE_MANAGER.fio.close()
                # try to copy the console log to the output directory
                print('Copying console log to target directory')
                try:
                    with open(self.console_log_file, 'r') as fin:
                        with open(
                            Path(self.output_dir) / 'console-log.txt', 'w'
                        ) as fout:
                            for line in fin:
                                fout.write(line)
                except Exception as e:
                    print('failed to copy console log : ', str(e))

            # clean up data outputs
            self.record.batch_update()
            self.record.output(write_recent_data=True)

            if (
                self.parameters['measurement_description']['DC_source_type']
                in SMU_POWER_METERS
                and not self.done
            ):
                power_meter = self.instruments[self.power_meter_name]
                power_meter.setup(source='off')

            self.done = True
            try:
                self.change_source_state(source_on=False)
            except KeyError as e:
                print(f'KeyError on source shutdown, likely not initialized : {e}.')

            # turn off anything else.
            for k, v in self.instruments.items():
                print('shutting down ', k)
                try:
                    v.close()
                except Exception as e:
                    print(f'Failed to shudown {k} - caught : {e}')
            self.closed = True
