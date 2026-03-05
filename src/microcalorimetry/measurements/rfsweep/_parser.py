"""
The parser module contains tools to extract data from microcalorimeter runs.

These tools are built to accommodate several types of complexity.
* Microcalorimetry data has several levels (Campaign, Run, Segment, Step).
* Several different types of measurements that happen during microcalorimeter runs.
* Several different file formats have been used over the years.

SignalAnalyzer
------------------
The responsibilities of the SignalAnalyzer class are:
* Hold data required to analyze a set of related data columns.
* Analyze segments
* Analyze steps
* Specific measurements (for example, thermopile monitor) are child classes.

Campaign
--------
A series of runs that are related (ex. same device and connect cycle)
The responsibilities of the Campaign class are:
* Call analysis functions for all of the runs.
* Collect results of the analysis into summary reports for further analysis.

Run
---
A period of time when the measurement ran without interruptions.
The responsibilities of the Run class are:
* Read file(s).
* Initialize SignalAnalyzers
* Accommodate the variety of data formats by extending the Run class.

Segment
-------
Two long periods of RF off with several frequency point measurements in the middle.
The responsibilities of the segment class are:
* Parse the segment into step
* Perform timeseries analysis on segments
* Store results associated with segments
* Save as legacy timeseries
* None of the logic at the segment level depends on the run data format.

Step
----
a period when the RF source was on, and its frequency was constant.
Periods of RF off on before and after are included
The responsibilities of the step class are:
* Perform timeseries analysis on steps
* Store results associated with steps
* None of the logic at the segment level depends on the run data format.
"""

from matplotlib.lines import segment_hits
from rminstr.data_structures import ExistingRecord, LegacyRecord, ExptParameters
import os.path
import xarray as xr
from rmellipse.uobjects import RMEMeas
import pylab as pl
import rminstr_specs.K2450 as k2450
import rminstr_specs.HP3458A as HP3458A
import rminstr_specs.HP34420A as HP34420A
from os.path import join, basename, dirname
import abc
import collections
import numpy as np
import pandas as pd
import copy
import warnings

# import pathlib
import pytz

# import builtins
# import pylab as pl
# import time
import scipy
from sphinx.builders.gettext import timestamp

# from scipy.optimize import curve_fit
local_tz = pytz.timezone('US/Mountain')

# from measurements.microcalorimeter.runner import MicrocalorimeterRunner
# instrument specifications
# indexed by model, role
INSTRUMENT_SPECS = {
    'HP34420A': {
        'bias_monitor': {'vdc': HP34420A.DatasheetDCV},
        'thermopile_monitor': {'e': HP34420A.DatasheetDCV},
        'voltage_monitor': {'vdc': HP34420A.DatasheetDCV},
    },
    'HP3458A': {
        'bias_monitor': {'vdc': HP3458A.DatasheetDCV},
        'thermopile_monitor': {'e': HP3458A.DatasheetDCV},
        'voltage_monitor': {'vdc': HP3458A.DatasheetDCV},
    },
    'K2450': {
        'SMU_power_meter': {
            'idc': k2450.DatasheetMeasureDCI,
            'vdc': k2450.DatasheetMeasureDCV,
        }
    },
}

CROWLEY_DEFAULT_CONFIG = {
    'stats_window': 9,  # vfluc.m line 66
    'instruments': {
        'DVM1': {
            'model': 'HP3458A',
            'serial': 'unknown',
            'role': 'bias_monitor',
            'output_column': 'DVM_volts',
            'initial_settings': {'v_range': 10},
            'monitor_mode_settings': {
                'nplc': 100,  # TODO: check
                'timer': 0,
                'timeout': 5,
            },
            'fast_off_mode': {'nplc': 100, 'timer': 0, 'timeout': 5},
        },
        'NVM1': {
            'model': 'HP34420A',
            'serial': 'unknown',
            'role': 'thermopile_monitor',
            'initial_settings': {'v_range': 0.002, 'nplc': 100, 'num_readings': 1},
            'monitor_mode_settings': {'timeout': 5, 'enable': True},
            'fast_off_mode_settings': {'enable': False},
        },
    },
    'analysis_config': {
        'DUT_power': {
            'instr_timing_tolerance': 5,
            'V_off_delay': 8,
            'V_off_function': 'single_sample',
            'V_off_fit_time_window': (-1, 10),
            'RF_on_average_window': 300,
        },
        'calorimeter_power': {'instr_timing_tolerance': 5, 'RF_on_average_window': 300},
    },
    'signal_config': {
        'DUT_power': {
            'type': 'bolometer',
            'resistance': 200,
            'input_signals': ['vdc'],
            'vdc': {'units': 'V', 'column': 'DVM_volts', 'instrument': 'DVM1'},
        },
        'calorimeter_power': {
            'type': 'thermoelectric',
            'input_signals': ['e'],
            'e': {'units': 'V', 'column': 'NVM_volts', 'instrument': 'NVM1'},
        },
    },
}

LEGACY_DEFAULT_CONFIG = {}  # TODO: add this.

DEFAULT_KEYS = [
    'instruments',
    'analysis_config',
    'signal_config',
    'levelling_settings',
    'measurement_description',
    'stats_settings',
    'stats_window',
]

VERBOSE = True
DEFAULT_MAXLEN = 10000
DEFAULT_MINLEN = 10000

PLOT_COLORS = pl.rcParams['axes.prop_cycle'].by_key()['color']

MAIN_TRACE_LINEWIDTH = 1
STABLE_TRACE_LINEWIDTH = 1
FIT_TRACE_LINEWIDTH = 1

MAIN_TRACE_COLOR = 'blue'
FIT_TRACE_COLOR = 'black'

INITIAL_STABLE_TRACE_COLOR = 'black'
FINAL_STABLE_TRACE_COLOR = 'black'
MIDDLE_STABLE_TRACE_COLOR = 'black'

RESAMPLE_POINTS_SIZE = 4
FIT_POINT_SIZE = 6


def _lin_plus_exp(t, a, b, vscale, tscale):
    return a * t + b + vscale * np.exp(-t / tscale)


def _linear(t, a, b):
    return a * t + b


def _merge_dicts(source_dicts: list[dict]):
    """
    Combine two dictionaries.

    Parameters
    ----------
    source_dicts : list[dict]
        Dictionaries later in the list have higher priority

    Returns
    -------
    None.

    """
    results = {}
    done = False
    key_set = set()
    for source_dict in source_dicts:
        try:
            key_set.update(source_dict.keys())
            results = {}
            done = False

        # It is possible that an item in source_dicts is not a dict.
        # In that case, we have reached the end of recursion.
        except AttributeError:
            results = source_dict
            done = True

    # if source_dicts contains a mixture of dictionaries and values,
    # what we do here depends on the ordering of dicts in source_dicts

    if not done:
        for key in key_set:
            sub_dicts = []
            for source_dict in source_dicts:
                try:
                    sub_dicts.append(source_dict[key])

                except KeyError:
                    pass

                # if we have reached something that isn't a
                except TypeError:
                    sub_dicts.append(source_dict)

            if sub_dicts:
                results[key] = _merge_dicts(sub_dicts)

    return results


class Campaign:
    """
    The Campaign class stores a list of related runs (same connect, device).

    Usage: Initialize run objects first, then add them to a campaign.
    """

    def __init__(self, run_list, output_dir, campaign_name: str = None):
        # parser status data
        self.run_list = run_list
        self.output_dir = output_dir
        self.campaign_name = campaign_name

    def _generate_steps(self):
        """Generate steps for output_RMEmeas."""
        for i, run_i in enumerate(self.run_list):
            for j, segment_j in enumerate(run_i.segments):
                for s, step_s in enumerate(segment_j.steps):
                    # print("run:", i, "segment:", j, "step:", s)
                    yield step_s

    def analyze(self):
        """
        Analyze all runs.

        Returns
        -------
        None.

        """
        for run in self.run_list:
            run.analyze()

    def output_text_segments(self):
        """
        Output text files that can be read by Tom Crowley's matlab scripts.

        Returns
        -------
        None.

        """
        for i, run_i in enumerate(self.run_list):
            for j, segment_j in enumerate(run_i.segments):
                segment_j.write_file(self.output_dir)

    def output_dataframe(self) -> pd.DataFrame:
        """
        Outputs data in a columnated dataframe format for pandas.

        Dataframe is indexed by segment number and row number in a
        multilevel index.

        Removes incomplete sets.
        Returns
        -------
            pd.DataFrame

        """
        output = {'frequency':[]}
        for i, step in enumerate(self._generate_steps()):
            output['frequency'].append(step.frequency)
            for column, val in step.results.items():
                try:
                    output[column].append(val)
                except KeyError:
                    output[column] = [val]
        output = pd.DataFrame(output)
        output = output[output.complete]
        output = output.drop(columns = 'complete')
        output = output.astype(float)

        return output

    def output_RMEmeas(
        self, include_time_std: bool = False, include_specs: bool = True
    )->RMEMeas:
        """
        Output the measurement records in RMEmeas format.

        Parameters
        ----------
        include_time_std : bool, optional
            If True, includes the timeseries standard deviation of ON/OFF values
            as an uncertainty mechanism. An overestimation of uncertainty that
            can be used as a substitute for repeated measurements if not possible
            to do so. Default is False. The default is False.

        include_specs : bool, optional
            If True, includes the instrument specifications
            as uncertainty mechanisms. The default is True.

        Returns
        -------
        None.

        """

        def col_is_perturbation(col):
            return_value = False
            if '_dev' in col or '_spec' in col:
                return_value = True
            return return_value

        def get_umech_id(col, index, step):
            if '_spec' in col:
                # pert_name = cdev + '_step' + str(s)
                instr = step.metadata[col]['instrument_name']
                umech_id = col + ':' + instr + ':spec:' + campaign_name
            elif '_dev' in col:
                umech_id = col + '_step_' + str(index)
            return umech_id

        def get_associated_column(col):
            if '_spec' in col:
                associated_col = col.replace('_spec', '')
            if '_dev' in col:
                associated_col = col.replace('_dev', '')
            return associated_col

        # make a nominal array
        campaign_name = self.campaign_name
        if self.campaign_name is None:
            campaign_name = basename(self.run_list[-1].name).split('.')[0]

        # count frequencies, columns, uncertainty mechanisms
        columns = set()
        umech_ids = set()
        frequencies = []
        for i, step in enumerate(self._generate_steps()):
            f = step.frequency
            frequencies.append(f)
            columns_for_step = [
                col for col in list(step.results.keys()) if not col_is_perturbation(col)
            ]
            umech_ids_for_step = [
                get_umech_id(col, i, step)
                for col in list(step.results.keys())
                if col_is_perturbation(col)
            ]
            columns.update(columns_for_step)
            umech_ids.update(umech_ids_for_step)

        # initialize nominal array
        columns = list(columns)
        shape = (len(frequencies), len(columns))
        dims = ('frequency', 'col')
        coords = {'frequency': frequencies, 'col': columns}
        vals = np.zeros(shape)
        nom = xr.DataArray(vals, coords, dims)

        # Assign nominal values
        for i, step in enumerate(self._generate_steps()):
            f = step.frequency
            for col in step.results.keys():
                val = step.results[col]
                if not col_is_perturbation(col):
                    nom.sel(col=col).data[i] = val

        # initialize meas object from nominal
        out = RMEMeas.from_nom(name='calorimetercampaign', nom=nom)

        # assign independent uncertanity mechanisms
        a_cats = {
            'Type': 'A',
            'Origin': 'Cal Run Noise',
            'Experiment': 'Calorimeter Run',
            'Instrument': 'Calorimeter',
        }
        b_cats = {
            'Type': 'B',
            'Origin': 'Cal Run DC Traceability',
            'Experiment': 'Calorimeter Run',
        }

        # add uncertainty mechanisms

        for i, step in enumerate(self._generate_steps()):
            f = step.frequency
            pert_cols = [col for col in step.results.keys() if col_is_perturbation(col)]
            for col in pert_cols:
                val = step.results[col]
                umech_id = get_umech_id(col, i, step)
                associated_col = get_associated_column(col)
                pert = nom.copy()
                pert.sel(col=associated_col).data[i] += val

                # Time series standard deviaion uncertainty
                if 'dev' in umech_id:
                    if include_time_std:
                        out.add_umech(umech_id, pert, category=a_cats)

                # uncertainty associated with machine specifications
                elif 'spec' in umech_id:
                    out.add_umech(
                        umech_id,
                        pert,
                        category=dict(**b_cats, Instrument=umech_id.split(':')[1]),
                    )

                else:
                    raise Exception(
                        f'Uncertainty {umech_id} doesnt have a defined category for RF sweeps.'
                    )

        return out

    def output_segments(
        self,
        fmt_for: str = 'matlab',
        include_time_std: bool = False,
        include_specs: bool = True,
    ):
        """
        Output the segments to files.

        Parameters
        ----------
        fmt_for : str, optional
            Specificies output format. The options are 'matlab', 'rmellipse'.
            The default is 'matlab'.

        include_time_std : bool, optional
            Used if fmt_for = rmellipse. If True, includes the timeseries standard deviation of ON/OFF values
            as an uncertainty mechanism. An overestimation of uncertainy that
            can be used as a substitute for repeated measurments if not possible
            to do so. Default is False. The default is False.

        include_specs : bool, optional
            Used if fmt_for = rmellipse. If True, includes the instrument specifications
            as uncertainty mechanisms. The default is True.

        Returns
        -------
        None.

        """
        # TODO: deprecate old behavior, as promised.
        # (DeprecationWarning("The output_segments method will be deprecated in version 0.4.0. Please use output_text_segments or output_RMEmeas methods instead."))
        if fmt_for == 'matlab':
            # TODO: make this do something
            # self.output_matlab()
            return None

        elif fmt_for == 'rmellipse':
            return self.output_RMEmeas(
                include_time_std=include_time_std, include_specs=include_specs
            )

        elif fmt_for == 'pandas':
            return self.output_dataframe()

        else:
            raise Exception('fmt_for ' + fmt_for + ' not recognized.')


class Run(abc.ABC):
    """
    A run object represents a single measurement run.

    This class is completely agnostic to the way that the data are organized,
    except that the data files can be read by a data_record object and that
    certain columns exist.
    """

    def __init__(self, name, **extra_config):
        """
        Initialize a run object.

        Parameters
        ----------
        name : str
            Human-readable name.

        Returns
        -------
        None.

        """
        # parser status data
        self.name = name
        # current data set
        self.parsed_config = None
        self.data = None

        # because different run settings files have different column names
        self.frequency_setting_name = 'Frequency_GHz'
        self.results = {}  # stores results of calculations performed on this run
        self.analyzers = {}
        self.parsed_config = {}
        self.segments = []
        self.extra_config = extra_config

    @abc.abstractmethod
    def _get_frequencies(self):
        """Get frequency list for run. In New Type Runs and Legacy Runs,
        the frequency list corresponds to the frequencies in the settings file.
        """
        pass

    @abc.abstractmethod
    def _get_measurement_description(self, key: str):
        """Get information from measurement description.
        The measurement description traditionally contains:
        lead_resistance, connector_type, mount_name, connect_number, calorimeter, notes, resistance.
        """
        pass

    # Do not extend or go around this function.
    def initialize_analyzers(self):
        """
        Initialize instrument analyzers.

        Returns
        -------
        None.

        """
        signal_classes = {
            'special': None,
            'commercial': PowerMeterAnalyzer,
            'thermoelectric': ThermoelectricAnalyzer,
            'bolometer': BolometerAnalyzer,
            'rf_source': None,
        }

        self.analyzers = {}
        signals = self.parsed_config['signal_config']
        for signal in self.parsed_config['analysis_config']:
            signal_config = self.parsed_config['signal_config'][signal]
            analysis_config = self.parsed_config['analysis_config'][signal]
            analysis_config['time_zero'] = self.results['time_zero']
            input_signal_names = self.parsed_config['signal_config'][signal][
                'input_signals'
            ]

            input_signal_config = {}
            instruments = {}
            for key in signal_config.keys():
                if key in input_signal_names:
                    instrument_name = self.parsed_config['signal_config'][signal][key][
                        'instrument'
                    ]
                    input_signal_config[key] = self.parsed_config['signal_config'][
                        signal
                    ][key]
                    try:
                        instruments[instrument_name] = self.parsed_config[
                            'instruments'
                        ][instrument_name]
                    except KeyError as e:
                        msg = f'instrument name {instrument_name} in signal_config.{signal} not one of {self.parsed_config["instruments"].keys()}'
                        raise KeyError(msg)

            signal_type = self.parsed_config['signal_config'][signal]['type']

            signal_class = signal_classes[signal_type]
            if signal_class is None:
                raise ValueError(
                    f'Signal type {signal_type} assigned to {signal} has no defined SignalAnalyzer.'
                )
            analyzer = signal_class(
                analysis_config, signal_config, input_signal_config, instruments
            )
            self.analyzers[signal] = analyzer

    @abc.abstractmethod
    def load(self):
        """
        Load data into memory.

        Returns
        -------
        None.

        """
        pass

    # Do not extend or go around this function.
    # All run analysis should pass through here.
    def analyze(self):
        """
        Process data from a run.

        Reads current data file, identifies segments, builds segment objects,
        and appends them to self.segments.

        Returns
        -------
        None.

        """
        # initialize analyzers
        self.initialize_analyzers()
        _check_for_meta = [
            'lead_resistance',
            'connector_type',
            'mount_name',
            'connect_number',
            'calorimeter',
            'resistance',
            'notes',
        ]
        for _meta in _check_for_meta:
            try:
                self.results[_meta] = self._get_measurement_description(_meta)
            except KeyError as e:
                self.results[_meta] = 'NA'
                warnings.warn(
                    f'{_meta} not found in measurement description.', stacklevel=3
                )

        # fix a specific bug associated with pandas parsing
        if self.results['calorimeter'] == 'nan':
            self.results['calorimeter'] = 'NA'

        # segments are determined by the frequency list
        frequencies = self._get_frequencies()
        numf = len(frequencies)
        segment_id = np.zeros(numf, dtype='int64')

        # segmement boundaries are marked with 0s.
        # the tricky thing about segments is that adjacent segments share 0's.
        # start by assigning non-zero frequency points to segments
        segment_counter = 0
        last_f = None
        current_f = None
        for i in range(0, numf):
            last_f = current_f
            current_f = frequencies[i]

            if current_f > 0 and last_f == 0:
                segment_counter += 1

            if current_f == 0:
                segment_id[i] = 0

            else:
                segment_id[i] = segment_counter

            if VERBOSE:
                print(i, current_f, segment_id[i])

        # now that segment numbers have been assigned, iterate through data.
        raw_segment_data = []

        # determine what columns are necessary

        for i in range(segment_counter):
            raw_segment_data.append({})
            for column in self.data.columns:
                # use deque because adding to long lists is slow
                raw_segment_data[i][column] = collections.deque()
                raw_segment_data[i][column + '_timestamp'] = collections.deque()

        def seg_append(index, column):
            raw_segment_data[index][column].append(self.data[column])

            try:
                timestamps = self.data.timestamps[column]
                if len(timestamps) > 0:
                    raw_segment_data[index][column + '_timestamp'].append(
                        timestamps[-1]
                    )

            except (KeyError, AttributeError):
                raw_segment_data[index][column + '_timestamp'].append(
                    self.data['timestamp']
                )

        # now that segment numbers have been assigned, iterate through data.
        highest_segment_id = 0
        while self.data.read_next_line():
            try:
                current_index = int(self.data['point_counter'])

            except TypeError:  # is nan
                continue  # the first segment has not started yet

            except ValueError:  # is None
                continue

            if current_index > len(segment_id) - 1:
                current_index = len(segment_id) - 1

            current_segment_id = segment_id[current_index]

            # try: # if possible, check that the segment id was determined correctly
            #     check_segment_id = self.data.segment_counter
            #     if check_segment_id != current_segment_id and self.data["frequency"] > 0:
            #         print("Parsed segment index incorrectly. Should be {}, was {}. Index: {}. Current index: {}.".format(check_segment_id, current_segment_id, self.data.index, current_index))
            #
            # except KeyError:
            #     pass

            for column in self.data.columns:
                if current_segment_id > highest_segment_id:
                    highest_segment_id = current_segment_id

                if current_segment_id > 0:
                    seg_append(current_segment_id - 1, column)

                # handle 0's correctly
                if current_segment_id == 0 and highest_segment_id == 0:
                    seg_append(0, column)

                # handle the ending correctly
                if current_segment_id == 0 and highest_segment_id == segment_counter:
                    seg_append(segment_counter - 1, column)

                if (
                    current_segment_id == 0
                    and highest_segment_id < segment_counter
                    and highest_segment_id > 0
                ):
                    seg_append(highest_segment_id - 1, column)
                    seg_append(highest_segment_id, column)

        # what was the status at the end of the run?
        segment_complete = []
        for i in range(segment_counter):
            segment_i_complete = False

            if i < highest_segment_id - 1:
                segment_i_complete = True

            if i == highest_segment_id - 1:
                # last_index_reached = parser.data["point_counter"] == numf-1
                try:
                    stats_stable_samples = self.parsed_config['stats_stable_samples']

                except KeyError:
                    stats_stable_samples = 1

                stable = stats_stable_samples == self.data['stable_samples']
                power_off = not self.data['power_on']
                segment_i_complete = stable and power_off
                # print(i, stable, power_off, segment_i_complete)

            if i > highest_segment_id - 1:
                segment_i_complete = False

            # print(i, segment_i_complete)
            segment_complete.append(segment_i_complete)

        # segments that were not reached due to interruption are
        # run_complete = last_index_reached and stable and power_off

        # cast to numpy arrays.
        # initialize segment objects
        for i in range(segment_counter):
            if segment_complete[i]:
                for column in self.data.columns:
                    raw_segment_data[i][column] = np.array(raw_segment_data[i][column])
                    raw_segment_data[i][column + '_timestamp'] = np.array(
                        raw_segment_data[i][column + '_timestamp']
                    )

                new_segment = Segment(raw_segment_data[i], segment_complete[i], self)
                self.segments.append(new_segment)
                new_segment.analyze()

    def plot_analysis(
        self, segment_index: int, signal: str, input_signal_name: str = None
    ):
        """
        Plot analysis results.

        Parameters
        ----------
        segment_index : int
            Index of segment to plot.

        signal : str
            Which signal to plot (corresponds to signal_config in config file.)

        input_signal_name : str, Optional
            Which input signal to plot (see input_signal in config file).
            Only necessary for signals with more than one input signal.
            The default is None.

        Returns
        -------
        None.

        """
        self.analyzers[signal].plot_analysis(
            self.segments[segment_index], input_signal_name
        )

    def __getitem__(self, key):
        """
        If key is a key in self.results, return associated data or if key is a key in self.raw_data, return associated data.

        Parameters
        ----------
        key : str
            Column to retrieve value from.

        Returns
        -------
        data

        """
        retval = None
        if key in self.results.keys():
            retval = self.results[key]

        if retval is None:
            raise (KeyError(key))

        return retval


class NewTypeRun(Run):
    """New_type runs are measured with newer software."""

    def __init__(self, working_folder, data_file, name, **extra_config):
        Run.__init__(self, name, **extra_config)
        self.working_folder = working_folder
        self.results = {}
        self.results['version'] = 'new type'
        self.data = None
        self.expt = None
        self.parsed_config = None
        self.config_file = None
        self.run_settings_file = None
        self.data_file = data_file

    def _get_measurement_description(self, key):
        """Get information from measurement description.
        The measurement description traditionally contains: lead_resistance, connector_type, mount_name, connect_number,
        calorimeter, notes, resistance.
        """
        measurement_description_keys = [
            'lead_resistance',
            'connector_type',
            'mount_name',
            'connect_number',
            'calorimeter',
            'notes',
        ]
        levelling_settings_keys = ['resistance']
        retval = None
        if key in measurement_description_keys:
            retval = self.parsed_config['measurement_description'][key]

        elif key in levelling_settings_keys:
            retval = self.parsed_config['levelling_settings'][key]

        if retval is None:
            raise KeyError('{} not found in measurement description'.format(key))

        return retval

    def _get_frequencies(self):
        """Get frequency list for run. In New Type Runs and Legacy Runs,
        the frequency list corresponds to the frequencies in the settings file.
        """
        frequency_setting_name = 'Frequency_GHz'
        return self.expt.get_column(self.frequency_setting_name)

    def load(self):
        """
        Load data from data file corresponding to self.run_index.

        Returns
        -------
        None.

        """
        data_file = self.data_file
        self.data = ExistingRecord(
            data_file,
            output_dir=self.working_folder,
            maxlen=DEFAULT_MAXLEN,
            minlen=DEFAULT_MINLEN,
        )

        self.config_file = self.data.metadata['config_file']
        self.run_settings_file = self.data.metadata['settings_file']

        try:
            self.expt = ExptParameters(self.config_file, self.run_settings_file)

        except FileNotFoundError:
            newdir = dirname(data_file)
            self.config_file = join(newdir, basename(self.config_file))
            self.run_settings_file = join(newdir, basename(self.run_settings_file))
            self.expt = ExptParameters(self.config_file, self.run_settings_file)

        self.results['time_zero'] = float(self.data.metadata['time_zero'])
        parsed_config = _merge_dicts([self.expt, self.extra_config])
        self.parsed_config = {
            key: parsed_config[key]
            for key in DEFAULT_KEYS
            if key in parsed_config.keys()
        }


class LegacyRun(Run):
    """The LegacyRun class describes data taken with the earlier versions of calorimeter-python."""

    def __init__(
        self,
        working_folder,
        data_file,
        config_file,
        run_settings_file,
        name,
        **extra_config,
    ):
        Run.__init__(self, name, **extra_config)
        self.working_folder = working_folder
        self.config_file = config_file
        self.run_settings_file = run_settings_file
        self.data_file = data_file
        self.parsed_config = None
        self.data = None

    def _get_frequencies(self):
        """Get frequency list for run. In New Type Runs and Legacy Runs,
        the frequency list corresponds to the frequencies in the settings file.
        """
        frequency_setting_name = 'Freq'
        return self.expt.get_column(self.frequency_setting_name)

    def load(self):
        """
        Load data.

        Returns
        -------
        None.

        """
        self.load_expt()
        self.load_data()

    def load_expt(self):
        """
        Load data from config file and run file corresponding to self.run_index.

        Returns
        -------
        None.

        """
        config_file = self.working_folder + self.config_file
        run_settings_file = self.working_folder + self.run_settings_file
        self.expt = ExptParameters(config_file, run_settings_file, header=11)
        parsed_config = _merge_dicts(
            [LEGACY_DEFAULT_CONFIG, self.expt, self.extra_config]
        )
        self.parsed_config = {
            key: parsed_config[key]
            for key in DEFAULT_KEYS
            if key in parsed_config.keys()
        }
        self.results['version'] = 'legacy'

    def load_data(self):
        """
        Load data from data file corresponding to self.run_index.

        Returns
        -------
        None.

        """
        data_file = self.data_file
        self.data = LegacyRecord(
            data_file,
            output_dir=self.working_folder,
            maxlen=DEFAULT_MAXLEN,
            minlen=DEFAULT_MINLEN,
        )

        # TODO: is this true?
        self.results['time_zero'] = float(self.data['timestamp'][0])


class _CrowleyData:
    """New Type Runs and Legacy Runs use data_records (a custom class) to store timeseries data.
    _Crowley data provides a similar interface, so that the run.analyse method will work the same
    for all three types of run.

    Crowley data is not a subclass of data_record because it only implements the bare minimum
    behaviors for run.analyze to work.
    """

    def __init__(self, segment_data):
        self.segments = segment_data
        self.index = None
        self.segment_counter = 0
        self.columns = [
            'timestamp',
            'frequency',
            'DVM_volts',
            'NVM_volts',
            'point_counter',
            'step_counter',
            'power_on',
            'stable_samples',
        ]

    def read_next_line(self):
        segment_length = len(self.segments[self.segment_counter]['timestamp'])
        num_segments = len(self.segments)
        if self.index == segment_length - 1:
            if self.segment_counter == num_segments - 1:
                return False
            else:
                self.segment_counter += 1
                self.index = 0

        if self.index is None:
            self.index = 0
        else:
            self.index += 1

        return True

    def __getitem__(self, key):
        return self.segments[self.segment_counter][key][self.index]


class CrowleyRun(Run):
    """Parser for matlab formatted for Tom Crowley's Analysis."""

    def __init__(
        self,
        working_folder: str,
        name: str,
        segment_files: list[str] = None,
        **extra_config,
    ):
        """
        Create a Crowley run for analysis.

        Crowley runs are legacy formats for calorimeter runs.

        Parameters
        ----------
        working_folder : str
            Folder working from.

        name : str
            Assign a name to the run.

        segment_files : list[str], optional
            Paths to segment files. If NONE, then it will search for any
            files in the working directory. The default is None.

        Raises
        ------
        Exception
            If segment_files = None and no segments were found in the working
            directory.
        """
        Run.__init__(self, name, **extra_config)
        self.working_folder = working_folder

        if segment_files is None:
            segment_files = [
                os.path.join(working_folder, d)
                for d in os.listdir(working_folder)
                if os.path.isfile(os.path.join(working_folder, d)) and '.' not in d
            ]
            if len(segment_files) < 1:
                raise Exception('No datafiles found.')

        self.default_config = CROWLEY_DEFAULT_CONFIG
        self.data = None
        self.results = {}
        self.results['version'] = 'crowley'

        self.segment_files = segment_files
        self.segments = []

    @staticmethod
    def _read_file(path: str):
        #  NOTES FROM MATLAB
        # % The standard file columns are time, frequency, vbias, and vpile
        # % A 2nd bias voltage is added between vbias and vpile (i.e. col 4).  When there are two
        # %    bias voltages, the one in column 3 was used to level the power in the Type 2 loop
        # %    while the one in column 4 is the other.  Data files since spring have a "calo" or
        # %    "side" in the header to indicate whether the bias voltage in column 3 represents the
        # %    calorimeter or sidearm voltage
        # % An EIP frequency is added as the last column.
        # % Both additions were done this way for historical reasons.  (i.e. we threw things
        # %    together and hoped it worked)

        with open(path) as f:
            # read header
            line1 = f.readline().split(',')
            line2 = f.readline().split(',')
            line3 = f.readline()
            header = {}
            header['segment_name'] = line1[0]
            header['connector_type'] = line1[1]
            header['mount_name'] = line1[2]
            header['connect_number'] = line1[3]
            header['min_frequency'] = line1[4]
            header['max_frequency'] = line1[5]
            header['step'] = line1[6]
            header['software'] = line2[0]
            header['version'] = line2[1]
            # header["line_2_field_3"] = line2[2]
            # header["line_2_field_4"] = line2[3]
            header['resistance'] = float(line2[4])
            # header["line_2_field_6"] = line2[5]
            header['calorimeter'] = line2[6]
            header['lead_resistance'] = float(line2[7])
            header['notes'] = line3

        # read data
        data = {}
        d = pd.read_csv(path, comment='"').to_numpy()
        # standard calorimeter run, put zeros for sidearm and eip voltage
        if d.shape[1] == 4:
            # TODO: Probably we need to multiply timestamps by 3600*24 to convert to seconds
            data['timestamp'] = d[:, 0]
            data['frequency'] = d[:, 1]
            data['DVM_volts'] = d[:, 2]
            data['NVM_volts'] = d[:, 3]

        # could be a side
        else:
            raise Exception(
                'Ability to parse historical matlab files with sidearms/EIP frequency not yet added.'
            )

        return data, header

    def _get_frequencies(self):
        return self.frequencies

    def _reformat_for_analysis(
        self, in_segments: list[dict[str, np.ndarray]]
    ) -> list[dict[str, np.ndarray]]:
        """
        Reformat column data so that run.analyze() will work.

        The analysis needs several columns that are not recorded in the Crowley-style data files:
        step_counter, point_counter, power_on, stable_samples. Here, we generate these columns and populate them
        so that the analysis will proceed correctly.

        We also modify the timestamps column so that time starts at 0 and is measured in seconds.

        Parameters
        ----------
        in_segments: list[dict[str, np.ndarray]]
            Column data from parsed files.

        Returns
        -------
        out_segment_data: list[dict[str, np.ndarray]]
            Data reformatted for analysis.
        """
        point_counter = 0
        time_zero = float('inf')
        self.frequencies = []  # assume 0 frequency at start

        out_segments = []
        for i, in_segment_i in enumerate(in_segments):
            out_segment_i = {}
            out_segments.append(out_segment_i)
            for key in ['frequency', 'DVM_volts', 'NVM_volts']:
                out_segment_i[key] = in_segment_i[key]

            time_zero_i = np.min(in_segment_i['timestamp'])
            time_zero = np.min([time_zero_i, time_zero])
            num_samples = len(in_segment_i['frequency'])
            index = np.arange(num_samples)
            out_segment_i['point_counter'] = np.zeros(num_samples)
            out_segment_i['step_counter'] = index
            out_segment_i['power_on'] = np.zeros(num_samples, dtype=np.bool)
            out_segment_i['power_on'][out_segment_i['frequency'] > 0] = True
            out_segment_i['stable_samples'] = np.zeros(num_samples, dtype=np.int64)

            # We are looking for the places where the frequency changes.
            # zero counts as a frequency. However, during a fast off,
            # there is a single samples with 0 frequency.
            # we want to exclude that frequency change.

            previous_frequency = 0.0

            # two things can be triggered by thermopile stability:
            # moving on to the next frequency, and ending the segment
            thermopile_stable_for_frequency_change = False
            thermopile_stable_for_segement_end = False

            for j, frequency_j in enumerate(in_segment_i['frequency']):
                # look forward to see if frequency has changed
                in_window = np.logical_and(
                    index >= j, index < j + self.parsed_config['stats_window']
                )
                frequencies_in_window = in_segment_i['frequency'][in_window]
                frequency_changes = (
                    frequencies_in_window[0] != frequencies_in_window[-1]
                )

                if frequency_changes and not thermopile_stable_for_frequency_change:
                    thermopile_stable_for_frequency_change = True
                    previous_frequency = frequency_j

                if (
                    j + self.parsed_config['stats_window'] == num_samples - 1
                    and not thermopile_stable_for_segement_end
                ):
                    thermopile_stable_for_segement_end = True
                    previous_frequency = frequency_j

                # if frequency change complete
                if thermopile_stable_for_frequency_change and not frequency_changes:
                    thermopile_stable_for_frequency_change = False
                    self.frequencies.append(previous_frequency)
                    point_counter += 1

                if (
                    thermopile_stable_for_segement_end
                    or thermopile_stable_for_frequency_change
                ):
                    out_segment_i['stable_samples'][j] = 1

                # if end of segment complete
                if thermopile_stable_for_segement_end and j == num_samples - 1:
                    thermopile_stable_for_segement_end = False
                    self.frequencies.append(previous_frequency)

                out_segment_i['point_counter'][j] = point_counter
            point_counter += (
                1  # always advance the step counter at the end of a segment
            )

        # convert timestamps to seconds since time 0
        for i, in_segment_i in enumerate(in_segments):
            out_segments[i]['timestamp'] = (
                (in_segment_i['timestamp'] - time_zero) * 3600 * 24
            )

        self.results['time_zero'] = time_zero
        return out_segments

    def load(self):
        """Load the data."""

        # read data, sort by time stamps, then subtract t0
        parsed_segment_data = []

        # most fields in the header are consistent across a run, but there are a few exceptions.
        measurement_description_keys = [
            'lead_resistance',
            'connector_type',
            'mount_name',
            'connect_number',
            'calorimeter',
            'notes',
        ]
        levelling_settings_keys = ['resistance']
        variable_by_segment = ['segment_name', 'min_frequency', 'max_frequency', 'step']
        header_info = {
            'measurement_description': {},
            'levelling_settings': {},
            'parsed_header': {},
        }

        def try_to_add(dictionary, key, value):
            try:
                existing_data = dictionary[key]
                if existing_data != value:
                    raise ValueError(
                        'Inconsistent data in header field {}.'.format(key)
                    )
            except KeyError:
                dictionary[key] = value

        for header_item in variable_by_segment:
            header_info[header_item] = {}

        for path in self.segment_files:
            data, header = self._read_file(path)
            parsed_segment_data.append(data)
            segment_name = header['segment_name']
            for item in header.items():
                key, value = item
                if key in variable_by_segment:
                    try:
                        header_info['parsed_header'][key][segment_name] = value
                    except KeyError:
                        header_info['parsed_header'][key] = {}
                else:
                    if key in measurement_description_keys:
                        try_to_add(header_info['measurement_description'], key, value)

                    elif key in levelling_settings_keys:
                        try_to_add(header_info['levelling_settings'], key, value)

                    else:
                        try_to_add(header_info['parsed_header'], key, value)

        parsed_config = _merge_dicts(
            [CROWLEY_DEFAULT_CONFIG, self.extra_config, header_info]
        )
        self.parsed_config = {
            key: parsed_config[key]
            for key in DEFAULT_KEYS
            if key in parsed_config.keys()
        }
        reformatted_segment_data = self._reformat_for_analysis(parsed_segment_data)
        self.data = _CrowleyData(reformatted_segment_data)

    def _get_measurement_description(self, key):
        """Get information from measurement description.
        The measurement description traditionally contains: lead_resistance, connector_type, mount_name, connect_number,
        calorimeter, notes, resistance.
        """
        measurement_description_keys = [
            'lead_resistance',
            'connector_type',
            'mount_name',
            'connect_number',
            'calorimeter',
            'notes',
        ]
        levelling_settings_keys = ['resistance']
        retval = None
        if key in measurement_description_keys:
            retval = self.parsed_config['measurement_description'][key]

        elif key in levelling_settings_keys:
            retval = self.parsed_config['levelling_settings'][key]

        if retval is None:
            raise KeyError('{} not found in measurement description'.format(key))

        return retval


class Segment:
    """
    A segment begins an ends with a long period of RF off.

    In the middle, there are steps, where RF is on at constant frequency.

    This is a useful way to divide the data for two reasons.

    First, both initial and final RF off need to exist to estimated effective
    efficiency.

    Second, Tom Crowley's matlab software is built on the assumption
    that data files are in this format.
    """

    def __init__(self, raw_data, complete, run: Run):
        """
        Initialize a segmet object.

        Parameters
        ----------
        raw_data : dict
            keys: str, all of the values in necessary_columns included
            values: numpy arrays of data

        complete : bool
            If the RF power was turned off and thermal equilibrium was achieved,
            this variable is true

        run : Run
            The run that this segment is a part of.

        Returns
        -------
        None.

        """
        self.run = run
        self.header = None
        self.name = None  # Name of data file
        self.raw_data = raw_data
        self.results = {}  # stores results of calculations performed on this segment
        self.metadata = {}
        self.results['complete'] = complete  # False if the run was interrupted
        self.index = np.arange(len(self.raw_data['timestamp']))
        self.steps = []  # stores data for each frequency step

    def analyze(self):
        """
        Analyze a segment.

        Closely analogous to run.analyze.
        * identifies the frequency steps present in the data
        * populates self.steps
        * calls "analyze" on each step

        Returns
        -------
        None.

        """
        for item in self.run.analyzers.items():
            key, analyzer = item
            results, metadata = analyzer.analyze_segment(self)
            self.results.update(results)
            self.metadata.update(metadata)

        # steps are determined by times when RF power was on
        frequencies = self.raw_data['frequency']
        power_on = self.raw_data['power_on']
        frequencies[np.logical_not(self.raw_data['power_on'])] = 0
        numf = len(frequencies)
        step_id = np.zeros(numf, dtype='int64')

        # step boundaries are marked with 0s.
        # the tricky thing about segments is that adjacent segments share 0's.
        # start by assigning non-zero frequency points to segments
        step_counter = 0
        last_on = None
        current_on = None
        for i in range(0, numf):
            last_on = current_on
            current_on = power_on[i]

            if current_on and not last_on:
                step_counter += 1
                # print(i,frequencies[i], current_on, step_id[i], step_counter)

            if not current_on:
                step_id[i] = 0

            else:
                step_id[i] = step_counter
                # print(i,frequencies[i], current_on, step_id[i], step_counter)#, step[i])

        #######################################################################
        # now that segment numbers have been assigned, iterate through data.
        # and find start and stop of each step

        step_frequencies = []
        step_start_indices = []
        step_end_indices = []

        fset = set()
        step_start = 0
        highest_step_id = 1
        for i in range(numf):
            current_step_id = step_id[i]
            # print(i, step_id[i])
            if current_step_id > highest_step_id or i == numf - 1:
                highest_step_id = current_step_id

                if len(fset) > 1:
                    if VERBOSE:
                        print('fset', fset)
                    raise (Exception("This shouldn't happen"))

                if len(fset) == 1:  # 0 is possible if incomplete data
                    f = fset.pop()
                    step_frequencies.append(f)
                    step_start_indices.append(step_start)
                    step_end_indices.append(i)
                    step_start = i
                    fset = set()

            f = frequencies[i]
            if f > 0 and power_on[i]:
                fset.add(f)

        if VERBOSE:
            print(step_start_indices, step_end_indices, step_frequencies)

        self.steps = []
        for i, start in enumerate(step_start_indices):
            end = step_end_indices[i]

            raw_data = {}
            for column in self.raw_data.keys():
                # print("column:", column)
                # print("column,start, end:", column, start, end)
                raw_data[column] = np.array(self.raw_data[column])[start:end]

            new_step = Step(self, raw_data, step_frequencies[i])
            self.steps.append(new_step)
            print('step index', i)
            new_step.analyze()

    def __getitem__(self, key):
        """
        If key is a key in self.results, return associated data or if key is a key in self.raw_data, return associated data.

        Parameters
        ----------
        key : str
            Column to retrieve value from.

        Returns
        -------
        data

        """
        retval = None
        if key in self.results.keys():
            retval = self.results[key]

        if key in self.raw_data.keys():
            if retval is not None:
                raise Exception(
                    'Could not find data because key is shared between self.results and self.raw_data.'
                )

            retval = self.raw_data[key]

        if retval is None:
            raise (KeyError(key))

        return retval


class Step:
    """
    A step is a time period when RF power was on at a constant frequency.

    Adjacent periods of RF off are included.
    """

    def __init__(self, segment: Segment, raw_data: dict, frequency: float):
        """
        Initialize a step object.

        Parameters
        ----------
        segment : Segment
            Segment that this step belongs to.

        raw_data : dict
            keys: variable names
            columns: timeseries data

        frequency : float
            frequency in GHz.

        Returns
        -------
        None.

        """
        self.segment = segment
        self.frequency = frequency  # in GHz
        self.raw_data = raw_data
        self.index = np.arange(len(self.raw_data['timestamp']))
        # Initialize results that analysis function expect
        self.results = {'RF_off_time': None}
        self.metadata = {}

    def __getitem__(self, key):
        """
        If key is a key in self.results, return associated data, or if key is a key in self.raw_data, return associated data.

        Parameters
        ----------
        key : str
            Column to retrieve value from.

        Returns
        -------
        data

        """
        retval = None
        if key in self.results.keys():
            retval = self.results[key]

        if key in self.raw_data.keys():
            if retval is not None:
                raise Exception(
                    'Could not find data because key is shared between self.results and self.raw_data.'
                )

            retval = self.raw_data[key]

        if retval is None:
            raise (KeyError(key))

        return retval

    def analyze(self):
        """
        Analyzes the RF off period.

        Returns
        -------
        None.

        """
        if VERBOSE:
            print('\n')
            print('STEP ANALYSIS', self.frequency, ' GHz')
            print('--------------------------------------')

        # within the resampled index, identify where the RF off point is
        timestamps = self.raw_data['timestamp']

        # index of steps in context of run?
        step_index = self.raw_data['step_counter']
        power_on = self.raw_data['power_on']
        mid_t = (timestamps[1:] + timestamps[:-1]) / 2.0

        # find where the RF was turned off according to the source
        # used as a starting point for finding the RF off for every other
        # instrument with a timeseries.
        switch = np.where(abs(np.diff(power_on, append=0)) > 0)[0]
        self.results['complete'] = True
        if len(switch) == 0:
            self.results['complete'] = False
            if VERBOSE:
                print('STEP INCOMPLETE. NO power switches found')
            return None

        switch = switch[-1]
        self.results['step_RF_off'] = step_index[switch]
        self.results['RF_off_time'] = mid_t[switch]

        for item in self.segment.run.analyzers.items():
            key, analyzer = item
            results, metadata = analyzer.analyze_step(self)
            self.results.update(results)
            self.metadata.update(metadata)

        return None


def _get_spec_uncertainties(spec, val):
    """The specs_class expects to deal with arrays, but we have scalar values."""
    val = np.array([val])
    uncert = spec.all_manufacturer_errors(val)
    return uncert[0]


# Long-term plan: use these classes in the runner as well
class SignalAnalyzer(abc.ABC):
    """Analyzes Signal generated by set of instruments doing related things."""

    def __init__(
        self, analysis_config, signal_config, input_signal_config, instruments
    ):
        self.analysis_config = analysis_config
        time_zero = self.analysis_config['time_zero']
        self.input_signal_config = input_signal_config
        self.signal_config = signal_config
        self.instruments = instruments
        self.specs = {}
        self.specs_metadata = {}
        for item in self.input_signal_config.items():
            input_signal_name, config = item
            instrument_name = config['instrument']
            role = instruments[instrument_name]['role']
            model = instruments[instrument_name]['model']
            serial = instruments[instrument_name]['serial']
            try:
                spec_cls = INSTRUMENT_SPECS[model][role][input_signal_name]
                spec = spec_cls(role, serial, time_zero=time_zero)
                self.specs[input_signal_name] = spec

                instr = spec.__module__.split('.')[1] + ':' + spec.serial
                self.specs_metadata[input_signal_name] = {'instrument_name': instr}
            except KeyError:
                print('NO SPEC CLASS FOR ', model, ' as ', role)

    @abc.abstractmethod
    def analyze_segment(self, segment: Segment) -> tuple:
        """
        Analyze segment.

        Parameters
        ----------
        segment : Segment
            Segment to analyze.

        Returns
        -------
        dict
            Analysis results

        dict
            metadata
        """
        pass

    @abc.abstractmethod
    def analyze_step(self, Step: Step) -> dict:
        """
        Analyze segment, return dictionary of results.

        Parameters
        ----------
        step : Step
            Step to analyze.

        Returns
        -------
        dict
            Analysis results

        """
        pass

    @abc.abstractmethod
    def plot_analysis(self, segment: Segment) -> pl.Figure | tuple[pl.Figure]:
        """
        Plot the analysis performed on a segment.
        
        The plot should demonstrate the analysis being performed ont he raw
        data (like highlighting the samples taken on a time series which
              are being averaged.)

        Parameters
        ----------
        segment : Segment
            segment to be analyed.

        Returns
        -------
        None.

        """

class ThermoelectricAnalyzer(SignalAnalyzer):
    """Analyzes thermopile voltage timeseries."""

    def __init__(
        self, analysis_config, signal_config, input_signal_config, instruments
    ):
        """Initialize a thermopile monitor."""
        SignalAnalyzer.__init__(
            self, analysis_config, signal_config, input_signal_config, instruments
        )
        self.RF_on_average_window = self.analysis_config['RF_on_average_window']
        self.column = self.input_signal_config['e']['column']
        try:
            self.stats_window_override = self.analysis_config['stats_window_override']
        except KeyError:
            self.stats_window_override = None

    def analyze_segment(self, segment: Segment) -> tuple:
        """
        Analyze segment.

        Specifically, analyze the slow RF off periods at the beginning and end
        of a segment.

        Determines the time periods where the temperature was stable
        at the beginning and end of the measurement.
        Performs linear fits vs. time for the initial and final time off periods.
        Also performs linear fit vs. time both off periods combined.

        The combined linear fit is the best estimate of e_off and V_off
        the other two are used for estimating uncertainty.

        Parameters
        ----------
        segment_index : Segment
            Segment to analyze.

        Returns
        -------
        dict
            Analysis results

        dict
            metadata
        """
        results = {}
        metadata = {}
        slow_results = _analyze_off_period(segment, self.column, stats_window_override = self.stats_window_override)
        results.update(slow_results)
        return results, metadata

    def analyze_step(self, step: Step) -> tuple:
        """
        Analyze segment, return dictionary of results.

        Parameters
        ----------
        step : Step
            Step to analyze.

        Returns
        -------
        dict
            Analysis results

        """
        segment = step.segment
        RF_off_time = step.results['RF_off_time']

        results = {}
        off_val = _linear(
            RF_off_time - segment.results['segment_time_zero'],
            segment.results[self.column + '_off_a'],
            segment.results[self.column + '_off_b'],
        )

        results[self.column + '_off'] = off_val
        spec = self.specs['e']

        on_results = _average_pre_fastoff(step, self.column, self.RF_on_average_window)
        results.update(on_results)
        on_val = results[self.column + '_on']

        # add specs
        results[self.column + '_off' + '_spec'] = _get_spec_uncertainties(spec, off_val)
        results[self.column + '_on' + '_spec'] = _get_spec_uncertainties(spec, on_val)

        metadata = {}
        metadata[self.column + '_on' + '_spec'] = {}
        metadata[self.column + '_off' + '_spec'] = {}

        metadata[self.column + '_on' + '_spec']['instrument_name'] = (
            self.specs_metadata['e']['instrument_name']
        )
        metadata[self.column + '_off' + '_spec']['instrument_name'] = (
            self.specs_metadata['e']['instrument_name']
        )
        return results, metadata

    def plot_analysis(self, segment: Segment, *args):
        """
        Generate plot of the thermopile voltage to allow user to see if the measurements look normal.

        Parameters
        ----------
        segment : Segment
            Segment to analyze.

        Returns
        -------
        fig : matplotlib figure object

        """
        column = self.column
        fig, ax = pl.subplots()
        segment_results = segment.results
        segment_raw_data = segment.raw_data

        # specific results
        initial_off_start = segment_results['initial_off_start']
        initial_off_stop = segment_results['initial_off_stop']
        final_off_start = segment_results['final_off_start']
        final_off_stop = segment_results['final_off_stop']

        start_time = segment_raw_data[column + '_timestamp'][0]
        plot_time = segment_raw_data[column + '_timestamp'] - start_time
        V_NVM = segment_raw_data[column]

        ax.plot(
            plot_time, V_NVM, color=MAIN_TRACE_COLOR, linewidth=MAIN_TRACE_LINEWIDTH
        )
        # e_i_fit = _linear(segment_raw_data["NVM_volts_timestamp"] - segment_results["time_zero_NVM_volts_i"], segment_results["NVM_volts_off_i_drift"], segment_results["NVM_volts_off_i"])
        # e_f_fit = _linear(segment_raw_data["NVM_volts_timestamp"] - segment_results["time_zero_NVM_volts_f"], segment_results["NVM_volts_off_f_drift"], segment_results["NVM_volts_off_f"])
        e_off_fit = _linear(
            segment_raw_data[column + '_timestamp']
            - segment_results['segment_time_zero'],
            segment_results[column + '_off_a'],
            segment_results[column + '_off_b'],
        )

        # ax.plot(plot_time, e_i_fit, color=INITIAL_STABLE_TRACE_COLOR, linewidth=STABLE_TRACE_LINEWIDTH)
        # ax.plot(plot_time, e_f_fit, color=FINAL_STABLE_TRACE_COLOR, linewidth=STABLE_TRACE_LINEWIDTH)
        ax.plot(
            plot_time,
            e_off_fit,
            color=MIDDLE_STABLE_TRACE_COLOR,
            linewidth=STABLE_TRACE_LINEWIDTH,
        )

        for i, step_i in enumerate(segment.steps):
            step_i_raw_data = step_i.raw_data
            plot_time_step = step_i_raw_data[column + '_timestamp'] - start_time
            V_NVM_step = step_i_raw_data[column]
            initial_stable = step_i.results[column + '_initial_stable']
            final_stable = step_i.results[column + '_final_stable']
            stable = np.logical_and(
                step_i.index >= initial_stable, step_i.index <= final_stable
            )

            ax.plot(
                plot_time_step,
                V_NVM_step,
                linewidth=MAIN_TRACE_LINEWIDTH,
                color=PLOT_COLORS[i % len(PLOT_COLORS)],
            )
            ax.plot(
                plot_time_step[stable],
                V_NVM_step[stable],
                linewidth=STABLE_TRACE_LINEWIDTH,
                color=MIDDLE_STABLE_TRACE_COLOR,
            )  # PLOT_COLORS[i % len(PLOT_COLORS)])

            RF_off_time = step_i.results['RF_off_time'] - start_time
            NVM_volts_on = step_i.results[column + '_on']
            NVM_volts_off = step_i.results[column + '_off']
            ax.plot(
                [RF_off_time, RF_off_time],
                [NVM_volts_on, NVM_volts_off],
                marker='o',
                markersize=FIT_POINT_SIZE,
                color=PLOT_COLORS[i % len(PLOT_COLORS)],
            )

        ax.plot(
            plot_time[initial_off_start:initial_off_stop],
            V_NVM[initial_off_start:initial_off_stop],
            color=INITIAL_STABLE_TRACE_COLOR,
            linewidth=STABLE_TRACE_LINEWIDTH,
        )
        ax.plot(
            plot_time[final_off_start:final_off_stop],
            V_NVM[final_off_start:final_off_stop],
            color=FINAL_STABLE_TRACE_COLOR,
            linewidth=STABLE_TRACE_LINEWIDTH,
        )

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Thermopile Voltage (V)')
        return fig


class BolometerAnalyzer(SignalAnalyzer):
    def __init__(
        self, analysis_config, signal_config, input_signal_config, instruments
    ):
        """Initialize a BiasMonitor analyzer."""
        SignalAnalyzer.__init__(
            self, analysis_config, signal_config, input_signal_config, instruments
        )

        # TODO: fix
        column = self.signal_config['vdc']['column']
        instr_timing_tolerance = self.analysis_config['instr_timing_tolerance']
        V_off_delay = self.analysis_config['V_off_delay']
        V_off_function = self.analysis_config['V_off_function']
        V_off_fit_time_window = self.analysis_config['V_off_fit_time_window']
        RF_on_average_window = self.analysis_config['RF_on_average_window']
        
        try:
            self.stats_window_override = self.analysis_config['stats_window_override']
        except KeyError:
            self.stats_window_override = None

        self.column = column
        self.instr_timing_tolerance = instr_timing_tolerance
        self.V_off_function = V_off_function
        self.V_off_delay = V_off_delay
        self.V_off_fit_time_window = V_off_fit_time_window
        self.RF_on_average_window = RF_on_average_window

    def analyze_segment(self, segment: Segment) -> tuple:
        """
        Analyze segment.

        Specifically, analyze the slow RF off periods at the beginning and end
        of a segment.

        Determines the time periods where the temperature was stable at the beginning and end of the measurement.
        Performs linear fits vs. time for the initial and final time off periods.
        Also performs linear fit vs. time both off periods combined.

        The combined linear fit is the best estimate of e_off and V_off
        the other two are used for estimating uncertainty.

        Parameters
        ----------
        segment : Segment
            Segment to analyze.

        metadata : dict
            Descriptive information reported with analysis results.
        Returns
        -------
        results : dict
            See _analyze_off_periods.

        metadata : dict
            Descriptive information reported with analysis results.
        """
        metadata = {}
        results = _analyze_off_period(segment, self.column, stats_window_override = self.stats_window_override)
        return results, metadata

    def analyze_step(self, step: Step) -> tuple:
        """
        Analyze segment, return dictionary of results.

        Parameters
        ----------
        step : Step
            Step to analyze.

        Returns
        -------
        results : dict
            Quantities extracted from analysis.

        metadata : dict
            Descriptive information reported with analysis results.
        """
        results = {}
        segment = step.segment
        RF_off_time = step.results['RF_off_time']
        spec = self.specs['vdc']

        off_results = _fit_fast_off_timeseries(
            step,
            self.column,
            self.V_off_function,
            self.V_off_delay,
            self.instr_timing_tolerance,
            self.V_off_fit_time_window,
        )

        on_results = _average_pre_fastoff(step, self.column, self.RF_on_average_window)
        on_val = on_results[self.column + '_on']
        results.update(on_results)

        results.update(off_results)
        off_slow_val = _linear(
            RF_off_time - segment.results['segment_time_zero'],
            segment.results[self.column + '_off_a'],
            segment.results[self.column + '_off_b'],
        )

        results[self.column + '_off_slow'] = off_slow_val

        # add specs
        off_fast_val = results[self.column + '_off_fast']
        results[self.column + '_off_slow' + '_spec'] = _get_spec_uncertainties(
            spec, off_slow_val
        )
        results[self.column + '_off_fast' + '_spec'] = _get_spec_uncertainties(
            spec, off_fast_val
        )
        results[self.column + '_on' + '_spec'] = _get_spec_uncertainties(spec, on_val)

        metadata = {}
        metadata[self.column + '_on' + '_spec'] = {}
        metadata[self.column + '_off_slow' + '_spec'] = {}
        metadata[self.column + '_off_fast' + '_spec'] = {}

        metadata[self.column + '_on' + '_spec']['instrument_name'] = (
            self.specs_metadata['vdc']['instrument_name']
        )
        metadata[self.column + '_off_slow' + '_spec']['instrument_name'] = (
            self.specs_metadata['vdc']['instrument_name']
        )
        metadata[self.column + '_off_fast' + '_spec']['instrument_name'] = (
            self.specs_metadata['vdc']['instrument_name']
        )
        return results, metadata

    def plot_analysis(self, segment: Segment, *args):
        """
        Generate plot of the bias voltage to allow user to see if the measurements look normal.

        Parameters
        ----------
        segment : Segment
            Segment to analyze.

        Returns
        -------
        fig : matplotlib figure object
        """
        V_off_function = self.V_off_function

        fig, ax = pl.subplots()
        segment_results = segment.results
        segment_raw_data = segment.raw_data
        start_time = segment_raw_data[self.column + '_timestamp'][0]
        plot_time = segment_raw_data[self.column + '_timestamp'] - start_time
        V_DVM = segment_raw_data[self.column]

        pl.plot(
            plot_time, V_DVM, color=MAIN_TRACE_COLOR, linewidth=MAIN_TRACE_LINEWIDTH
        )

        initial_off_start = segment_results['initial_off_start']
        initial_off_stop = segment_results['initial_off_stop']
        final_off_start = segment_results['final_off_start']
        final_off_stop = segment_results['final_off_stop']

        ax.plot(
            plot_time, V_DVM, color=MAIN_TRACE_COLOR, linewidth=MAIN_TRACE_LINEWIDTH
        )

        # e_i_fit = _linear(segment_raw_data["NVM_volts_timestamp"] - segment_results["time_zero_NVM_volts_i"], segment_results["NVM_volts_off_i_drift"], segment_results["NVM_volts_off_i"])
        # e_f_fit = _linear(segment_raw_data["NVM_volts_timestamp"] - segment_results["time_zero_NVM_volts_f"], segment_results["NVM_volts_off_f_drift"], segment_results["NVM_volts_off_f"])
        e_off_fit = _linear(
            segment_raw_data[self.column + '_timestamp']
            - segment_results['segment_time_zero'],
            segment_results[self.column + '_off_a'],
            segment_results[self.column + '_off_b'],
        )

        # ax.plot(plot_time, e_i_fit, color=INITIAL_STABLE_TRACE_COLOR, linewidth=STABLE_TRACE_LINEWIDTH)
        # ax.plot(plot_time, e_f_fit, color=FINAL_STABLE_TRACE_COLOR, linewidth=STABLE_TRACE_LINEWIDTH)
        ax.plot(
            plot_time,
            e_off_fit,
            color=MIDDLE_STABLE_TRACE_COLOR,
            linewidth=STABLE_TRACE_LINEWIDTH,
        )

        for i, step_i in enumerate(segment.steps):
            off_time_delta = step_i.results[self.column + '_off_time_delta']
            plot_time_step = step_i.raw_data[self.column + '_timestamp'] - start_time
            V_DVM_step = step_i.raw_data[self.column]

            t_off = step_i.results['RF_off_time'] - start_time + off_time_delta
            eval_time = t_off + self.V_off_delay

            V_off_slow = step_i.results[self.column + '_off_slow']
            V_off_fast = step_i.results[self.column + '_off_fast']
            V_on = step_i.results[self.column + '_on']

            plot_time_step = step_i.raw_data[self.column + '_timestamp'] - start_time
            V_DVM_step = step_i.raw_data[self.column]

            fit_region_time = step_i.raw_data[self.column + '_timestamp'][
                step_i.results['initial_fit_region'] : step_i.results[
                    'final_fit_region'
                ]
            ]
            fit_time_zero = step_i.results['fit_time_zero']
            fit_time = fit_region_time - fit_time_zero

            if V_off_function == 'lin_plus_exp':
                fit_volts = _lin_plus_exp(
                    fit_time,
                    step_i.results[self.column + '_fit_a'],
                    step_i.results[self.column + '_fit_b'],
                    step_i.results[self.column + '_fit_vscale'],
                    step_i.results[self.column + '_fit_tscale'],
                )

            elif V_off_function == 'linear':
                fit_volts = _linear(
                    fit_time,
                    step_i.results[self.column + '_fit_a'],
                    step_i.results[self.column + '_fit_b'],
                )

            if V_off_function == 'lin_plus_exp' or V_off_function == 'linear':
                pl.plot(
                    fit_region_time - start_time,
                    fit_volts,
                    color=MIDDLE_STABLE_TRACE_COLOR,
                )
            pl.plot(plot_time_step, V_DVM_step)
            pl.plot(
                [t_off, t_off, t_off],
                [V_off_fast, V_off_slow, V_on],
                marker='o',
                linestyle='--',
                color=FIT_TRACE_COLOR,
                markersize=RESAMPLE_POINTS_SIZE,
            )
            pl.plot(
                [eval_time],
                [V_off_fast],
                marker='o',
                color=FIT_TRACE_COLOR,
                markersize=RESAMPLE_POINTS_SIZE,
            )

        ax.plot(
            plot_time[initial_off_start:initial_off_stop],
            V_DVM[initial_off_start:initial_off_stop],
            color=INITIAL_STABLE_TRACE_COLOR,
            linewidth=STABLE_TRACE_LINEWIDTH,
        )
        ax.plot(
            plot_time[final_off_start:final_off_stop],
            V_DVM[final_off_start:final_off_stop],
            color=FINAL_STABLE_TRACE_COLOR,
            linewidth=STABLE_TRACE_LINEWIDTH,
        )

        pl.xlabel('Time (s)')
        pl.ylabel('Bias Voltage (V)')
        return fig


class SMUPowerMeterAnalyzer(SignalAnalyzer):
    def __init__(
        self, analysis_config, signal_config, input_signal_config, instruments
    ):
        """Initialize an SMUPowerMeter analyzer."""
        SignalAnalyzer.__init__(
            self, analysis_config, signal_config, input_signal_config, instruments
        )

        # TODO: fix
        i_column = signal_config['idc']['column']
        v_column = signal_config['vdc']['column']

        instr_timing_tolerance = self.analysis_config['instr_timing_tolerance']
        V_off_delay = self.analysis_config['V_off_delay']
        V_off_function = self.analysis_config['V_off_function']
        V_off_fit_time_window = self.analysis_config['V_off_fit_time_window']
        RF_on_average_window = self.analysis_config['RF_on_average_window']

        self.i_column = i_column
        self.v_column = v_column
        self.instr_timing_tolerance = instr_timing_tolerance
        self.V_off_function = V_off_function
        self.V_off_delay = V_off_delay
        self.V_off_fit_time_window = V_off_fit_time_window
        self.RF_on_average_window = RF_on_average_window

    def analyze_segment(self, segment: Segment) -> tuple:
        """
        Analyze segment.

        Specifically, analyze the slow RF off periods at the beginning and end
        of a segment.

        Determines the time periods where the temperature was stable at the beginning and end of the measurement.
        Performs linear fits vs. time for the initial and final time off periods.
        Also performs linear fit vs. time both off periods combined.

        The combined linear fit is the best estimate of e_off and V_off
        the other two are used for estimating uncertainty.

        Parameters
        ----------
        segment : Segment
            Segment to analyze.

        Returns
        -------
        results : dict
            Quantities extracted from analysis.

        metadata : dict
            Descriptive information reported with analysis results.
        """

    def analyze_step(self, step: Step) -> tuple:
        """
        Analyze segment, return dictionary of results.

        Parameters
        ----------
        Step : step
            Step to analyze.

        Returns
        -------
        results : dict
            See _fit_fast_off_timeseries.

        metadata : dict
            Descriptive information reported with analysis results.
        """
        results = {}
        off_v_results = _fit_fast_off_timeseries(
            step,
            self.v_column,
            self.V_off_function,
            self.instr_timing_tolerance,
            self.V_off_fit_time_window,
        )

        off_i_results = _fit_fast_off_timeseries(
            step,
            self.i_column,
            self.V_off_function,
            self.instr_timing_tolerance,
            self.V_off_fit_time_window,
        )

        on_v_results = _average_pre_fastoff(
            step, self.v_column, self.RF_on_average_window
        )
        on_i_results = _average_pre_fastoff(
            step, self.i_column, self.RF_on_average_window
        )

        results.update(off_v_results)
        results.update(off_i_results)
        results.update(on_v_results)
        results.update(on_i_results)

        metadata = {}
        return results, metadata


class PowerMeterAnalyzer(SignalAnalyzer):
    def __init__(
        self, analysis_config, signal_config, input_signal_config, instruments
    ):
        """
        Initialize a PowerMeter analyzer.

        Parameters
        ----------
        column : str
            Name of data column.

        RF_on_average_window : float
            Time to average power measurements, in seconds.

        Returns
        -------
        None.

        """
        SignalAnalyzer.__init__(
            self, analysis_config, signal_config, input_signal_config, instruments
        )
        self.column = self.input_signal_config['power']['column']
        self.instr_timing_tolerance = self.analysis_config['instr_timing_tolerance']
        self.RF_on_average_window = self.analysis_config['RF_on_average_window']
        try:
            self.stats_window_override = self.analysis_config['stats_window_override']
        except KeyError:
            self.stats_window_override = None

    def analyze_segment(self, segment: Segment) -> tuple:
        """
        Analyze segment.

        Specifically, analyze the slow RF off periods at the beginning and end
        of a segment.

        Determines the time periods where the temperature was stable at the beginning and end of the measurement.
        Performs linear fits vs. time for the initial and final time off periods.
        Also performs linear fit vs. time both off periods combined.

        The combined linear fit is the best estimate of e_off and V_off
        the other two are used for estimating uncertainty.

        Parameters
        ----------
        segment : Segment
            Segment to analyze.

        Returns
        -------
        results : dict
            Quantities extracted from analysis.

        metadata : dict
            Descriptive information reported with analysis results.
        """
        # this should envoke analyze_step?
        metadata = {}
        results = _analyze_off_period(segment, self.column, stats_window_override = self.stats_window_override)
        return results, metadata

    def analyze_step(self, step: Step) -> tuple:
        """
        Analyze step, return dictionary of results.

        Parameters
        ----------
        step: Step : int
            Index of the segment in run.segments.

        Returns
        -------
        results : dict
            See _average_pre_fastoff.

        metadata : dict
            Descriptive information reported with analysis results.
        """
        metadata = {}
        results = _average_pre_fastoff(step, self.column, self.RF_on_average_window)
        return results, metadata

    def plot_analysis(self, segment: Segment, *args):
        """
        Generate plot of the power meter to allow user to see if the measurements look normal.

        Parameters
        ----------
        segment : Segment
            Segment to analyze.

        Returns
        -------
        fig : matplotlib figure object

        """
        column = self.column
        fig, ax = pl.subplots()
        segment_results = segment.results
        segment_raw_data = segment.raw_data

        # specific results
        initial_off_start = segment_results['initial_off_start']
        initial_off_stop = segment_results['initial_off_stop']
        final_off_start = segment_results['final_off_start']
        final_off_stop = segment_results['final_off_stop']

        start_time = segment_raw_data[column + '_timestamp'][0]
        plot_time = segment_raw_data[column + '_timestamp'] - start_time
        V_NVM = segment_raw_data[column]

        ax.plot(
            plot_time, V_NVM, color=MAIN_TRACE_COLOR, linewidth=MAIN_TRACE_LINEWIDTH
        )
        # e_i_fit = _linear(segment_raw_data["NVM_volts_timestamp"] - segment_results["time_zero_NVM_volts_i"], segment_results["NVM_volts_off_i_drift"], segment_results["NVM_volts_off_i"])
        # e_f_fit = _linear(segment_raw_data["NVM_volts_timestamp"] - segment_results["time_zero_NVM_volts_f"], segment_results["NVM_volts_off_f_drift"], segment_results["NVM_volts_off_f"])
        e_off_fit = _linear(
            segment_raw_data[column + '_timestamp']
            - segment_results['segment_time_zero'],
            segment_results[column + '_off_a'],
            segment_results[column + '_off_b'],
        )

        # ax.plot(plot_time, e_i_fit, color=INITIAL_STABLE_TRACE_COLOR, linewidth=STABLE_TRACE_LINEWIDTH)
        # ax.plot(plot_time, e_f_fit, color=FINAL_STABLE_TRACE_COLOR, linewidth=STABLE_TRACE_LINEWIDTH)
        ax.plot(
            plot_time,
            e_off_fit,
            color=MIDDLE_STABLE_TRACE_COLOR,
            linewidth=STABLE_TRACE_LINEWIDTH,
        )

        for i, step_i in enumerate(segment.steps):
            step_i_raw_data = step_i.raw_data
            plot_time_step = step_i_raw_data[column + '_timestamp'] - start_time
            V_NVM_step = step_i_raw_data[column]
            initial_stable = step_i.results[column + '_initial_stable']
            final_stable = step_i.results[column + '_final_stable']
            stable = np.logical_and(
                step_i.index >= initial_stable, step_i.index <= final_stable
            )

            ax.plot(
                plot_time_step,
                V_NVM_step,
                linewidth=MAIN_TRACE_LINEWIDTH,
                color=PLOT_COLORS[i % len(PLOT_COLORS)],
            )
            ax.plot(
                plot_time_step[stable],
                V_NVM_step[stable],
                linewidth=STABLE_TRACE_LINEWIDTH,
                color=MIDDLE_STABLE_TRACE_COLOR,
            )  # PLOT_COLORS[i % len(PLOT_COLORS)])

            RF_off_time = step_i.results['RF_off_time'] - start_time
            P_on = step_i.results[column + '_on']
            # NVM_volts_off = step_i.results[column + '_off']
            ax.plot(
                [RF_off_time],
                [P_on],
                marker='o',
                markersize=FIT_POINT_SIZE,
                color=PLOT_COLORS[i % len(PLOT_COLORS)],
            )

        ax.plot(
            plot_time[initial_off_start:initial_off_stop],
            V_NVM[initial_off_start:initial_off_stop],
            color=INITIAL_STABLE_TRACE_COLOR,
            linewidth=STABLE_TRACE_LINEWIDTH,
        )
        ax.plot(
            plot_time[final_off_start:final_off_stop],
            V_NVM[final_off_start:final_off_stop],
            color=FINAL_STABLE_TRACE_COLOR,
            linewidth=STABLE_TRACE_LINEWIDTH,
        )

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Power (W)')
        return fig


def _average_pre_fastoff(step: Step, column: str, RF_on_average_window: float) -> dict:
    """
    Average samples just prior to the fast offs.

    This is mostly used for commercial power meters, which can self
    zero during the off periods between segments and do not need an "off"
    measurement to measured power at a particular time.

    Parameters
    ----------
    step: Step
        Step to analyze.

    column : str
        Name of column to analyze.

    RF_on_average_window : float
        How long to average just prior to fast RF off in seconds.

    Returns
    -------
    results : dict
        column + '_on: average reading.
        column + '_on_dev': standard deviation of readings.

    """
    results = {}
    RF_off_time = step.results['RF_off_time']
    timestamps = step.raw_data[column + '_timestamp']
    index = np.arange(len(timestamps))

    logical_index = np.logical_and(
        timestamps >= RF_off_time - RF_on_average_window,
        timestamps
        < RF_off_time
        - 0.01,  # just to help avoid collisions with a step that is right after the step
    )
    indexed_vals = step.raw_data[column][logical_index]
    # filler value, doesn't mean anything for these sensors
    try:
       results[column + '_initial_stable'] = index[logical_index][0]
    except IndexError as e:
        # this happens is the window to average over was too tight.
        # so print a helpful message.
        if not logical_index.any():
            raise IndexError("No values found in the RF_on_average window. It may be too tight of an averaging window.") from e
        else:
            raise e from e
    results[column + '_final_stable'] = index[logical_index][-1]

    results[column + '_on'] = np.mean(indexed_vals)
    results[column + '_on_dev'] = np.std(indexed_vals, ddof=1)
    if VERBOSE:
        print(column, ' on', results[column + '_on'])

    return results


def _analyze_off_period(
        segment: Segment,
        column: str,
        stats_window_override: float | None = None
    ) -> dict:
    """
    Analyze segment.

    Specifically, analyze the slow RF off periods at the beginning and end
    of a segment.

    Determines the time periods where the temperature was stable
    at the beginning and end of the measurement.
    Performs linear fits vs. time for the initial and final time off periods.
    Also performs linear fit vs. time both off periods combined.

    The combined linear fit is the best estimate of e_off and V_off
    the other two are used for estimating uncertainty.

    Parameters
    ----------
    segment : Segment
        Segment to analyze.

    column : str
        Column to analyze.

    stats_window_override : float | None, optional
        Can be used to manually set the stats window rather then
        use the one defined in the measurement sweep. Useful if the stats
        windows indicates stability too early. If None, then uses
        the stats window of the measurment. The default is None.

    Returns
    -------
    results : dict
        Quantities extracted from analysis.

    """
    run = segment.run

    results = {}
    index = np.arange(len(segment.raw_data['point_counter']))
    RF_on_range = segment.raw_data['power_on']
    first_on_point = index[RF_on_range][0]
    last_on_point = index[RF_on_range][-1]

    initial_off = index < first_on_point
    final_off = index > last_on_point
    stable = segment.raw_data['stable_samples'] > 0

    initial_off_stable = np.logical_and(initial_off, stable)
    final_off_stable = np.logical_and(final_off, stable)

    if not np.any(initial_off_stable) or not np.any(final_off_stable):
        print(np.sum(initial_off_stable), np.sum(final_off_stable))
        results['complete'] = False
        return results

    if type(run) is NewTypeRun:
        # In NewTypeRuns, the microcalorimeter runner's behavior was based
        # on duration of measurements (in seconds),
        # rather than number of samples
        stats_window_seconds = run.parsed_config['stats_settings']['stats_window']
        if stats_window_override:
            stats_window_seconds = stats_window_override
        
        initial_off_start_time = (
            segment.raw_data['timestamp'][initial_off_stable][0] - stats_window_seconds
        )
        initial_off_start = index[
            segment.raw_data['timestamp'] >= initial_off_start_time
        ][0]
        final_off_start_time = segment.raw_data['timestamp'][-1] - stats_window_seconds
        final_off_start = index[segment.raw_data['timestamp'] >= final_off_start_time][
            0
        ]
        initial_off_stop = index[initial_off][-1]
        final_off_stop = index[final_off][-1]

        results['initial_off_start_time'] = initial_off_start_time
        results['final_off_start_time'] = final_off_start_time

    elif type(run) is LegacyRun or type(run) is CrowleyRun:
        stats_window = run.parsed_config['stats_window']
        initial_off_start = index[initial_off_stable][0] - stats_window
        final_off_start = index[final_off_stable][0] - stats_window
        initial_off_stop = index[initial_off][-1]
        final_off_stop = index[final_off][-1]

    results['initial_off_start'] = initial_off_start
    results['final_off_start'] = final_off_start
    results['initial_off_stop'] = initial_off_stop
    results['final_off_stop'] = final_off_stop

    results['segment_time_zero'] = segment.raw_data['timestamp'][0]

    # get off times (initial period, final period, and both combined)
    initial_off_times = segment.raw_data[column + '_timestamp'][
        initial_off_start:initial_off_stop
    ]
    final_off_times = segment.raw_data[column + '_timestamp'][
        final_off_start:final_off_stop
    ]
    off_times = (
        np.hstack((initial_off_times, final_off_times)) - results['segment_time_zero']
    )
    results['time_zero_' + column + '_i'] = initial_off_times[-1]
    results['time_zero_' + column + '_f'] = final_off_times[0]

    # get values of column
    initial_off_vals = segment.raw_data[column][initial_off_start:initial_off_stop]
    final_off_vals = segment.raw_data[column][final_off_start:final_off_stop]
    vals = np.hstack((initial_off_vals, final_off_vals))

    # fit values of combined off regions
    popt, pcov = scipy.optimize.curve_fit(_linear, off_times, vals)
    results[column + '_off_a'], results[column + '_off_b'] = popt

    # fit initial off period bias slope and average
    # try:
    #     popt, pcov = scipy.optimize.curve_fit(_linear, initial_off_times - results["time_zero_" + column + "_i"], initial_off_vals)
    #     results[column + "_off_i_drift"], results[column + "_off_i"] = popt

    # except TypeError as e:
    #     print("error fitting ", column, "initial off period to line. stats_window is probably shorter than the sampling rate")
    #     print(e)
    #     results[column + "_off_i_drift"] = 0
    #     results[column + "_off_i"] = initial_off_vals[0]

    popt, pcov = scipy.optimize.curve_fit(
        _linear,
        initial_off_times - results['time_zero_' + column + '_i'],
        initial_off_vals,
    )
    results[column + '_off_i_drift'], results[column + '_off_i'] = popt

    # determine final slope and average of final off period
    popt, pcov = scipy.optimize.curve_fit(
        _linear, final_off_times - results['time_zero_' + column + '_f'], final_off_vals
    )
    results[column + '_off_f_drift'], results[column + '_off_f'] = popt

    # determine residuals
    initial_off_residuals = initial_off_vals - _linear(
        initial_off_times - results['time_zero_' + column + '_i'],
        results[column + '_off_i_drift'],
        results[column + '_off_i'],
    )
    results[column + '_off_i_dev'] = np.std(initial_off_residuals, ddof=1)

    final_off_residuals = final_off_vals - _linear(
        final_off_times - results['time_zero_' + column + '_f'],
        results[column + '_off_f_drift'],
        results[column + '_off_f'],
    )
    results[column + '_off_f_dev'] = np.std(final_off_residuals, ddof=1)

    return results


def _find_rf_off_delta(step: Step, column: str, instr_timing_tolerance: float) -> float:
    """
    Determine time delay between RF power turn off, and change in sensor reading.

    Parameters
    ----------
    step : Step
        Step to analyze.

    column : str
        Name of column of containing sensor data.

    instr_timing_tolerance : float
        Maximum difference between RF power off, change in sensor reading.

    Returns
    -------
    float
        Time delay.

    """
    timestamps = step.raw_data[column + '_timestamp']
    vals = step.raw_data[column]
    RF_off_time = step.results['RF_off_time']
    # look for max derivative (derivative of step is impulse) near
    # where the source thinks it was turned off.
    # putting a 30 second window
    diff = abs(np.diff(vals, prepend=[vals[0]]))
    bool_in = np.logical_and(
        timestamps <= timestamps[-1],  # usually there is a
        timestamps >= RF_off_time - instr_timing_tolerance,
    )

    step_size = np.max(diff[bool_in])
    i_step = np.where(diff == step_size)[0][-1]
    return timestamps[i_step] - RF_off_time


def _fit_fast_off_timeseries(
    step: Step,
    column: str,
    V_off_function: str,
    V_off_delay: float,
    instr_timing_tolerance: float,
    V_off_fit_time_window: tuple[float, float],
) -> dict:
    """
    Fit a fast off step for a timeseries.

    Parameters
    ----------
    step : Step
        Index of step to analyze.

    column : str
        column to analyze

    V_off_function : str
        Function to use to analyze fast off.
        Options are "lin_plus_exp" or "linear".

    V_off_delay : float
        Time to sample instrument, relative to RF off.

    instr_timing_tolerance : float
        Max delay between RF off and instrument response.

    V_off_fit_time_window : tuple[float, float]
        Window to fit fast off data.

    Returns
    -------
    results: dict
        column + "_off_slow": slow off reading
        column + "_off_fast": fast off reading
        column + "_fit_a": linear fit slope
        column + "_fit_b": linear fit offset
        column + '_fit_vscale']: exponential vertical scale (lin_plus_exp only)
        column + '_fit_tscale']: exponential time scale (lin_plus_exp only)
        column + "_on": "on" reading average
        column + "_on_dev": "on" reading standard deviation
    """
    results = {}
    index = np.arange(len(step.raw_data['point_counter']))
    timestamps = step.raw_data[column + '_timestamp']
    vals = step.raw_data[column]

    # within the resampled index, identify where the RF off point is.
    # use the source as a reference, but find the step in the actual
    # instruments time series
    # find when this instrument thinks source thinks RF was turned off
    # in reference to sources time series

    off_time_delta = _find_rf_off_delta(step, column, instr_timing_tolerance)
    results[column + '_off_time_delta'] = off_time_delta

    if VERBOSE:
        print('RF_off_time (no offset)', step.results['RF_off_time'])
        print('RF off time offset from source ', off_time_delta)

    RF_off_time = step.results['RF_off_time'] + off_time_delta

    # search for when the timeseries thinks RF was turned off

    same_point = np.logical_and(
        (timestamps - RF_off_time) > V_off_fit_time_window[0],
        (timestamps - RF_off_time) < V_off_fit_time_window[1],
    )

    try:
        initial_fit_region = index[same_point][0]
        final_fit_region = index[same_point][-1]
        print(initial_fit_region, final_fit_region)

    except IndexError:
        print('Caught IndexError Trying to fit. Error for :')
        print('fit window = ', V_off_fit_time_window)
        print(
            'relative times (min,max)',
            (min(timestamps - RF_off_time), max(timestamps - RF_off_time)),
        )
        raise IndexError('See above message')

    a = None
    b = None
    vscale = None
    tscale = None
    if V_off_function == 'lin_plus_exp' or V_off_function == 'linear':
        sub_point_volts = vals[initial_fit_region:final_fit_region]
        sub_point_times = timestamps[initial_fit_region:final_fit_region]

        fit_time_zero = sub_point_times[0]
        fit_eval_time = RF_off_time - fit_time_zero + V_off_delay
        fit_times = sub_point_times - fit_time_zero
        fit_volts = sub_point_volts  # [fit_times>1]

        # fit
        if V_off_function == 'lin_plus_exp':
            popt, pcov = scipy.optimize.curve_fit(
                _lin_plus_exp,
                fit_times,
                fit_volts,
                p0=(0, np.max(sub_point_volts) - 0.1, 0, 1),
            )

            V_off_fast = _lin_plus_exp(fit_eval_time, *popt)
            a, b, vscale, tscale = popt

        elif V_off_function == 'linear':
            popt, pcov = scipy.optimize.curve_fit(
                _linear, fit_times, fit_volts, p0=(0, np.max(sub_point_volts))
            )

            V_off_fast = _linear(fit_eval_time, *popt)
            (
                a,
                b,
            ) = popt

    elif V_off_function == 'single_sample':
        if initial_fit_region != final_fit_region:
            raise IndexError(
                'Using single_sample analysis, but RF off region is longer than 1 sample: {} to {}'.format(
                    initial_fit_region, final_fit_region
                )
            )

        fit_time_zero = timestamps[initial_fit_region]
        fit_eval_time = fit_time_zero
        a = vals[initial_fit_region]
        b = 0
        V_off_fast = vals[initial_fit_region]

    else:
        raise ValueError(
            'Fitting function not recognized, Use "lin_plus_exp" or "linear".'
        )

    results['initial_fit_region'] = initial_fit_region
    results['final_fit_region'] = final_fit_region
    results['fit_time_zero'] = fit_time_zero
    results['fit_eval_time'] = fit_eval_time
    results[column + '_off_fast'] = V_off_fast
    results[column + '_fit_a'] = a
    results[column + '_fit_b'] = b
    if vscale is not None and tscale is not None:
        results[column + '_fit_vscale'] = vscale
        results[column + '_fit_tscale'] = tscale

    if VERBOSE:
        print(column, ' off_fast', results[column + '_off_fast'])

    return results
