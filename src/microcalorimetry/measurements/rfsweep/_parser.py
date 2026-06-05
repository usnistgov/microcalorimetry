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
import microcalorimetry.configs as configs
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
from functools import partial

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
            'therm_v': k2450.DatasheetMeasureDCV,
            'therm_i': k2450.DatasheetMeasureDCV,
        },
        'thermometer_monitor': {
            'idc': k2450.DatasheetMeasureDCI,
            'vdc': k2450.DatasheetMeasureDCV,
            'therm_v': k2450.DatasheetMeasureDCV,
            'therm_i': k2450.DatasheetMeasureDCI,
        }
    },
    'K2401': {
        'SMU_power_meter': {
            'idc': k2450.DatasheetMeasureDCI,
            'vdc': k2450.DatasheetMeasureDCV,
            'therm_v': k2450.DatasheetMeasureDCV,
            'therm_i': k2450.DatasheetMeasureDCI
        },
        'thermometer_monitor': {
            'idc': k2450.DatasheetMeasureDCI,
            'vdc': k2450.DatasheetMeasureDCV,
            'therm_v': k2450.DatasheetMeasureDCV,
            'therm_i': k2450.DatasheetMeasureDCI,
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
        },
        'calorimeter_power': {'instr_timing_tolerance': 5},
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
        output = {
            'frequency': [],
            'run': [],
            'segment': [],
            'step': []
        }
        for ri, run in enumerate(self.run_list):
            for seg_i, segment in enumerate(run.segments):
                for step_i, step in enumerate(segment.steps):
                    output['step'].append(step_i)
                    output['run'].append(ri)
                    output['segment'].append(seg_i)
                    output['frequency'].append(step.frequency)
                    for column, val in step.results.items():
                        try:
                            output[column].append(val)
                        except KeyError:
                            output[column] = [val]
        output = pd.DataFrame(output)
        output = output[output.complete]
        output = output.drop(columns='complete')
        output = output.astype(float)

        return output

    def output_RMEmeas(
        self, include_time_std: bool = False, include_specs: bool = True
    ) -> RMEMeas:
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

        def get_dev_umech_id(col, index, step, campaign_name, campaign_uid):
            umech_id = col + '_step_' + \
                str(index) + campaign_name + campaign_uid

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

        # make a uid for campaign
        campaign_uid = str(hash(str(self.run_list[-1])))[:5]

        # count frequencies, columns, uncertainty mechanisms
        columns = set()
        frequencies = []
        for i, step in enumerate(self._generate_steps()):
            f = step.frequency
            frequencies.append(f)
            columns_for_step = [
                col for col in list(step.results.keys()) if not col_is_perturbation(col)
            ]
            columns.update(columns_for_step)

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

        # categories from the uncertainty mechanisms
        a_cats = {
            'Type': 'A',
            'Origin': 'Cal Run Noise',
            'Experiment': 'Calorimeter Run',
        }
        b_cats = {
            'Type': 'B',
            'Origin': 'Cal Run DC Traceability',
            'Experiment': 'Calorimeter Run',
        }

        # add the uncertainty associated with
        # taking the average of a time series
        for i, step in enumerate(self._generate_steps()):
            f = step.frequency
            pert_cols = [col for col in step.results.keys()
                         if col_is_perturbation(col)]
            for col in pert_cols:
                val = step.results[col]
                umech_id = get_dev_umech_id(
                    col, i, step, campaign_name, campaign_uid)
                # Time series standard deviaion uncertainty
                if 'dev' in umech_id:
                    associated_col = get_associated_column(col)
                    pert = nom.copy()
                    try:
                        pert.sel(col=associated_col).data[i] += val
                    except TypeError as e:
                        msg = str(e) + \
                            f' - value is {val} for {associated_col}'
                        raise TypeError(msg) from e
                    # this needs to be inside this loop or if will cause
                    # the else statement to crash
                    if include_time_std:
                        out.add_umech(umech_id, pert, category=a_cats)

        # add the influence of each instruments
        # traceability uncertainties, correlate
        # across segments which last a couple days
        # and the instru,ent likely hasn't drifted signifigantly
        # between segments
        def get_spec_umech_id(col, instr, run_count, segment_count, campaign_name, campaign_uid):
            # we shouldnt assume that campaign names are unique
            umech_id = col + ':' + instr + ':run:' + \
                str(run_count)+':seg:'+str(segment_count) + \
                ':cam:' + campaign_name + campaign_uid
            return umech_id

        pert_cols = [col for col in step.results.keys()
                     if col_is_perturbation(col)]
        use_cols = set([get_associated_column(c) for c in pert_cols])
        instr_col_prefixes = set(
            ['_'.join(c.split('_')[:-1]) for c in use_cols])

        for instr_col_prefix in instr_col_prefixes:
            row_count = 0
            use_pert_cols = [
                pc for pc in pert_cols if instr_col_prefix in pc and 'spec' in pc]
            # use_pert columns is empty
            # so dont add a mechanishsm
            if not use_pert_cols:
                continue
            # this instrument has a spec associated
            # so perturb every segment of data
            for ri, run in enumerate(self.run_list):
                for si, segment in enumerate(run.segments):
                    pert = nom.copy()
                    first_step = segment.steps[0]
                    instrument_name = first_step.metadata[use_pert_cols[0]
                                                          ]['instrument_name']
                    umech_id = get_spec_umech_id(
                        instr_col_prefix, instrument_name, ri, si, campaign_name, campaign_uid)
                    # perturb each dataset originating
                    # from this instrument
                    for sti, step in enumerate(segment.steps):
                        for pert_col in use_pert_cols:
                            val = step.results[pert_col]
                            associated_col = get_associated_column(pert_col)
                            try:
                                pert.sel(col=associated_col).data[row_count] += val
                            except IndexError as e:
                                raise e from e

                        row_count += 1

                    # add this as an uncertainty mechanism
                    out.add_umech(
                        umech_id,
                        pert,
                        category=dict(
                            **b_cats, Instrument=umech_id.split(':')[1])
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
            as an uncertainty mechanism. An overestimation of uncertainty that
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
            'commercial': CommercialPowerMeterAnalyzer,
            'thermoelectric': ThermoelectricAnalyzer,
            'bolometer': BolometerAnalyzer,
            'RF_source': RFSourceAnalyzer,
            # to catch typos
            'rf_source': RFSourceAnalyzer
        }

        self.analyzers = {}
        signals = self.parsed_config['signal_config']
        for signal in self.parsed_config['analysis_config']:
            signal_config = self.parsed_config['signal_config'][signal]
            analysis_config = self.parsed_config['analysis_config'][signal]
            analysis_config['time_zero'] = self.results['time_zero']

            try:
                input_signal_names = self.parsed_config['signal_config'][signal][
                    'input_signals'
                ]
            except KeyError as e:
                raise KeyError(f"input_signals for signal = {signal}") from e

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
                        msg = f'instrument name {instrument_name} in signal_config.{
                            signal} not one of {self.parsed_config["instruments"].keys()}'
                        raise KeyError(msg)

            signal_type = self.parsed_config['signal_config'][signal]['type']

            signal_class = signal_classes[signal_type]
            if signal_class is None:
                raise ValueError(
                    f'Signal type {signal_type} assigned to {
                        signal} has no defined SignalAnalyzer.'
                )

            analyzer = signal_class(
                analysis_config,
                signal_config,
                input_signal_config,
                instruments
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
                raw_segment_data[i][column +
                                    '_timestamp'] = collections.deque()

        def seg_append(index, column, debug = False):
            if debug:
                print(column, '\n time: ', self.data['timestamp'],'\n  val: ', self.data[column])
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
        line_count = 0
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
            debug = False
            for column in self.data.columns:
                if current_segment_id > highest_segment_id:
                    highest_segment_id = current_segment_id

                if current_segment_id > 0:
                    seg_append(current_segment_id - 1, column, debug = debug)

                # handle 0's correctly
                if current_segment_id == 0 and highest_segment_id == 0:
                    seg_append(0, column, debug = debug)

                # handle the ending correctly
                if current_segment_id == 0 and highest_segment_id == segment_counter:
                    seg_append(segment_counter - 1, column, debug = debug)

                if (
                    current_segment_id == 0
                    and highest_segment_id < segment_counter
                    and highest_segment_id > 0
                ):
                    seg_append(highest_segment_id - 1, column, debug = debug)
                    seg_append(highest_segment_id, column, debug = debug)
            line_count += 1

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
                first_sample_is_none = False
                for column in self.data.columns:
                    raw_segment_data[i][column] = np.array(
                        raw_segment_data[i][column])
                    raw_segment_data[i][column + '_timestamp'] = np.array(
                        raw_segment_data[i][column + '_timestamp']
                    )
                
                # Sometimes the first sample of a column will be read as None because
                # of how the data record initiailizes things.
                    # print(f'Segment index 0 {column}: ',raw_segment_data[i][column + '_timestamp'][0], raw_segment_data[i][column][0])

                new_segment = Segment(
                    raw_segment_data[i], segment_complete[i], self)
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
            raise KeyError(
                '{} not found in measurement description'.format(key))

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
            self.expt = ExptParameters(
                self.config_file, self.run_settings_file)

        except FileNotFoundError:
            newdir = dirname(data_file)
            self.config_file = join(newdir, basename(self.config_file))
            self.run_settings_file = join(
                newdir, basename(self.run_settings_file))
            self.expt = ExptParameters(
                self.config_file, self.run_settings_file)

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
            out_segment_i['stable_samples'] = np.zeros(
                num_samples, dtype=np.int64)

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
        variable_by_segment = ['segment_name',
                               'min_frequency', 'max_frequency', 'step']
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
                        try_to_add(
                            header_info['measurement_description'], key, value)

                    elif key in levelling_settings_keys:
                        try_to_add(
                            header_info['levelling_settings'], key, value)

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
        reformatted_segment_data = self._reformat_for_analysis(
            parsed_segment_data)
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
            raise KeyError(
                '{} not found in measurement description'.format(key))

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

            use_start = start
            use_end = end

            # first step needs to trim off the slow initial off
            # otherwise it will get confused when it tries
            # to find the fast off
            if i == 0:
                # look for second to last time it goes from stable to unstable
                # this should be where RF turns on for the first time in a segment
                rf_on_actually = np.diff(
                    self.raw_data['stable_samples'][use_start:use_end], append=0) < 0
                use_start = np.arange(use_start, use_end)[rf_on_actually][-2]+2

            # last step needs to trim off the slow starts at the end of the
            # segment so they dont confuse the step analyzer
            if i == len(step_start_indices) - 1:
                # cut off everything before the second stable period
                # so the stable samples can still be use to check the fast offs
                rf_on_actually = np.diff(
                    self.raw_data['stable_samples'][use_start:use_end], append=0) < 0
                try:
                    use_end = np.arange(use_start, use_end)[rf_on_actually][1]-1
                # if that fails just include the slow final off as part of the last step
                # it will make the plotting uglier but still work
                except IndexError:
                    use_end = use_end

            raw_data = {}
            for column in self.raw_data.keys():
                # print("column:", column)
                # print("column,start, end:", column, start, end)

                raw_data[column] = np.array(self.raw_data[column])[
                    int(use_start):int(use_end)]

            new_step = Step(self, raw_data, step_frequencies[i])
            if new_step.frequency == 8.0:
                pass
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
        try:
            self.results['RF_off_time'] = mid_t[switch]
        except Exception as e:
            raise e from e

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
                self.specs_metadata[input_signal_name] = {
                    'instrument_name': instr}
            except KeyError:
                print('NO SPEC CLASS FOR ', model, ' as ', role)

        print(type(self), analysis_config)

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
    """
    Analyzes thermopile signal in the microcalorimeter.

    """

    def __init__(
        self, analysis_config, signal_config, input_signal_config, instruments
    ):
        """Initialize a thermopile monitor."""
        SignalAnalyzer.__init__(
            self, analysis_config, signal_config, input_signal_config, instruments
        )

        self.column = self.input_signal_config['e']['column']
        try:
            self.instr_timing_tolerance = self.analysis_config['instr_timing_tolerance']
        except KeyError:
            self.instr_timing_tolerance = 5.0

        # incase you want to do fast off analysis
        try:
            self.fast_off_analysis = self.analysis_config['fast_off_analysis']
        except KeyError:
            self.fast_off_analysis = False

        if self.fast_off_analysis:
            self.V_off_delay = self.analysis_config['V_off_delay']
            self.V_off_function = self.analysis_config['V_off_function']
            self.V_off_fit_time_window = self.analysis_config['V_off_fit_time_window']

        try:
            self.stats_window_override = self.analysis_config['stats_window_override']
        except KeyError:
            self.stats_window_override = None

        # manually set the relative window location
        try:
            self.initial_off_window = self.analysis_config['initial_off_window']
        except KeyError:
            self.initial_off_window = None

        try:
            self.final_off_window = self.analysis_config['final_off_window']
        except KeyError:
            self.final_off_window = None

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

        slow_results = _analyze_off_period(
            segment,
            self.column,
            stats_window_override=self.stats_window_override,
            initial_off_window=self.initial_off_window,
            final_off_window=self.final_off_window
        )
        results.update(slow_results)

        # if thermometer column is in there then
        # need to parse that too so do a slow analysis
        # on the thermometer
        if self.has_thermometer():
            therm_v_slow_results = _analyze_off_period(
                segment,
                self.input_signal_config['therm_v']['column'],
                stats_window_override=self.stats_window_override,
                initial_off_window=self.initial_off_window,
                final_off_window=self.final_off_window
            )
            results.update(therm_v_slow_results)

            therm_i_slow_results = _analyze_off_period(
                segment,
                self.input_signal_config['therm_i']['column'],
                stats_window_override=self.stats_window_override,
                initial_off_window=self.initial_off_window,
                final_off_window=self.final_off_window
            )
            results.update(therm_i_slow_results)
            ...
        return results, metadata

    def has_thermometer(self):
        return 'therm_v' in self.input_signal_config

    def maybe_fast_off(
            self,
            step: Step,
            column: str,
            input_signal: str,
            do_fast_off:  bool = True
    ):

        results = {}
        segment = step.segment
        RF_off_time = step.results['RF_off_time']
        spec = self.specs[input_signal]

        on_results = _average_pre_fastoff(
            step,
            column,
            self.stats_window_override
        )

        on_val = on_results[column + '_on']
        results.update(on_results)

        off_slow_val = _linear(
            RF_off_time - segment.results['segment_time_zero'],
            segment.results[column + '_off_a'],
            segment.results[column + '_off_b'],
        )

        results[column + '_off_slow'] = off_slow_val

        # add specs

        results[column + '_off_slow' + '_spec'] = _get_spec_uncertainties(
            spec, off_slow_val
        )

        results[column + '_on' +
                '_spec'] = _get_spec_uncertainties(spec, on_val)

        metadata = {}
        metadata[column + '_on' + '_spec'] = {}
        metadata[column + '_off_slow' + '_spec'] = {}

        metadata[column + '_on' + '_spec']['instrument_name'] = (
            self.specs_metadata['e']['instrument_name']
        )
        metadata[column + '_off_slow' + '_spec']['instrument_name'] = (
            self.specs_metadata[input_signal]['instrument_name']
        )

        if do_fast_off:
            off_results = _fit_fast_off_timeseries(
                step,
                column,
                self.V_off_function,
                self.V_off_delay,
                self.instr_timing_tolerance,
                self.V_off_fit_time_window,
            )
            results.update(off_results)
            off_fast_val = results[column + '_off_fast']
            results[column + '_off_fast' + '_spec'] = _get_spec_uncertainties(
                spec, off_fast_val
            )
            metadata[column + '_off_fast' + '_spec'] = {}
            metadata[column + '_off_fast' + '_spec']['instrument_name'] = (
                self.specs_metadata[input_signal]['instrument_name']
            )
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
        results = {}
        metadata = {}

        # get the on values from each step, possibly a
        # fast off values depending on the config settings
        results_e, meta_e = self.maybe_fast_off(
            step,
            self.column,
            'e',
            do_fast_off=self.fast_off_analysis
        )

        results |= results_e
        metadata |= meta_e

        if self.has_thermometer():
            results_therm_v, meta_therm_v = self.maybe_fast_off(
                step,
                self.input_signal_config['therm_v']['column'],
                'therm_v',
                do_fast_off=False
            )
            results_therm_i, meta_therm_i = self.maybe_fast_off(
                step,
                self.input_signal_config['therm_i']['column'],
                'therm_i',
                do_fast_off=False
            )

            results |= results_therm_i | results_therm_v
            metadata |= meta_therm_i | meta_therm_v

        return results, metadata

    def plot_analysis(self, segment, *args) -> list[pl.Figure]:
        figures = []
        review = [(self.column, 'Thermopile Voltage', self.fast_off_analysis)]
        if self.has_thermometer():
            review += [
                 (self.input_signal_config['therm_i']['column'],'Thermometer Current (A)',False),
                 (self.input_signal_config['therm_v']['column'],'Thermometer Volts (V)',False),
                ]
        
        for column,title,fast in review:
            print(column)
            if fast:
                # print(self.column, 'fast')
                figures.append(_plot_fast_off_analysis(
                    column,
                    segment,
                    title,
                    self.V_off_function,
                    self.V_off_delay,
                    *args
                ))
            else:
                # print(self.column, 'slow')
                figures.append(_plot_slow_off_analysis(
                    column,
                    segment,
                    title,
                    *args
                ))
        return figures


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

        try:
            self.stats_window_override = self.analysis_config['stats_window_override']
        except KeyError:
            self.stats_window_override = None

        self.column = column
        self.instr_timing_tolerance = instr_timing_tolerance
        self.V_off_function = V_off_function
        self.V_off_delay = V_off_delay
        self.V_off_fit_time_window = V_off_fit_time_window

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
        results = _analyze_off_period(
            segment, self.column, stats_window_override=self.stats_window_override)
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

        on_results = _average_pre_fastoff(
            step,
            self.column,
            self.stats_window_override
        )

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
        results[self.column + '_on' +
                '_spec'] = _get_spec_uncertainties(spec, on_val)

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

    def plot_analysis(self, segment, *args):
        return _plot_fast_off_analysis(
            self.column,
            segment,
            'Bias Voltage (V)',
            self.V_off_function,
            self.V_off_delay,
            *args
        )


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

        self.i_column = i_column
        self.v_column = v_column
        self.instr_timing_tolerance = instr_timing_tolerance
        self.V_off_function = V_off_function
        self.V_off_delay = V_off_delay
        self.V_off_fit_time_window = V_off_fit_time_window

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
            step,
            self.v_column,
            self.stats_window_override,
        )
        on_i_results = _average_pre_fastoff(
            step, self.i_column,
            self.stats_window_override,
        )

        results.update(off_v_results)
        results.update(off_i_results)
        results.update(on_v_results)
        results.update(on_i_results)

        metadata = {}
        return results, metadata


class RFSourceAnalyzer(SignalAnalyzer):
    def __init__(
        self, analysis_config, signal_config, input_signal_config, instruments
    ):
        """
        Analyzes the RF Source Signal.

        Parameters
        ----------
        column : str
            Name of data column.

        Returns
        -------
        None.

        """
        SignalAnalyzer.__init__(
            self, analysis_config, signal_config, input_signal_config, instruments
        )
        self.source_column = self.input_signal_config['power']['column']

        try:
            self.am_voltage_column = self.input_signal_config['vdc']['column']
        except KeyError:
            self.am_voltage_column = None

        self.analysis_config = analysis_config
        self.signal_config = signal_config
        self.instruments = instruments
        self.instr_timing_tolerance = None
        try:
            self.stats_window_override = self.analysis_config['stats_window_override']
        except KeyError:
            self.stats_window_override = None


    def analyze_segment(self, segment: Segment) -> tuple:
        """
        Analyze segment.

        Does nothing for an RF source.

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
        results = {}
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

        def dBm_mean_func(x):
            return 10*np.log10(np.mean(10**(x/10)))

        def dBm_std_func(x):
            # return a small number if we are using 1 sample
            if len(x) == 1:
                return -1000
            return 10*np.log10(np.std(10**(x/10), ddof=1))

        # results for source
        results = _average_pre_fastoff(
            step,
            self.source_column,
            self.stats_window_override,
            mean_func=dBm_mean_func,
            std_func=dBm_std_func
        )

        # results for

        if self.am_voltage_column:
            results = results | _average_pre_fastoff(
                step,
                self.am_voltage_column,
                self.stats_window_override
            )

        return results, metadata

    def plot_analysis(self, segment: Segment, *args) -> pl.Figure:
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

        fig, ax = pl.subplots()
        # segment_results = segment.results
        segment_raw_data = segment.raw_data
        start_time = segment_raw_data[self.source_column + '_timestamp'][0]
        plot_time = segment_raw_data[self.source_column +
                                     '_timestamp'] - start_time
        source_setting = segment_raw_data[self.source_column]

        pl.plot(
            plot_time, source_setting, color=MAIN_TRACE_COLOR, linewidth=MAIN_TRACE_LINEWIDTH
        )

        len_step = len(segment.steps)
        for i, step_i in enumerate(segment.steps):

            plot_time_step = step_i.raw_data[self.source_column +
                                             '_timestamp'] - start_time
            setting_step = step_i.raw_data[self.source_column]

            initial_stable = step_i.results[self.source_column +
                                            '_initial_stable']
            final_stable = step_i.results[self.source_column + '_final_stable']
            step_line = pl.plot(
                plot_time_step,
                setting_step,
            )  # PLOT_COLORS[i % len(PLOT_COLORS)])
            # i want to plot what samples were used but doesn't seem to be working.
            label = None
            if i == len_step-1:
                label = 'On Samples'
            pl.plot(
                plot_time_step[initial_stable:final_stable],
                setting_step[initial_stable:final_stable],
                'x',
                label=label,
                linewidth=STABLE_TRACE_LINEWIDTH,
                color=MIDDLE_STABLE_TRACE_COLOR,
            )  # PLOT_COLORS[i % len(PLOT_COLORS)])

            label = None
            if i == len_step-1:
                label = 'On Average'
            pl.plot(
                plot_time_step[final_stable],
                step_i.results[self.source_column + '_on'],
                'o',
                label=label,
                linewidth=STABLE_TRACE_LINEWIDTH,
                color=MIDDLE_STABLE_TRACE_COLOR,
            )  # PLOT_COLORS[i % len(PLOT_COLORS)])
            # plot amplitude modulations section
            if self.am_voltage_column:
                final_setting = step_i.results[self.source_column + '_on']
                source_name = self.signal_config['power']['instrument']
                percent_per_volt = self.instruments[source_name][
                    'initial_settings']['AM_ext_sensitivity_percent_per_volt']
                am_step = step_i.raw_data[self.am_voltage_column]
                am_step_time = step_i.raw_data[self.am_voltage_column+'_timestamp']
                ind = am_step > 0
                am_step = am_step[ind]
                am_step_time = am_step_time[ind]

                adjusted_source = final_setting * \
                    (1 + am_step*percent_per_volt/100)

                label = None
                if i == len_step-1:
                    label = 'AM Adjusted Source'
                pl.plot(
                    am_step_time - start_time,
                    adjusted_source,
                    '-.',
                    color=step_line[0].get_color(),
                    linewidth=STABLE_TRACE_LINEWIDTH,
                )
        pl.legend(loc='best')
        pl.xlabel('Time (s)')
        pl.ylabel('RF Source Setting')
        return fig


class CommercialPowerMeterAnalyzer(SignalAnalyzer):
    def __init__(
        self, analysis_config, signal_config, input_signal_config, instruments
    ):
        """
        """
        SignalAnalyzer.__init__(
            self, analysis_config, signal_config, input_signal_config, instruments
        )
        self.column = self.input_signal_config['power']['column']
        try:
            self.instr_timing_tolerance = self.analysis_config['instr_timing_tolerance']
        except KeyError:
            self.instr_timing_tolerance = 5.0

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
        results = _analyze_off_period(
            segment, self.column, stats_window_override=self.stats_window_override)
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
        results = _average_pre_fastoff(
            step,
            self.column,
            self.stats_window_override
        )
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
        return _plot_slow_off_analysis(self.column, segment, 'Power (W)', *args)


def _plot_slow_off_analysis(
        column,
        segment: Segment,
        ylabel: str,
        *args) -> pl.Figure:
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
    fig, ax = pl.subplots()
    segment_results = segment.results
    segment_raw_data = segment.raw_data

    # specific results
    initial_off_start = segment_results[f'{column}_initial_off_start']
    initial_off_stop = segment_results[f'{column}_initial_off_stop']
    final_off_start = segment_results[f'{column}_final_off_start']
    final_off_stop = segment_results[f'{column}_final_off_stop']

    start_time = segment_raw_data[column + '_timestamp'][0]
    plot_time = segment_raw_data[column + '_timestamp'] - start_time
    raw = segment_raw_data[column]

    ax.plot(
        plot_time, raw, color=MAIN_TRACE_COLOR, linewidth=MAIN_TRACE_LINEWIDTH
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
        raw_step = step_i_raw_data[column]
        initial_stable = step_i.results[column + '_initial_stable']
        final_stable = step_i.results[column + '_final_stable']
        stable = np.logical_and(
            step_i.index >= initial_stable, step_i.index <= final_stable
        )

        ax.plot(
            plot_time_step,
            raw_step,
            linewidth=MAIN_TRACE_LINEWIDTH,
            color=PLOT_COLORS[i % len(PLOT_COLORS)],
        )
        ax.plot(
            plot_time_step[stable],
            raw_step[stable],
            linewidth=STABLE_TRACE_LINEWIDTH,
            color=MIDDLE_STABLE_TRACE_COLOR,
        )  # PLOT_COLORS[i % len(PLOT_COLORS)])

        RF_off_time = plot_time_step[final_stable]
        raw_on = step_i.results[column + '_on']
        # some columns dont distinguish between slow and fast,
        # so _off means _off_slow, or they just don't
        # have an off measurment in general
        try:
            raw_off = step_i.results[column + '_off_slow']
        except KeyError:
            try:
                raw_off = step_i.results[column + '_off']
            except KeyError:
                raw_off = None
        if raw_off is not None:
            ax.plot(
                [RF_off_time, RF_off_time],
                [raw_on, raw_off],
                marker='o',
                markersize=FIT_POINT_SIZE,
                color=PLOT_COLORS[i % len(PLOT_COLORS)],
            )

    ax.plot(
        plot_time[initial_off_start:initial_off_stop],
        raw[initial_off_start:initial_off_stop],
        color=INITIAL_STABLE_TRACE_COLOR,
        linewidth=STABLE_TRACE_LINEWIDTH,
    )
    ax.plot(
        plot_time[final_off_start:final_off_stop],
        raw[final_off_start:final_off_stop],
        color=FINAL_STABLE_TRACE_COLOR,
        linewidth=STABLE_TRACE_LINEWIDTH,
    )

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(ylabel)
    return fig


def _plot_fast_off_analysis(
        column: str,
        segment: Segment,
        ylabel: str,
        V_off_function: str,
        V_off_delay: float,
        *args) -> pl.Figure:
    """
    Generate plot of a data column in a segment that had fast off data.

    Parameters
    ----------
    segment : Segment
        Segment to analyze.

    Returns
    -------
    fig : matplotlib figure object
    """
    fig, ax = pl.subplots()
    segment_results = segment.results
    segment_raw_data = segment.raw_data
    start_time = segment_raw_data[column + '_timestamp'][0]
    plot_time = segment_raw_data[column + '_timestamp'] - start_time
    V_DVM = segment_raw_data[column]

    pl.plot(
        plot_time, V_DVM, color=MAIN_TRACE_COLOR, linewidth=MAIN_TRACE_LINEWIDTH
    )

    initial_off_start = segment_results[f'{column}_initial_off_start']
    initial_off_stop = segment_results[f'{column}_initial_off_stop']
    final_off_start = segment_results[f'{column}_final_off_start']
    final_off_stop = segment_results[f'{column}_final_off_stop']

    ax.plot(
        plot_time, V_DVM, color=MAIN_TRACE_COLOR, linewidth=MAIN_TRACE_LINEWIDTH
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
        label='Slow Off Fit',
        ls='-.',
        color=MIDDLE_STABLE_TRACE_COLOR,
        linewidth=STABLE_TRACE_LINEWIDTH,
    )

    slow_sample_line = ax.plot(
        plot_time[initial_off_start:initial_off_stop],
        V_DVM[initial_off_start:initial_off_stop],
        'v',
        label='Slow Off Samples',
        color=INITIAL_STABLE_TRACE_COLOR,
        linewidth=STABLE_TRACE_LINEWIDTH,
    )
    ax.plot(
        plot_time[final_off_start:final_off_stop],
        V_DVM[final_off_start:final_off_stop],
        'v',
        color=FINAL_STABLE_TRACE_COLOR,
        linewidth=STABLE_TRACE_LINEWIDTH,
    )

    len_step = len(segment.steps)
    for i, step_i in enumerate(segment.steps):
        off_time_delta = step_i.results[column + '_off_time_delta']
        plot_time_step = step_i.raw_data[column + '_timestamp'] - start_time
        V_DVM_step = step_i.raw_data[column]

        t_off = step_i.results['RF_off_time'] - start_time + off_time_delta
        initial_stable = step_i.results[column + '_initial_stable']
        final_stable = step_i.results[column + '_final_stable']

        stable = np.logical_and(
            step_i.index >= initial_stable, step_i.index <= final_stable
        )
        eval_time = t_off + V_off_delay

        V_off_slow = step_i.results[column + '_off_slow']
        V_off_fast = step_i.results[column + '_off_fast']
        V_on = step_i.results[column + '_on']

        plot_time_step = step_i.raw_data[column + '_timestamp'] - start_time
        V_DVM_step = step_i.raw_data[column]

        fit_region_time = step_i.raw_data[column + '_timestamp'][
            step_i.results['initial_fit_region']: step_i.results[
                'final_fit_region'
            ]
        ]
        fit_region_volts = step_i.raw_data[column][
            step_i.results['initial_fit_region']: step_i.results[
                'final_fit_region'
            ]
        ]
        fit_time_zero = step_i.results['fit_time_zero']
        fit_time = fit_region_time - fit_time_zero

        if V_off_function == 'lin_plus_exp':
            fit_volts = _lin_plus_exp(
                fit_time,
                step_i.results[column + '_fit_a'],
                step_i.results[column + '_fit_b'],
                step_i.results[column + '_fit_vscale'],
                step_i.results[column + '_fit_tscale'],
            )

        elif V_off_function == 'linear':
            fit_volts = _linear(
                fit_time,
                step_i.results[column + '_fit_a'],
                step_i.results[column + '_fit_b'],
            )

        if V_off_function == 'lin_plus_exp' or V_off_function == 'linear':
            pl.plot(
                fit_region_time - start_time,
                fit_volts,
                color=MIDDLE_STABLE_TRACE_COLOR,
            )
        pl.plot(plot_time_step, V_DVM_step, linewidth=STABLE_TRACE_LINEWIDTH)

        label = None
        if i == len_step-1:
            label = 'Fast Off Samples'
        pl.plot(
            fit_region_time - start_time,
            fit_region_volts,
            label=label,
            marker='o',
            ls='',
            color=FIT_TRACE_COLOR,
            markersize=RESAMPLE_POINTS_SIZE,
        )

        label = None
        if i == len_step-1:
            label = 'Off/On Summary'
        pl.plot(
            [t_off, t_off, t_off],
            [V_off_fast, V_off_slow, V_on],
            marker='*',
            linestyle='--',
            label=label,
            color=FIT_TRACE_COLOR,
            markersize=RESAMPLE_POINTS_SIZE,
            linewidth=STABLE_TRACE_LINEWIDTH
        )
        label = None
        if i == len_step-1:
            label = 'Fast Off'
        pl.plot(
            [eval_time],
            [V_off_fast],
            marker='D',
            ls='',
            label=label,
            color=FIT_TRACE_COLOR,
            markersize=RESAMPLE_POINTS_SIZE,
        )

        # i want to plot what samples were used but doesn't seem to be working.
        label = None
        if i == len_step-1:
            label = 'On Samples'
        pl.plot(
            plot_time_step[initial_stable:final_stable],
            V_DVM_step[initial_stable:final_stable],
            'x',
            label=label,
            linewidth=STABLE_TRACE_LINEWIDTH,
            color=MIDDLE_STABLE_TRACE_COLOR,
        )  # PLOT_COLORS[i % len(PLOT_COLORS)])

    pl.legend(loc='best')
    pl.xlabel('Time (s)')
    pl.ylabel(ylabel)
    return fig


def _average_pre_fastoff(
        step: Step,
        column: str,
        stats_window_override: float,
        mean_func: callable = None,
        std_func: callable = None
) -> dict:
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

    instr_timing_tolerance : float
        Max delay between RF off and instrument response.

    stats_window_override : float
        How long to average just prior to fast RF off in seconds.

    mean_func : callable | None
        Funciton to apply to the array of values to calculate the mean.
        By defualt np.mean. Must taken in an array and return the mean.

    std_func : callable | None
        Function to apply to the array of selected values to calculate
        the standard deviatio. Be defautl np.std with DDOF = 1. Must
        take in the array and return the standard deviation estimate.

    Returns
    -------
    results : dict
        column + '_on: average reading.
        column + '_on_dev': standard deviation of readings.

    """
    run = step.segment.run
    stats_window = run.parsed_config['stats_settings']['stats_window']
    if stats_window_override is not None:
        stats_window = stats_window_override

    results = {}
    timestamps = step.raw_data[column + '_timestamp']
    index = np.arange(len(timestamps))

    # use the last stable sample to find the off switch
    last_stable_sample = timestamps[step.raw_data['stable_samples'] == 1][-1]
    logical_index = np.logical_and(
        timestamps > last_stable_sample - stats_window,
        timestamps <= last_stable_sample
    )
    # throw away first value if can, sometimes on a transition
    # if moving fast and using every on value in the segment
    # if its a slow segment then it won't matter if we throw
    # away 1 sample of 100
    avg_time = timestamps[logical_index]
    start_offset = 0
    if len(avg_time) > 1:
        start_offset = 1

    use_vals = step.raw_data[column][logical_index][start_offset:]

    results[column + '_initial_stable'] = index[logical_index][start_offset]
    results[column + '_final_stable'] = index[logical_index][-1]

    if mean_func is None:
        mean_func = np.mean
    if std_func is None:
        std_func = partial(np.std, ddof=1)

    results[column + '_on'] = mean_func(use_vals)
    if len(use_vals) <= 2:
        results[column + '_on_dev'] = 0.0
    else:
        results[column + '_on_dev'] = std_func(use_vals)
    if VERBOSE:
        print(column, ' on', results[column + '_on'])

    return results


def _analyze_off_period(
    segment: Segment,
    column: str,
    stats_window_override: float | None = None,
    initial_off_window: list[float] = None,
    final_off_window: list[float] = None
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

    initial_off_window : tuple[float], optional
        Can be used to manually set the fitting window for the
        initial of period of the segment. Otherwise utitlized the
        stability of the calorimeter. Set relative to the first
        on point (i.e. [-1000, 0] selects points from 1000 seconds before
        the first on point up to the first on point.)

    final_off_window : tuple[float], optional
        Can be used to manually set the fitting window for the
        final off period of the segment. Otherwise the
        stability test of the calorimeter's thermopile signal
        and the stats window will be used to pick samples.
        Set relative to the last on point (i.e. [0, 1000] selects
        from the last on point plus 1000 seconds.

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
        print("step incomplete")
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
            segment.raw_data['timestamp'][initial_off_stable][-1] -
            stats_window_seconds
        )
  
        initial_off_start = index[
            segment.raw_data['timestamp'] >= initial_off_start_time 
        ][0]

        final_off_start_time = segment.raw_data['timestamp'][-1] - \
            stats_window_seconds
        final_off_start = index[
            np.logical_and((segment.raw_data['timestamp'] >= final_off_start_time),
            np.logical_not(segment.raw_data['power_on'])
            )
            ][
            0
        ]

        # if the manual override stats window is too big, use the
        # maximum value
        print(final_off_start, last_on_point)
        if final_off_start < last_on_point+1:
            final_off_start = last_on_point+1
            final_off_start_time = segment.raw_data['timestamp'][final_off_start]

        print(final_off_start, last_on_point)
        initial_off_stop = index[initial_off][-1]
        final_off_stop = index[final_off][-1]

        results[f'{column}_initial_off_start_time'] = initial_off_start_time
        results[f'{column}_final_off_start_time'] = final_off_start_time

        # if asked to, manually set the averaging window
        if initial_off_window is not None:
            # pick times relative to window
            initial_off_times = segment.raw_data[column +
                                                 '_timestamp'][initial_off]
            initial_off_ind = np.where(
                np.logical_and(
                    segment.raw_data['timestamp'] > initial_off_times[-1] +
                    initial_off_window[0],
                    segment.raw_data['timestamp'] < initial_off_times[-1] +
                    initial_off_window[1],
                )
            )[0]
            initial_off_start = initial_off_ind[0]
            initial_off_stop = initial_off_ind[-1]

        if final_off_window is not None:
            # pick times relative to window
            final_off_times = segment.raw_data[column +
                                               '_timestamp'][final_off]
            final_off_ind = np.where(
                np.logical_and(
                    segment.raw_data['timestamp'] > final_off_times[0] +
                    final_off_window[0],
                    segment.raw_data['timestamp'] < final_off_times[0] +
                    final_off_window[1],
                )
            )[0]
            final_off_start = final_off_ind[0]
            final_off_stop = final_off_ind[-1]

    elif type(run) is LegacyRun or type(run) is CrowleyRun:
        stats_window = run.parsed_config['stats_window']
        initial_off_start = index[initial_off_stable][0] - stats_window
        final_off_start = index[final_off_stable][0] - stats_window
        initial_off_stop = index[initial_off][-1]
        final_off_stop = index[final_off][-1]


    # sometimes it pre-fills the first value with a 
    # None. Why? only happens on the first segment I think?
    # and only at the start of a segment? I think it has something
    # to do with how the DatRecord tries to align time samples 
    # but I spent an hour trying to find where that happens and I couldnt
    # so I am doing a stupid check here. If I just move up 1 sample then it
    # doesn't catch the None it seems. I can not for the life of me figure out
    # what is happening.

    check_none_val = segment.raw_data[column][initial_off_start:initial_off_stop]
    if check_none_val[0] is None:
        initial_off_start +=1

    results[f'{column}_initial_off_start'] = initial_off_start
    results[f'{column}_final_off_start'] = final_off_start
    results[f'{column}_initial_off_stop'] = initial_off_stop
    results[f'{column}_final_off_stop'] = final_off_stop

    print(f'{column} fit indexes:')
    print('   initial: ', initial_off_start, initial_off_stop)
    print('     final: ', final_off_start, final_off_stop)

    results['segment_time_zero'] = segment.raw_data['timestamp'][0]

    # get off times (initial period, final period, and both combined)
    initial_off_times = segment.raw_data[column + '_timestamp'][
        initial_off_start:initial_off_stop
    ]
    final_off_times = segment.raw_data[column + '_timestamp'][
        final_off_start:final_off_stop
    ]
    off_times = (
        np.hstack((initial_off_times, final_off_times)) -
        results['segment_time_zero']
    )

    results['time_zero_' + column + '_i'] = initial_off_times[-1]
    results['time_zero_' + column + '_f'] = final_off_times[0]

    # get values of column
    initial_off_vals = segment.raw_data[column][initial_off_start:initial_off_stop]

    final_off_vals = segment.raw_data[column][final_off_start:final_off_stop]


  

    print(column, 'slow off initial fit time length',
          initial_off_times[-1] - initial_off_times[0])
    print(column, 'slow off final fit time length',
          final_off_times[-1] - final_off_times[0])
    vals = np.hstack((initial_off_vals, final_off_vals))



    # fit values of combined off regions
    try:
        # print(off_times)
        # print(vals)
        popt, pcov = scipy.optimize.curve_fit(_linear, off_times, vals)
        results[column + '_off_a'], results[column + '_off_b'] = popt
    except Exception as e:
        msg = f'Failed to fit  slow off period for {column} for "{e}".'
        fig,ax = pl.subplots(1,1)
        # ax.plot(segment.raw_data[column + '_timestamp'])
        # print(segment.raw_data[column + '_timestamp'].shape)
        # print(segment.raw_data[column].shape)
        # ax.plot(segment.raw_data[column + '_timestamp'], segment.raw_data[column])
        ax.plot(off_times, vals, 'o', label = 'All samples')
        ax.legend(loc = 'best')
        ax.set_xlabel('off times')
        ax.set_ylabel(column)
        fig.suptitle(f"Fitting Failure Report: \n {column} slow offs for segment")
        raise type(e)(msg) from e



    # fit initial off period bias slope and average
    # try:
    #     popt, pcov = scipy.optimize.curve_fit(_linear, initial_off_times - results["time_zero_" + column + "_i"], initial_off_vals)
    #     results[column + "_off_i_drift"], results[column + "_off_i"] = popt

    # except TypeError as e:
    #     print("error fitting ", column, "initial off period to line. stats_window is probably shorter than the sampling rate")
    #     print(e)
    #     results[column + "_off_i_drift"] = 0
    #     results[column + "_off_i"] = initial_off_vals[0]

    try:
        popt, pcov = scipy.optimize.curve_fit(
            _linear,
            initial_off_times - results['time_zero_' + column + '_i'],
            initial_off_vals,
        )
        results[column + '_off_i_drift'], results[column + '_off_i'] = popt
    except Exception as e:
        msg = f"Failed to fit initial off period drift for {column}. Likely not enough samples in stable period."
        raise type(e)(msg) from e

    # determine final slope and average of final off period
    try:
        popt, pcov = scipy.optimize.curve_fit(
            _linear, final_off_times -
            results['time_zero_' + column + '_f'], final_off_vals
        )
        results[column + '_off_f_drift'], results[column + '_off_f'] = popt
    except Exception as e:
        msg = f"Failed to fit final off period drift for {column}. Likely not enough samples in stable period."
        raise type(e)(msg) from e

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
    # throw away first point because it
    # sometimes is actually during the off state?
    timestamps = step.raw_data[column + '_timestamp'][1:]
    vals = step.raw_data[column][1:]
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
    delta = timestamps[i_step]-RF_off_time
    return delta


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
        (timestamps - RF_off_time) >= V_off_fit_time_window[0],
        (timestamps - RF_off_time) <= V_off_fit_time_window[1],
    )

    try:
        initial_fit_region = index[same_point][0]
        final_fit_region = index[same_point][-1]+1
        print(f"{column} fit region index: ",
              initial_fit_region, final_fit_region)

    except IndexError as e:

        print('Caught IndexError Trying to fit. Error for :')
        print('fit window = ', V_off_fit_time_window)
        print(
            'relative times (min,max)',
            (min(timestamps - RF_off_time), max(timestamps - RF_off_time)),
        )
        fig,ax = pl.subplots(1,1)
        ax.plot(timestamps, vals,'-', label = 'Fast Off Data')
        ax.axvline(RF_off_time)
        ax.legend(loc = 'best')
        ax.set_xlabel('Time Stamp')
        ax.set_ylabel(column)
        fig.suptitle(f'Fast Off Fitting Failure: {column}')
        raise IndexError from e

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
            V_off_fast = a * V_off_delay + b

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
            print(
                'Using single_sample analysis, but RF off region is longer than 1 sample, averaging: {} to {}'.format(
                    initial_fit_region, final_fit_region
                )
            )

        fit_time_zero = np.mean(timestamps[initial_fit_region])
        fit_eval_time = fit_time_zero
        a = vals[initial_fit_region]
        b = 0
        V_off_fast = np.mean(vals[initial_fit_region])

    else:
        raise ValueError(
            'Fitting function not recognized, Use "lin_plus_exp" or "linear" or "single_sample".'
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
