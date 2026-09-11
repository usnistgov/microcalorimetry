import microcalorimetry.analysis as anl
import microcalorimetry.measurements.rfsweep as rfsweep
import microcalorimetry.measurements.dcsweep as dcsweep
import microcalorimetry.configs as configs
import os
import h5py
import io
from contextlib import redirect_stdout
from pathlib import Path
from rmellipse.utils import save_object
from copy import copy, deepcopy


LOCAL = Path(__file__).parents[0]
MUTABLE = LOCAL / 'mutable_datafiles'
SAMPLE_CALRUNS = LOCAL / 'sample_calruns'
THINFILM_MODEL = '8474E-K01'
KSTE_MODEL = 'KSTE'
SRUNS = LOCAL / 'sample_sensitivity_runs'
S1P_FILES = LOCAL / 's1p_files'
ETA_REFS = LOCAL / 'eta_references'

HISTORICAL_MODEL = LOCAL / 'sample_historical_data/model.h5/24histmodel'
HISTORICAL_DATA = configs.EtaHistorical(
    {
        '077': ETA_REFS / 'C24N118_077.eff',
        '078': ETA_REFS / 'C24N118_078.eff',
        '080': ETA_REFS / 'C24N118_080.eff',
        '082': ETA_REFS / 'C24N118_082.eff',
    }
)


def test_C24N118_from_scratch(
    resave_reference_results: Path = False,
    mutable_dir='C24N118',
    make_plots: bool = False,
):
    """
    This function calculates the Correction factor of C23N118 from
    some test data from start to finish. If asked, it will also
    save some intermediate data files to 'test_analysis_scripted'
    so that they can be reffered to by other scripts.

    Parameters
    ----------
    resave_reference_results : bool, optional
        If True, saves new referene data.
    mutable_dir: str
        name of directory to save throw-away results too
        (if any) inside the 'tests/mutable' directory which
        is ignored by git. By defualt 'C24N118'
    """

    dir = LOCAL / 'mutable_datafiles' / mutable_dir
    dir.mkdir(exist_ok=True, parents=True)

    reference_results = LOCAL / 'test_analysis_scripted_refs' / 'C24N118.h5'

    # parse the voltage steps
    metadata = SRUNS / 'C24S002_k_c000_r000' / 'voltage_stair_case_metadata.csv'
    dc_parsed, figures = dcsweep.parse_v0(
        metadata=metadata,
        measlist=metadata.parents[0] / 'meas_list.csv',
        transition_threshhold_watts=1e-4,
    )
    print(dc_parsed.keys())
    # make sensitivity coeffs for each NVM column
    C24S002_coeffs, fig = anl.make_k_coeffs(
        dc_parsed, p_of_e=False, deg=2, make_plots=False
    )

    if resave_reference_results:
        with h5py.File(reference_results, 'a') as f:
            g = f.require_group('C24S002')
            if 'k' in g:
                del g['k']
            save_object(g, 'k', C24S002_coeffs)
            if 'dc_parsed' in g:
                del g['dc_parsed']
            save_object(g, 'dc_parsed', dc_parsed)

    C24N118_config = {
        'analysis_config': {
            'DUT_power': {
                'instr_timing_tolerance': 5,
                'V_off_delay': 8,
                'V_off_function': 'linear',
                'V_off_fit_time_window': (1, 10),
                'RF_on_average_window': 1800,
                'RF_off_time_offset_method': 'max_change',
            },
            'monitor_power': {
                'instr_timing_tolerance': 5,
                'RF_on_average_window': 1800,
                'RF_off_time_offset_method': 'max_change',
            },
            'calorimeter_power': {
                'instr_timing_tolerance': 5,
                'RF_on_average_window': 1800,
                'RF_off_time_offset_method': 'first_data_point',
            },
        },
        'signal_config': {
            'DUT_power': {
                'vdc': {'column': 'DVM_volts', 'instrument': 'DVM1', 'units': 'V'},
                'input_signals': ['vdc'],
                'resistance': 943.3,
                'type': 'bolometer',
                'units': 'W',
                'can_level': True,
            },
            'monitor_power': {
                'power': {'column': 'P3 (W)', 'instrument': 'PM1', 'units': 'W'},
                'input_signals': ['power'],
                'type': 'commercial',
                'units': 'W',
                'can_level': True,
            },
            'calorimeter_power': {
                'e': {'column': 'NVM_volts', 'instrument': 'NVM1', 'units': 'V'},
                'input_signals': ['e'],
                'type': 'thermoelectric',
                'units': 'W',
                'can_level': True,
                'coeffs': C24S002_coeffs['V_NVM (V)'],
            },
            'RF_source_power': {
                'power': {
                    'column': 'rf_power_level_setting',
                    'instrument': 'RF_source',
                    'units': 'dBm',
                },
                'input_signals': ['power'],
                'type': 'rf_source',
                'units': 'W',
                'can_level': False,
            },
        },
    }

    C24N118_parsed = rfsweep.parse(
        [SAMPLE_CALRUNS / r'C24N118/calrun_1/20250226_metadata.csv'],
        analysis_config=C24N118_config,
        # fast_off_delay=0.1,
        # fast_off_fit_window=(5, 15),
        verbose=False,
        make_plots=False,
    )[0]

    if resave_reference_results:
        with h5py.File(reference_results, 'a') as f:
            g = f.require_group('C24N118')
            if 'rf_parsed' in g:
                del g['rf_parsed']
            save_object(g, 'rf_parsed', C24N118_parsed)

    # parse the calibration measurements
    C24O002_config = {
        'analysis_config': {
            # "DUT_power": {
            #     "instr_timing_tolerance": 5,
            #     "V_off_delay": 8,
            #     "V_off_function": "linear",
            #     "V_off_fit_time_window": (1, 10),
            #     "RF_on_average_window": 1800
            #     },
            'monitor_power': {
                'instr_timing_tolerance': 5,
                'RF_on_average_window': 1800,
                'RF_off_time_offset_method': 'max_change',
            },
            'calorimeter_power': {
                'instr_timing_tolerance': 5,
                'RF_on_average_window': 1800,
                'RF_off_time_offset_method': 'first_data_point',
            },
        },
        'signal_config': {
            'DUT_power': {
                'input_signals': [],
                'resistance': 943.3,
                'type': 'special',
                'units': 'W',
                'can_level': False,
            },
            'monitor_power': {
                'power': {'column': 'P3 (W)', 'instrument': 'PM1', 'units': 'W'},
                'input_signals': ['power'],
                'type': 'commercial',
                'units': 'W',
                'can_level': True,
            },
            'calorimeter_power': {
                'e': {'column': 'NVM_volts', 'instrument': 'NVM1', 'units': 'V'},
                'input_signals': ['e'],
                'type': 'thermoelectric',
                'units': 'W',
                'can_level': True,
                'coeffs': C24S002_coeffs['V_NVM (V)'],
            },
            'RF_source_power': {
                'power': {
                    'column': 'rf_power_level_setting',
                    'instrument': 'RF_source',
                    'units': 'dBm',
                },
                'input_signals': ['power'],
                'type': 'rf_source',
                'units': 'W',
                'can_level': False,
            },
        },
    }

    C24O002_parsed = rfsweep.parse(
        [SAMPLE_CALRUNS / r'C24O002/c001_run_0/20250317_metadata.csv'],
        analysis_config=C24O002_config,
        verbose=False,
        make_plots=False,
    )[0]

    if resave_reference_results:
        with h5py.File(reference_results, 'a') as f:
            g = f.require_group('C24O002')
            if 'rf_parsed' in g:
                del g['rf_parsed']
            save_object(g, 'rf_parsed', C24O002_parsed)

    # parse the calibration measurements
    C24S002_config = {
        'analysis_config': {
            # "DUT_power": {
            #     "instr_timing_tolerance": 5,
            #     "V_off_delay": 8,
            #     "V_off_function": "linear",
            #     "V_off_fit_time_window": (1, 10),
            #     "RF_on_average_window": 1800
            #     },
            'monitor_power': {
                'instr_timing_tolerance': 5,
                'RF_on_average_window': 1800,
                'RF_off_time_offset_method': 'max_change',
            },
            'calorimeter_power': {
                'instr_timing_tolerance': 5,
                'RF_on_average_window': 1800,
                'RF_off_time_offset_method': 'first_data_point',
            },
        },
        'signal_config': {
            'DUT_power': {
                'input_signals': [],
                'resistance': 943.3,
                'type': 'special',
                'units': 'W',
                'can_level': False,
            },
            'monitor_power': {
                'power': {'column': 'P3 (W)', 'instrument': 'PM1', 'units': 'W'},
                'input_signals': ['power'],
                'type': 'commercial',
                'units': 'W',
                'can_level': True,
            },
            'calorimeter_power': {
                'e': {'column': 'NVM_volts', 'instrument': 'NVM1', 'units': 'V'},
                'input_signals': ['e'],
                'type': 'thermoelectric',
                'units': 'W',
                'can_level': True,
                'coeffs': C24S002_coeffs['V_NVM (V)'],
            },
            'RF_source_power': {
                'power': {
                    'column': 'rf_power_level_setting',
                    'instrument': 'RF_source',
                    'units': 'dBm',
                },
                'input_signals': ['power'],
                'type': 'rf_source',
                'units': 'W',
                'can_level': False,
            },
        },
    }

    C24S002_parsed = rfsweep.parse(
        [SAMPLE_CALRUNS / r'C24S002/c001_run_0/20250313_metadata.csv'],
        analysis_config=C24S002_config,
        verbose=False,
        make_plots=False,
    )[0]

    if resave_reference_results:
        with h5py.File(reference_results, 'a') as f:
            g = f.require_group('C24S002')
            if 'rf_parsed' in g:
                del g['rf_parsed']
            save_object(g, 'rf_parsed', C24S002_parsed)

    # load in s-parameters
    # and configure a correction factor analysis
    C24N118_s1p_config = configs.S11Like(Path(S1P_FILES / 'C24N118.dut'))
    gc_config = configs.CorrectionFactorModelInputs(
        {
            # specify a row
            'C24S002_c001': {
                # specifiy this one as an hdf5 files
                'splitter': S1P_FILES
                / '24splitter_proto.h5'
                / '24mm_clrmproto_splitter',
                # specify this one with a DUT file
                'special': {
                    's11': S1P_FILES / 'C24S002.dut',
                    'parsed_calibration': C24S002_parsed,
                    'clrm_coeffs': C24S002_coeffs['V_NVM (V)'],
                },
                # specify this one with the actual data
                'standard': {
                    's11': C24N118_s1p_config,
                    'parsed_calibration': C24N118_parsed,
                    'clrm_coeffs': C24S002_coeffs['V_NVM (V)'],
                },
            },
            # specify a row
            'C24O002_c001': {
                # specifiy this one as an hdf5 files
                'splitter': S1P_FILES
                / '24splitter_proto.h5'
                / '24mm_clrmproto_splitter',
                # specify this one with a DUT file
                'special': {
                    's11': S1P_FILES / 'C24O002.dut',
                    'parsed_calibration': C24O002_parsed,
                    'clrm_coeffs': C24S002_coeffs['V_NVM (V)'],
                },
                # specify this one with the actual data
                'standard': {
                    's11': C24N118_s1p_config,
                    'parsed_calibration': C24N118_parsed,
                    'clrm_coeffs': C24S002_coeffs['V_NVM (V)'],
                },
            },
        }
    )
    gc, figures = anl.make_correction_factor(gc_config, 1, make_plots=True)

    if resave_reference_results:
        with h5py.File(reference_results, 'a') as f:
            if 'gc' in f:
                del f['gc']
            save_object(f, 'gc', gc)

    # calculate the effective efficiency
    fig, eta = anl.make_eta(
        gc,
        C24N118_s1p_config,
        C24N118_parsed,
        historical_data=HISTORICAL_DATA,
        make_plots=True,
    )

    if resave_reference_results:
        with h5py.File(reference_results, 'a') as f:
            if 'eta' in f:
                del f['eta']
            save_object(f, 'eta', eta)

    # Check if the file exists before attempting to remove it
    save_path = MUTABLE / 'test_C24N118_from_scratch.h5'
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"File '{save_path}' removed successfully.")

    with h5py.File(save_path, 'a') as f:
        if eta.name in f:
            del f[eta.name]
        save_object(f, eta.name, eta)


def test_S24P02_from_scratch(
    make_plots=False,
    resave_reference_results: Path = False,
):
    """
    This function calculates the Correction factor of C23N118 from
    some test data from start to finish. If asked, it will also
    save some intermediate data files to 'test_analysis_scripted'
    so that they can be reffered to by other scripts.

    Parameters
    ----------
    make_plots : bool, optional
        The default is False,
    resave_reference_results : bool, optional
        If True, saves new referene data.
    """

    # this is where reference data is saved
    # reference data gets used in other test scripts,
    # or for comparing newly calculated results to make sure
    # analysis code hasn't broken
    reference_results = LOCAL / 'test_analysis_scripted_refs' / 'S24P02.h5'

    # config used for parsing the load sensor
    # this isn't included in the older data files
    # so needs to be re-written here.
    load_rfparser_config = RF_PARSER_CONFIG = {
        'analysis_config': {
            'DUT_power': {
                'instr_timing_tolerance': 5,
                'RF_on_average_window': 1800,
                # needs to be replaced with coeffs after they are calculated
                'coeffs': None,
                'RF_off_time_offset_method': 'max_change',
            },
            'monitor_power': {
                'instr_timing_tolerance': 5,
                'RF_on_average_window': 1800,
                'RF_off_time_offset_method': 'max_change',
            },
            'calorimeter_power': {
                'instr_timing_tolerance': 5,
                'RF_on_average_window': 1800,
                'RF_off_time_offset_method': 'first_data_point',
            },
        },
        'signal_config': {
            'DUT_power': {
                'e': {'column': 'PM2_NVM (V)', 'instrument': 'PM2', 'units': 'V'},
                'input_signals': ['e'],
                'type': 'thermoelectric',
                'units': 'W',
                'can_level': True,
                # needs to be replaced with coeffs after they are calculated
                'coeffs': None,
            },
            'monitor_power': {
                'power': {'column': 'P3 (W)', 'instrument': 'PM3', 'units': 'W'},
                'input_signals': ['power'],
                'type': 'commercial',
                'units': 'W',
                'can_level': True,
            },
            'calorimeter_power': {
                'e': {'column': 'NVM_volts', 'instrument': 'NVM1', 'units': 'V'},
                'input_signals': ['e'],
                'type': 'thermoelectric',
                'units': 'W',
                'can_level': True,
                # needs to be replaced with coeffs after they are calculated
                'coeffs': None,
            },
            'RF_source_power': {
                'power': {
                    'column': 'rf_power_setting',
                    'instrument': 'RF_source',
                    'units': 'dBm',
                },
                'input_signals': ['power'],
                'type': 'rf_source',
                'units': 'W',
                'can_level': False,
            },
        },
        # ths instrument seems to be missing info
        # I think because the data was taken on a proto-type build for the
        # thermoelecctircs where the model meta data wasn't very consistent.
        # And the keysight sensors were doing acive and passive monitoring at
        # the same time to try and compare them. So, this info is being overloaded
        # here, only relevant for the S24P02 (load) sensor.
        'instruments': {
            'PM2': {'model': 'HP34420A', 'serial': 'XXX', 'role': 'thermopile_monitor'}
        },
    }

    # modify the DUT configuration (remove from analysis and adjust signal)
    # for the reflect signa;s
    reflect_rfparser_config = deepcopy(load_rfparser_config)
    reflect_rfparser_config['analysis_config'].pop('DUT_power')
    # special sensor isn't analyze byt he parser, so this is just
    # here so the validation that happens (originally meant for the runner)
    # doesn't scream that there isn't a DUT signal defined.
    reflect_rfparser_config['signal_config']['DUT_power'] = {
        'input_signals': [],
        'type': 'special',
        'units': 'W',
        'can_level': False,
    }
    # I used different sensor names for the monitor here
    # again, sorrry, was workshopping the prototype at this time
    # sub signal is coming from PM1, which is the RNS commercial power meter
    reflect_rfparser_config['signal_config']['monitor_power']['power']['instrument'] = (
        'PM1'
    )

    # Points to the RF and DC metadata needed for each sensor
    big_config = {
        'S24P02': {
            'is_load': True,
            'dc_sweep_metadata': LOCAL
            / Path(
                r'sample_sensitivity_runs/S24P02_c000_krun__2/c000_krun__metadata.csv'
            ),
            'rf_sweep_metadata': LOCAL
            / Path(Path(r'sample_calruns/S24P02/c000_gc_r000/20250627_metadata.csv')),
        },
        'S24S03': {
            'is_load': False,
            'dc_sweep_metadata': LOCAL
            / Path(
                r'sample_sensitivity_runs/S24S03_c000_krun__1/c000_krun__metadata.csv'
            ),
            'rf_sweep_metadata': [
                LOCAL
                / Path(r'sample_calruns/S24S03/c000_calrun_2/20250512_metadata.csv'),
                LOCAL
                / Path(r'sample_calruns/S24S03/c000_calrun_1/20250505_metadata.csv'),
            ],
        },
        'S24S02': {
            'is_load': False,
            'dc_sweep_metadata': LOCAL
            / Path(
                r'sample_sensitivity_runs/S24S02_c000_krun__0/c000_krun__metadata.csv'
            ),
            'rf_sweep_metadata': LOCAL
            / Path(r'sample_calruns/S24S02/c000_calrun_0/20250415_metadata.csv'),
        },
        'S24S01': {
            'is_load': False,
            'dc_sweep_metadata': LOCAL
            / Path(
                r'sample_sensitivity_runs/S24S01_c000_krun__7/c000_krun__metadata.csv'
            ),
            'rf_sweep_metadata': [
                LOCAL
                / Path(r'sample_calruns/S24S01/c000_calrun_1/20250404_metadata.csv'),
                LOCAL
                / Path(r'sample_calruns/S24S01/c000_calrun_2/20250409_metadata.csv'),
            ],
        },
    }

    # this parses the RF and DC sweeps
    # the thermoelectric fit coefficients are needed for parsing
    # the RF data, so including that here too as well.
    results = {}
    for sensor, sweep_configs in big_config.items():
        print('Parsing ', sensor)
        print('--------')

        # suppresing stdout because it is noisy
        # only warnings and errors get through here.
        output_capture = io.StringIO()
        with redirect_stdout(output_capture):
            # parse the DC sweep and calculate the sensitivity
            parsed_dc, figs = dcsweep.parse(
                Path(sweep_configs['dc_sweep_metadata']),
                repeatability_id=Path(sweep_configs['dc_sweep_metadata']).stem,
            )

            k, figs = anl.make_k_coeffs(
                parsed_dc,
                constrain_zero=False,
                p_of_e=False,
                deg=2,
                make_plots=True,
            )

            # assign the coefficients we just calculated
            # to the DUT and calorimeter thermopile
            if sweep_configs['is_load']:
                config = deepcopy(load_rfparser_config)
                config['signal_config']['DUT_power']['coeffs'] = k['V_NVM_sensor (V)']
            else:
                config = deepcopy(reflect_rfparser_config)

            # config
            config['signal_config']['calorimeter_power']['coeffs'] = k['V_NVM (V)']

            parsed_rf, figs = rfsweep.parse(
                sweep_configs['rf_sweep_metadata'],
                make_plots=False,
                analysis_config=config,
            )

            # save data to a reference file if asked to
            results[sensor] = {'k': k, 'parsed_rf': parsed_rf, 'parsed_dc': parsed_dc}

            # save results if asked too
            if resave_reference_results:
                with h5py.File(reference_results, 'a') as f:
                    if sensor in f:
                        del f[sensor]
                    save_object(f, sensor, results[sensor])

    # build a correction factor matric
    # standard data doesn't change between rows
    standard_data = {
        's11': S1P_FILES / 'S24P02.dut',
        'parsed_calibration': results['S24P02']['parsed_rf'],
        'clrm_coeffs': results['S24P02']['k']['V_NVM (V)'],
    }

    splitter_s11 = S1P_FILES / '24splitter_proto.h5' / '24mm_clrmproto_splitter'
    # don't use the S24S02 data for a 1 term model because
    # it doesn't work very well like that (mismatched is to close to zero)
    # but it is still being added to the reference data on the output
    # of this script so it can be used in other test scripts
    gc_config = configs.CorrectionFactorModelInputs(
        {
            # # specify a row
            'S24S01': {
                # specifiy this one as an hdf5 files
                'splitter': splitter_s11,
                # specify this one with a DUT file
                'special': {
                    's11': S1P_FILES / 'S24S01.dut',
                    'parsed_calibration': results['S24S01']['parsed_rf'],
                    'clrm_coeffs': results['S24S01']['k']['V_NVM (V)'],
                },
                'standard': standard_data,
            },
            # # specify a row
            # 'S24S02': {
            #     # specifiy this one as an hdf5 files
            #     'splitter':splitter_s11,
            #     # specify this one with a DUT file
            #     'special': {
            #         's11': S1P_FILES / 'S24S02.dut',
            #         'parsed_calibration': results['S24S02']['parsed_rf'],
            #         'clrm_coeffs': results['S24S02']['k']['V_NVM (V)'],
            #     },
            #     'standard': standard_data
            # },
            # specify a row
            'S24S03': {
                # specifiy this one as an hdf5 files
                'splitter': splitter_s11,
                # specify this one with a DUT file
                'special': {
                    's11': S1P_FILES / 'S24S03.dut',
                    'parsed_calibration': results['S24S03']['parsed_rf'],
                    'clrm_coeffs': results['S24S03']['k']['V_NVM (V)'],
                },
                'standard': standard_data,
            },
        }
    )

    gc, figures = anl.make_correction_factor(gc_config, 1, make_plots=make_plots)

    # save results if asked too
    if resave_reference_results:
        with h5py.File(reference_results, 'a') as f:
            if 'gc' in f:
                del f['gc']
            save_object(f, 'gc', gc)

    # calculate the effective efficiency
    fig, eta = anl.make_eta(
        gc,
        S1P_FILES / 'S24P02.dut',
        results['S24P02']['parsed_rf'],
        historical_data=None,
        make_plots=True,
    )

    eta = anl.apply_uncertainty_model(eta, HISTORICAL_MODEL)

    if resave_reference_results:
        with h5py.File(reference_results, 'a') as f:
            if 'eta' in f:
                del f['eta']
            save_object(f, 'eta', eta)


if __name__ == '__main__':
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # mpl.use('QtAgg')
    fig = test_C24N118_from_scratch(resave_reference_results=False, make_plots=True)

    fig = test_S24P02_from_scratch(
        # dont switch this to True unless you want to override reference data.
        resave_reference_results=False,
        make_plots=True,
    )
    # anl.review_eta(MUTABLE / "test_C24N118_from_scratch.h5/new_eta", HISTORICAL_DATA)
    plt.show()
