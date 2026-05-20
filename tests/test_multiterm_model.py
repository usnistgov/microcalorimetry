import microcalorimetry.analysis as anl
import microcalorimetry.configs as configs
import matplotlib.pyplot as plt
from microcalorimetry.math import rmemeas_extras
from pathlib import Path

LOCAL = Path(__file__).parents[0]
KSTE_REF_DATA = LOCAL / 'test_analysis_scripted_refs' / 'S24P02.h5'
THINFILM_REF_DATA = LOCAL / 'test_analysis_scripted_refs' / 'C24N118.h5'
S1P_FILES = LOCAL / 's1p_files'
C24N118_HISTORY = LOCAL / 'sample_historical_data' / 'C24N118'


def test_thermal_weights_test(make_plots: bool = False):
    # commonly used things
    splitter = S1P_FILES / '24splitter_proto.h5' / '24mm_clrmproto_splitter'

    # C24N118 calibration datas
    C24N118_data = {
        's11': configs.S11(S1P_FILES / 'C24N118.dut').load(),
        'parsed_calibration': configs.ParsedRFSweep(
            THINFILM_REF_DATA / 'C24N118' / 'rf_parsed'
        ),
        'clrm_coeffs': THINFILM_REF_DATA / 'C24S002' / 'k' / 'V_NVM (V)',
    }

    # C24N118 calibration datas
    S24P02_data = {
        's11': configs.S11(S1P_FILES / 'S24P02.dut').load(),
        'parsed_calibration': configs.ParsedRFSweep(
            KSTE_REF_DATA / 'S24P02' / 'parsed_rf'
        ),
        'clrm_coeffs': KSTE_REF_DATA / 'S24P02' / 'k' / 'V_NVM (V)',
    }

    gc_thinfilm_configs = {
        # specify a row
        'C24S002_C24N118': {
            # specifiy this one as an hdf5 files
            'splitter': splitter,
            # specify this one with a DUT file
            'special': {
                's11': S1P_FILES / 'C24S002.dut',
                'parsed_calibration': configs.ParsedRFSweep(
                    THINFILM_REF_DATA / 'C24S002' / 'rf_parsed'
                ),
                'clrm_coeffs': THINFILM_REF_DATA / 'C24S002' / 'k' / 'V_NVM (V)',
            },
            # specify this one with the actual data
            'standard': C24N118_data,
        },
        # specify a row
        'C24O002_C24N118': {
            # specifiy this one as an hdf5 files
            'splitter': splitter,
            # specify this one with a DUT file
            'special': {
                's11': S1P_FILES / 'C24O002.dut',
                'parsed_calibration': configs.ParsedRFSweep(
                    THINFILM_REF_DATA / 'C24O002' / 'rf_parsed'
                ),
                'clrm_coeffs': THINFILM_REF_DATA / 'C24S002' / 'k' / 'V_NVM (V)',
            },
            # specify this one with the actual data
            'standard': C24N118_data,
        },
    }
    gc_kste_configs = {
        # specify a row
        'S24S01_S24P02': {
            # specifiy this one as an hdf5 files
            'splitter': splitter,
            # specify this one with a DUT file
            'special': {
                's11': S1P_FILES / 'S24S01.dut',
                'parsed_calibration': configs.ParsedRFSweep(
                    KSTE_REF_DATA / 'S24S01' / 'parsed_rf'
                ),
                'clrm_coeffs': KSTE_REF_DATA / 'S24S01' / 'k' / 'V_NVM (V)',
            },
            # specify this one with the actual data
            'standard': S24P02_data,
        },
        # specify a row
        'S24S03_S24P02': {
            # specifiy this one as an hdf5 files
            'splitter': splitter,
            # specify this one with a DUT file
            'special': {
                's11': S1P_FILES / 'S24S03.dut',
                'parsed_calibration': configs.ParsedRFSweep(
                    KSTE_REF_DATA / 'S24S03' / 'parsed_rf'
                ),
                'clrm_coeffs': KSTE_REF_DATA / 'S24S03' / 'k' / 'V_NVM (V)',
            },
            # specify this one with the actual data
            'standard': S24P02_data,
        },
    }
    gc_kste_configs_mismatch = {
        # specify a row
        'S24S02_S24P02': {
            # specifiy this one as an hdf5 files
            'splitter': splitter,
            # specify this one with a DUT file
            'special': {
                's11': S1P_FILES / 'S24S02.dut',
                'parsed_calibration': configs.ParsedRFSweep(
                    KSTE_REF_DATA / 'S24S02' / 'parsed_rf'
                ),
                'clrm_coeffs': KSTE_REF_DATA / 'S24S02' / 'k' / 'V_NVM (V)',
            },
            # specify this one with the actual data
            'standard': S24P02_data,
        },
    }

    gc_configs = gc_thinfilm_configs | gc_kste_configs | gc_kste_configs_mismatch

    # %% 1 term correction factor

    gc1_thinfilm, figs = anl.make_correction_factor(
        gc_thinfilm_configs,
        correction_terms=1,
        make_plots=make_plots,
        calc_thermal_weights=False,
    )


    kc1_thinfilm, figs = anl.make_correction_factor(
        gc_thinfilm_configs,
        correction_terms=1,
        make_plots=make_plots,
        calc_thermal_weights=True,
    )

    gc3_kste_w_mismatch, figs = anl.make_correction_factor(
        gc_kste_configs | gc_kste_configs_mismatch,
        correction_terms=3,
        make_plots=make_plots,
        calc_thermal_weights=make_plots,
    )

    kc3_kste_w_mismatch, figs = anl.make_correction_factor(
        gc_kste_configs | gc_kste_configs_mismatch,
        correction_terms=3,
        make_plots=make_plots,
        calc_thermal_weights=True,
    )



if __name__ == '__main__':
    test_thermal_weights_test(make_plots=True)
