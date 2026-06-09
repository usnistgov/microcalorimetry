"""
This script is for testing review charts. It needs to be run manually.
"""

import microcalorimetry.configs
import microcalorimetry.measurements.rfsweep as rfsweep
import matplotlib.pyplot as plt

if __name__ == '__main__':
    plt.close('all')
    # this is an older dataset so the signal config needs to be defined
    # in the analysis config, normally this would be defined when you reun the
    # experiment.
    parsed, figs = rfsweep.parse(
        metadata='tests/sample_calruns/C24N118/calrun_1/20250226_metadata.csv',
        make_plots=True,
        plot_all_segments_analysis=True,
        analysis_config={
            'signal_config': {
                'DUT_power': {
                    'type': 'bolometer',
                    'units': 'W',
                    'can_level': True,
                    'input_signals': ['vdc'],
                    'resistance': 643.4,
                    'vdc': {'units': 'V', 'column': 'DVM_volts', 'instrument': 'DVM1'},
                },
                'RF_source_power': {
                    'type': 'RF_source',
                    'units': 'W',
                    'can_level': True,
                    'input_signals': ['power'],
                    'power': {
                        'units': 'dBm',
                        'column': 'rf_power_setting',
                        'instrument': 'DVM1',
                    },
                },
                'calorimeter_power': {
                    'type': 'thermoelectric',
                    'units': 'W',
                    'can_level': False,
                    'input_signals': ['e'],
                    'coeffs': 0.4,
                    'e': {'units': 'V', 'column': 'NVM_volts', 'instrument': 'NVM1'},
                },
            }
        },
        DUT_power_analysis={
            'fast_off_analysis': True,
            'V_off_delay': 0.36,
            'V_off_function': 'linear',
            'V_off_fit_time_window': [1, 10],
        },
        calorimeter_power_analysis={},
    )

    plt.show()

    pass
