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
        make_plots=False,
        calc_thermal_weights=False,
    )

    gc1_kste, figs = anl.make_correction_factor(
        gc_kste_configs,
        correction_terms=1,
        make_plots=False,
        calc_thermal_weights=False,
    )

    gc1_kste_w_mismatch, figs = anl.make_correction_factor(
        gc_kste_configs | gc_kste_configs_mismatch,
        correction_terms=1,
        make_plots=False,
        calc_thermal_weights=False,
    )

    kc1_thinfilm, figs = anl.make_correction_factor(
        gc_thinfilm_configs,
        correction_terms=1,
        make_plots=False,
        calc_thermal_weights=True,
    )

    kc1_kste, figs = anl.make_correction_factor(
        gc_kste_configs, correction_terms=1, make_plots=False, calc_thermal_weights=True
    )

    kc1_kste_w_mismatch, figs = anl.make_correction_factor(
        gc_kste_configs | gc_kste_configs_mismatch,
        correction_terms=1,
        make_plots=False,
        calc_thermal_weights=True,
    )

    kc1_all, figs = anl.make_correction_factor(
        gc_kste_configs | gc_thinfilm_configs,
        correction_terms=1,
        make_plots=False,
        calc_thermal_weights=True,
    )

    kc1_all_w_mismatch, figs = anl.make_correction_factor(
        gc_kste_configs | gc_kste_configs_mismatch | gc_thinfilm_configs,
        correction_terms=1,
        make_plots=False,
        calc_thermal_weights=True,
    )

    gc3_kste_w_mismatch, figs = anl.make_correction_factor(
        gc_kste_configs | gc_kste_configs_mismatch,
        correction_terms=3,
        make_plots=False,
        calc_thermal_weights=False,
    )

    kc3_kste_w_mismatch, figs = anl.make_correction_factor(
        gc_kste_configs | gc_kste_configs_mismatch,
        correction_terms=3,
        make_plots=False,
        calc_thermal_weights=True,
    )

    kc3_all, figs = anl.make_correction_factor(
        gc_thinfilm_configs | gc_kste_configs,
        correction_terms=3,
        make_plots=False,
        calc_thermal_weights=True,
    )

    kc3_all_w_mismatch, figs = anl.make_correction_factor(
        gc_thinfilm_configs | gc_kste_configs | gc_kste_configs_mismatch,
        correction_terms=3,
        make_plots=False,
        calc_thermal_weights=True,
    )

    kc4_all, figs = anl.make_correction_factor(
        gc_thinfilm_configs | gc_kste_configs,
        correction_terms=4,
        make_plots=False,
        calc_thermal_weights=True,
    )

    kc4_all_w_mismatch, figs = anl.make_correction_factor(
        gc_thinfilm_configs | gc_kste_configs | gc_kste_configs_mismatch,
        correction_terms=4,
        make_plots=False,
        calc_thermal_weights=True,
    )

    if make_plots:
        # %%% Plot 1 term gc
        plt.close('1-term-gc')
        fig, ax = plt.subplots(1, 1, num='1-term-gc')
        weights = [gc1_thinfilm, gc1_kste, gc1_kste_w_mismatch]
        colors = ['r', 'b', 'g']
        names = ['TF', 'KSTE', 'KSTE (w/ Mismatch)']
        for w, color, model in zip(weights, colors, names):
            gred = rmemeas_extras.categorize_by(w, 'Origin')
            for i in range(1):
                ax.plot(
                    w.sel(gc=i).nom.frequency,
                    w.nom.sel(gc=i),
                    'o-',
                    color=color,
                    label=model,
                )
                lb = w.sel(gc=i, drop=True).uncbounds(k=-1).cov
                ub = w.sel(gc=i, drop=True).uncbounds(k=1).cov
                ax.fill_between(lb.frequency, lb, ub, alpha=0.2, color=color)
                ax.set_ylabel(f'$k_{{c{i}}}$')
                ax.set_xlabel('Frequency (GHz)')
                ax.legend(loc='best')
                ax.set_ylabel('Thermal Correction Factor $(k_{c1})$')
        ax.legend(loc='best')
        ax.set_title('1-Term Correction Factor Model')
        plt.show()

        plt.close('1-term-weights')
        fig, ax = plt.subplots(1, 1, num='1-term-weights')
        weights = [
            kc1_thinfilm,
            kc1_kste,
            kc1_kste_w_mismatch,
            kc1_all,
            kc1_all_w_mismatch,
        ]
        colors = ['r', 'b', 'g', 'm', 'c']
        names = [
            'TF',
            'KSTE',
            'KSTE (w/ Mismatch)',
            'KSTE + TF',
            'KSTE (w/ Mismatch) + TF',
        ]
        for w, color, model in zip(weights, colors, names):
            gred = rmemeas_extras.categorize_by(w, 'Origin')
            for i in range(1):
                ax.plot(
                    w.sel(gc=i).nom.frequency,
                    w.nom.sel(gc=i),
                    'o-',
                    color=color,
                    label=model,
                )
                lb = w.sel(gc=i, drop=True).uncbounds(k=-1).cov
                ub = w.sel(gc=i, drop=True).uncbounds(k=1).cov
                ax.fill_between(lb.frequency, lb, ub, alpha=0.2, color=color)
                ax.set_ylabel(f'$k_{{c{i}}}$')
                ax.set_xlabel('Frequency (GHz)')
                ax.legend(loc='best')
                ax.set_ylabel('Thermal Correction Factor $(k_{c1})$')
        ax.legend(loc='best')
        ax.set_title('1-Term Thermally Weighted Correction Factor Model')
        plt.show()

        # %% Compare 1 term model calculated by 1 sensor, evaluated on another
        one_term_model = configs.Eta(THINFILM_REF_DATA / 'eta').load()
        historical = {'1 Term Model (new value)': one_term_model}
        for f in C24N118_HISTORY.iterdir():
            historical[f.stem] = f
        figs, C24N118_eta_1term = anl.make_eta(
            kc1_kste_w_mismatch,
            C24N118_data['s11'],
            C24N118_data['parsed_calibration'],
            thermal_weights=THINFILM_REF_DATA / 'C24S002' / 'k' / 'V_NVM (V)',
            historical_data=historical,
        )
        figs[0].suptitle('C24N118 calculated using KSTE (w/ Mismatch) $k_{c1}$')

        one_term_model = configs.Eta(THINFILM_REF_DATA / 'eta').load()
        historical = {'1 Term Model (new value)': one_term_model}
        for f in C24N118_HISTORY.iterdir():
            historical[f.stem] = f
        figs, C24N118_eta_1term = anl.make_eta(
            kc1_all_w_mismatch,
            C24N118_data['s11'],
            C24N118_data['parsed_calibration'],
            thermal_weights=THINFILM_REF_DATA / 'C24S002' / 'k' / 'V_NVM (V)',
            historical_data=historical,
        )
        figs[0].suptitle('C24N118 calculated using KSTE (w/ Mismatch) + TF $k_{c1}$')

        # %%% Plot correction factors

        weights = [gc3_kste_w_mismatch]
        colors = ['r', 'b', 'g']
        names = ['KSTE (w/mismatch)']

        plt.close('3-term-gc')
        fig, ax = plt.subplots(3, 1, num='3-term-gc', sharex=True, figsize=(6, 11))
        ax = ax.flatten()
        for w, color, model in zip(weights, colors, names):
            gred = rmemeas_extras.categorize_by(w, 'Origin')
            for i in range(3):
                ax[i].plot(
                    w.sel(gc=i).nom.frequency,
                    w.nom.sel(gc=i),
                    'o-',
                    color=color,
                    label=model,
                )
                lb = w.sel(gc=i, drop=True).uncbounds(k=-1).cov
                ub = w.sel(gc=i, drop=True).uncbounds(k=1).cov
                ax[i].fill_between(lb.frequency, lb, ub, alpha=0.2, color=color)
                ax[i].set_ylabel(f'$g_{{c{i + 1}}}$')
                ax[i].set_xlabel('Frqeuency (GHz)')
            h, l = ax[0].get_legend_handles_labels()
        ax[0].set_ylim((-0.15, 0.2))
        fig.legend(
            h, l, loc='upper right', bbox_to_anchor=(0.5, 0.9)
        )  # Example placement
        fig.suptitle('3-Term Correction Factor Model')
        plt.show()

        # %%% Plot thermally weighted correction factors

        weights = [kc3_kste_w_mismatch, kc3_all, kc3_all_w_mismatch]
        colors = ['r', 'b', 'g']
        names = ['KSTE (w/mismatch)', 'KSTE + TF', 'KSTE (w/ Mismatch) + TF']

        plt.close('3-term-gc')
        fig, ax = plt.subplots(3, 1, num='3-term-gc', sharex=True, figsize=(6, 11))
        ax = ax.flatten()
        for w, color, model in zip(weights, colors, names):
            gred = rmemeas_extras.categorize_by(w, 'Origin')
            for i in range(3):
                ax[i].plot(
                    w.sel(gc=i).nom.frequency,
                    w.nom.sel(gc=i),
                    'o-',
                    color=color,
                    label=model,
                )
                lb = w.sel(gc=i, drop=True).uncbounds(k=-1).cov
                ub = w.sel(gc=i, drop=True).uncbounds(k=1).cov
                ax[i].fill_between(lb.frequency, lb, ub, alpha=0.2, color=color)
                ax[i].set_ylabel(f'$g_{{c{i}}}$')

            h, l = ax[0].get_legend_handles_labels()

        fig.legend(
            h, l, loc='lower center', bbox_to_anchor=(0.5, 0.1)
        )  # Example placement
        fig.suptitle('3-Term Thermally Weighted Correction Factor Model')
        plt.show()

        weights = [kc3_all, kc3_all_w_mismatch]
        colors = ['b', 'g']
        names = ['KSTE + TF', 'KSTE (w/ Mismatch) + TF']

        plt.close('3-term-kc')
        fig, ax = plt.subplots(3, 1, num='3-term-kc', sharex=True, figsize=(6, 11))
        ax = ax.flatten()
        for w, color, model in zip(weights, colors, names):
            gred = rmemeas_extras.categorize_by(w, 'Origin')
            for i in range(3):
                ax[i].plot(
                    w.sel(gc=i).nom.frequency,
                    w.nom.sel(gc=i),
                    'o-',
                    color=color,
                    label=model,
                )
                lb = w.sel(gc=i, drop=True).uncbounds(k=-1).cov
                ub = w.sel(gc=i, drop=True).uncbounds(k=1).cov
                ax[i].fill_between(lb.frequency, lb, ub, alpha=0.2, color=color)
                ax[i].set_ylabel(f'$g_{{c{i}}}$')

        ax[0].plot(
            kc1_all_w_mismatch.sel(gc=0).nom.frequency,
            kc1_all_w_mismatch.nom.sel(gc=0),
            'o-',
            color='c',
            label='1-Term KSTE (w/ Mismatch) + TF',
        )
        lb = kc1_all_w_mismatch.sel(gc=0, drop=True).uncbounds(k=-1).cov
        ub = kc1_all_w_mismatch.sel(gc=0, drop=True).uncbounds(k=1).cov
        ax[0].fill_between(lb.frequency, lb, ub, alpha=0.2, color=color)
        h, l = ax[0].get_legend_handles_labels()
        fig.legend(
            h, l, loc='lower center', bbox_to_anchor=(0.5, 0.1)
        )  # Example placement
        fig.suptitle('3-Term Thermally Weighted Correction Factor Model')
        plt.show()

        # %% Compare 3 term model to historical 1 term model data
        # based on different thermal weight calculations

        one_term_model = configs.Eta(THINFILM_REF_DATA / 'eta').load()
        historical = {'1 Term Model (new value)': one_term_model}
        for f in C24N118_HISTORY.iterdir():
            historical[f.stem] = f
        figs, C24N118_eta_3term = anl.make_eta(
            kc3_all,
            C24N118_data['s11'],
            C24N118_data['parsed_calibration'],
            thermal_weights=THINFILM_REF_DATA / 'C24S002' / 'k' / 'V_NVM (V)',
            historical_data=historical,
        )
        figs[0].suptitle('C24N118 calculated using KSTE +TF $k_{ci} 3-Terms$')

        one_term_model = configs.Eta(THINFILM_REF_DATA / 'eta').load()
        historical = {'1 Term Model (new value)': one_term_model}
        for f in C24N118_HISTORY.iterdir():
            historical[f.stem] = f
        figs, C24N118_eta_3term = anl.make_eta(
            kc3_all_w_mismatch,
            C24N118_data['s11'],
            C24N118_data['parsed_calibration'],
            thermal_weights=THINFILM_REF_DATA / 'C24S002' / 'k' / 'V_NVM (V)',
            historical_data=historical,
        )
        figs[0].suptitle(
            'C24N118 calculated using KSTE (w/ Mismatch) +TF $k_{ci} 3-Terms$'
        )

        # %% plot 4 term weights
        weights = [kc4_all, kc4_all_w_mismatch]
        colors = ['b', 'r']
        names = ['KSTE + TF', 'KSTE (w/ Mismatch) + TF']

        plt.close('3-term-gc')
        fig, ax = plt.subplots(2, 2, num='3-term-gc', sharex=True, figsize=(6, 11))
        ax = ax.flatten()
        for w, color, model in zip(weights, colors, names):
            gred = rmemeas_extras.categorize_by(w, 'Origin')
            for i in range(4):
                ax[i].plot(
                    w.sel(gc=i).nom.frequency,
                    w.nom.sel(gc=i),
                    'o-',
                    color=color,
                    label=model,
                )
                lb = w.sel(gc=i, drop=True).uncbounds(k=-1).cov
                ub = w.sel(gc=i, drop=True).uncbounds(k=1).cov
                ax[i].fill_between(lb.frequency, lb, ub, alpha=0.2, color=color)
                ax[i].set_ylabel(f'$g_{{c{i}}}$')

            h, l = ax[0].get_legend_handles_labels()
        fig.legend(
            h, l, loc='lower center', bbox_to_anchor=(0.5, 0.1)
        )  # Example placement
        fig.suptitle('4-Term Thermally Weighted Correction Factor Model')
        plt.show()

        # %% compare thinfilm
        one_term_model = configs.Eta(THINFILM_REF_DATA / 'eta').load()
        historical = {'1 Term Model (new value)': one_term_model}
        for f in C24N118_HISTORY.iterdir():
            historical[f.stem] = f
        figs, C24N118_eta_3term = anl.make_eta(
            kc4_all,
            C24N118_data['s11'],
            C24N118_data['parsed_calibration'],
            thermal_weights=THINFILM_REF_DATA / 'C24S002' / 'k' / 'V_NVM (V)',
            historical_data=historical,
        )
        figs[0].suptitle('C24N118 calculated using KSTE +TF $k_{ci} 4-Terms$')

        one_term_model = configs.Eta(THINFILM_REF_DATA / 'eta').load()
        historical = {'1 Term Model (new value)': one_term_model}
        for f in C24N118_HISTORY.iterdir():
            historical[f.stem] = f
        figs, C24N118_eta_3term = anl.make_eta(
            kc4_all,
            C24N118_data['s11'],
            C24N118_data['parsed_calibration'],
            thermal_weights=THINFILM_REF_DATA / 'C24S002' / 'k' / 'V_NVM (V)',
            historical_data=historical,
        )
        figs[0].suptitle(
            'C24N118 calculated using KSTE (w/ Mismatch) +TF $k_{ci} 4-Terms$'
        )
