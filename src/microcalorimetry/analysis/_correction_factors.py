# third party dependencies
import matplotlib.pyplot as plt
import numpy as np

# uncertainty propagation
from rmellipse.propagators import RMEProp
from rmellipse.uobjects import RMEMeas
from typing import Iterable

# local packages
from microcalorimetry.math import rfpower, fitting
from microcalorimetry._helpers._collections import try_sel, mean_unique_values, concat
import microcalorimetry.configs as configs
import microcalorimetry._gwex as _gwex
import xarray as xr


__all__ = ['make_correction_factor','gc_from_model']

def gc_from_model(
    frequency: np.array,
    model: configs.PythonFunction,
    terms: int = 1,
    )-> tuple[configs.GC, plt.Figure]:
    """
    Supply a parsed RF sweep and a python function to generate a GC model.

    Parameters
    ----------
    frequency : np.array
        Frequencies to evaluate model at.
    model : configs.PythonFunction
        Python function that takes in a frequency (GHz) list and outputs correction
        factor values with the same shape.
    terms : int, optional
        How many terms int he gc model. Default is 1.

    Raises
    ------
    NotImplementedError
        Only 1 term currently supported.

    Returns
    -------
    gc : configs.GC
        Correction factor object.
    fig : plt.Figure
        Plot of the generated GC.

    """
    if terms != 1:
        NotImplementedError('only term=1 is implemented currently.')


    frequency = np.unique(frequency)
    gc = model(frequency)


    # put into an RMEMeas with the expected format

    data = xr.DataArray(
        gc.astype(float),
        dims = ('frequency',),
        coords = {'frequency':frequency}).expand_dims({'gc':[0]}, axis = -1)

    data = _gwex.as_format(data, _gwex.gc1)

    data = RMEMeas.from_nom(f'gc{terms}',data)


    fig,ax = plt.subplots(1,1)
    ax.plot(frequency, data.nom[:,0],'o-')
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel(f'gc model {model.__name__}')

    return data, fig

def make_correction_factor(
    gc_regressor_rows: configs.CorrectionFactorModelInputs,
    correction_terms: int,
    calc_thermal_weights: bool = False,
    make_plots: bool = True,
    nominals: bool = False,
    cache_s11: bool = True,
) -> tuple[configs.GC, list[plt.Figure]]:
    """
    Calculate the correction factor for a given sensor model.

    The normal_standards and special_standards are used to generate
    the regressor matrix. All possible combinations of connect cycles
    that use the same splitter_id are used to build the regressor matrix.

    Parameters
    ----------
    gcinputs: configs.CorrectionFactorDataInputs
        Dictionary of CorrectionFactorDataInputs.
    correction_terms : int
        Number of terms in correction factor. Usually 1.
    calc_thermal_weights : bool, optional
        If true, returns thermal weighting coefficients
        isntead of the correction factors (g_ci = wi/W0) where W0 is
        the sensitivity of the calorimeter.
    make_plots : bool, optional
        Makes plots. The default is True.
    nominals : bool, optional
        If true, propagates uncertainties.
    cache_s11 : bool, optional
        If true, read ins every s-parameter file once to avoid redundant loading of files, memory intensive.

    Returns
    -------
    gc : RMEMeas
        Correction factor in RMEMeas format.
    figs : list[Figure]
        Matplotlib figures generated (if asked to make plots.)

    """
    # save inputs to add to metadata later
    inputs = {**locals()}
    inputs.pop('gc_regressor_rows')

    basic = RMEProp(sensitivity=not nominals)

    calc_delta_power = basic.propagate(rfpower.calorimetric_power_delta_general)
    calc_te_power = basic.propagate(rfpower.openloop_thermoelectric_power)
    calc_alpha = basic.propagate(rfpower.calorimetric_alpha_xs)
    calc_row = basic.propagate(rfpower.gc_device_row)
    calc_gc = basic.propagate(rfpower.gc_correction_factor)
    concat_along = basic.propagate(concat)
    mean_unq = basic.propagate(mean_unique_values)
    polyderive = basic.propagate(fitting.polyderive)
    polyval = basic.propagate(fitting.polyval2)

    # Cache s-parameters into a hash-map using the file+grp string as a key
    # to avoid re-reading s-parameters and coeffs that are saved into the same file
    figures = []
    pairs = []
    rows = []
    solutions = []

    # from notation i math,
    # fs = special sensor (flush short, but could be open)
    # std = normal sensor (standard sensor)
    for row_name, gcr in gc_regressor_rows.items():
        print('\n')
        print(f'Building Regressor Matrix Row : {row_name}')
        print('-------------------------------')
        parsed_standard = configs.ParsedRFSweep(gcr['standard']['parsed_calibration'])
        p3_fast_std = mean_unq(
            configs.RFSweep(parsed_standard['p3_fast']).load(), 'frequency'
        )
        p2_fast_std = mean_unq(
            configs.RFSweep(parsed_standard['p2_fast']).load(), 'frequency'
        )
        zeta_std = mean_unq(
            configs.RFSweep(parsed_standard['zeta']).load(), 'frequency'
        )

        parsed_special = configs.ParsedRFSweep(gcr['special']['parsed_calibration'])
        p3_fast_fs = mean_unq(
            configs.RFSweep(parsed_special['p3_fast']).load(), 'frequency'
        )
        e_off_fs = mean_unq(
            configs.RFSweep(parsed_special['e_off']).load(), 'frequency'
        )
        e_on_fs = mean_unq(configs.RFSweep(parsed_special['e_on']).load(), 'frequency')

        # try to read the slow dc power from special reflect
        # assume zero if it's not present

        if 'p2dc_on' in gcr['special']['parsed_calibration']:
            p2dc_on_fs = mean_unq(
                configs.RFSweep(parsed_special['p2dc_on']).load(), 'frequency'
            )
        else:
            p2dc_on_fs = 0
        if 'p2dc_on' in parsed_special:
            p2dc_off_slow_fs = mean_unq(
                configs.RFSweep(parsed_special['p2dc_off_slow'].load(), 'frequency')
            )
        else:
            p2dc_off_slow_fs = 0

        # check that we have matching frequency grids
        frq_std = np.unique(p3_fast_std.nom.frequency)
        frq_fs = np.unique(p3_fast_fs.nom.frequency)

        # they need to have at least the same frequency grid
        if not np.array_equal(frq_std, frq_fs):
            print(f'Skipping row, mismatched Frequency Grid :  {row_name}')

        # calorimeter coefficients when
        # sensor fs (special) was measured
        fs_clrm_coeffs = configs.ThermoelectricFitCoefficients(
            gcr['special']['clrm_coeffs']
        ).load()

        # rf power absorved by mount assuming gx = 1
        delta_x = calc_delta_power(
            e_on_fs,
            e_off_fs,
            fs_clrm_coeffs,
            fs_clrm_coeffs.attrs['p_of_e'],
            P_dc_on_slow=p2dc_on_fs,
            P_dc_off_slow=p2dc_off_slow_fs,
        )

        # down select frequencies on S1P files
        # interpolate if missing, tell user
        # I think its ok to interpolate on the S1P if
        # it's not provided, but I want to be made aware,
        # since we generally measure S1P files on relatively
        # dense grids

        Gamma_G = configs.S11(gcr['splitter']).load()
        Gamma_std = configs.S11(gcr['standard']['s11']).load()
        Gamma_fs = configs.S11(gcr['special']['s11']).load()

        # use the paths a
        Gamma_G = try_sel(Gamma_G, f'{row_name}-gamma_g', frq_std)
        Gamma_std = try_sel(Gamma_std, f'{row_name}-standard s11', frq_std)
        Gamma_fs = try_sel(Gamma_fs, f'{row_name}-special s11', frq_std)

        alpha_xs = calc_alpha(
            p2_fast_std, p3_fast_fs, p3_fast_std, Gamma_std, Gamma_fs, Gamma_G
        )

        if correction_terms == 2:
            raise Exception('not yet implemented')
            # if basis_id is None:
            #     raise ValueError('Must provide a basis_id for a 2 term model.')
            # print('2 term model, rotating to basis')
            # basis = RMEMeas.from_h5(
            #     f['s1p_basis'][sensor_model][basis_id],
            #     nominal_only= nominals
            # )
            # basis = try_sel(basis, basis_id, frq_std)
            # Gamma_s = rotate_s1p(Gamma_s, basis)
            # Gamma_x = rotate_s1p(Gamma_x, basis)
        row, solution = calc_row(
            alpha_xs, delta_x, zeta_std, Gamma_std, Gamma_fs, correction_terms
        )

        # thermal correction factors are calculated the same weigh
        # but by weighting the correction factor regressor
        # by the sensitivity
        if calc_thermal_weights:
            derivative = polyderive(fs_clrm_coeffs)
            # evaluate the sensitivity at the power
            # levels being measureed
            if fs_clrm_coeffs.attrs['p_of_e']:
                # because of inverse function theorem, 
                # I can just invert the derivate of P(e)
                kinv = polyval(derivative, e_on_fs - e_off_fs)
                k = 1 / kinv
            else:
                E = calc_te_power(fs_clrm_coeffs,e_on_fs - e_off_fs, p_of_e = False)
                k = polyval(derivative, E)
            k = np.abs(k)
            # divide by row
            row = row / k

        rows.append(row)
        solutions.append(solution)

    regressor = concat_along(*rows, dim='row', new_coords=np.arange(0, len(rows)))
    regressor_sol = concat_along(
        *solutions, dim='row', new_coords=np.arange(0, len(rows))
    )
    gc = calc_gc(regressor, regressor_sol, n_correction_terms=correction_terms)
    gc.name = f'gc{correction_terms}'
    gc.attrs.update(inputs)

    if make_plots:
        for i in range(correction_terms):
            fig, ax = plt.subplots(1, 1)
            ax.plot(gc.sel(gc=i).nom.frequency, gc.nom.sel(gc=i), 'ko-')
            lb = gc.sel(gc=i).uncbounds(k=-1)[0]
            ub = gc.sel(gc=i).uncbounds(k=1)[0]
            lcint, ucint = gc.sel(gc=i).confint(0.95)
            ax.plot(lb.frequency, lb, 'b--', label='k = 1 std unc')
            ax.plot(ub.frequency, ub, 'b--')
            ax.plot(lcint.frequency, lcint, 'r--', label='0.95 conf int')
            ax.plot(ucint.frequency, ucint, 'r--')
            ax.set_ylabel(f'gc{i}')
            ax.set_xlabel('Frequency (GHz)')
            ax.legend(loc='best')
            fig.suptitle('')
            # fig.savefig(plot_dir / 'gc1.png')
            figures.append(fig)

            fig, ax = plt.subplots(1, 1)
            ax.plot(
                gc.sel(gc=i).nom.frequency,
                gc.sel(gc=i).stdunc()[0],
                'ko-',
                label='Total',
            )
            if not nominals:
                gred = gc.categorize_by('Origin')
            else:
                gred = gc
            for pl in gred.umech_id:
                pert = gred.cov.loc[pl, ...]
                ax.plot(gc.nom.frequency, pert - gc.sel(gc=i).nom, label=pl)
            ax.set_ylabel(f'gc{i} (k = 1)')
            ax.set_xlabel('Frequency (GHz)')
            ax.legend(loc='best')
            fig.suptitle('')
            # fig.savefig(plot_dir / 'gc1_uncbudget.png')

            figures.append(fig)

    return gc, figures
