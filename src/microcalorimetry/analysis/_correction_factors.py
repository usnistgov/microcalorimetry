# third party dependencies
import matplotlib.pyplot as plt
import numpy as np

# uncertainty propagation
from rmellipse.propagators import RMEProp
from rmellipse.uobjects import RMEMeas
from typing import Iterable

# local packages
from microcalorimetry.math import rfpower, fitting, rmemeas_extras, numbers
from microcalorimetry._helpers._collections import try_sel, concat
import microcalorimetry.configs as configs
import microcalorimetry._gwex as _gwex
import xarray as xr


__all__ = ['make_correction_factor','gc_from_model','review_correction_factor']

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

def review_correction_factor(
    gc: configs.GC,
    units: str = '',
    budget: bool = True
    ) -> list[plt.Figure,...]:
    """
    Review a correction factor model.

    Parameters
    ----------
    gc : configs.GC
        Correction factor model.
    units : str, optional
        Units string for y-axis. The default is ''.
    budget : bool, optional
        Plot the uncertainty budget. The default is True.

    Returns
    -------
    list[plt.Figure,...]
        List of figures generated.
    """

    w = configs.GC(gc).load()
    gred = rmemeas_extras.categorize_by(w,'Origin')

    nterms = int(max(w.nom.gc))
    # print(nterms)
    out_figs = []

    if units:
        units = '( '+units + ' )'
    else:
        units = ''


    for i in range(nterms+1):
        # plot resultt
        fig, ax = plt.subplots(1,1)

        ax.plot(
            w.sel(gc=i).nom.frequency,
            w.nom.sel(gc=i),
            'o-',
        )
        lb = w.sel(gc=i, drop=True).uncbounds(k=-2).cov
        ub = w.sel(gc=i, drop=True).uncbounds(k=2).cov
        ax.fill_between(lb.frequency, lb, ub, alpha=0.2, label = 'k = 2')
        ax.set_title(f'Correction Term ${{{i}}}$')
        ax.set_ylabel(f'Correction Term ${{{i}}}$ '+ units)
        ax.legend(loc = 'best')
        ax.set_xlabel('Frequency (GHz)')
        fig.tight_layout()

        out_figs.append(fig)


        # plot uncertainties for term
        if budget:
            fig, ax = plt.subplots(1,1)
            wi = gred.sel(gc = i)
            for uid in wi.umech_id:
                ax.plot(
                    wi.nom.frequency,
                    wi.usel(umech_id = uid).stdunc().cov,
                    label = uid,
                    )
            ax.plot(wi.nom.frequency, wi.stdunc(k=1).cov, color = 'k', label = 'Total')
            ax.set_xlabel('Frequency (GHz)')
            ax.set_ylabel('Contribution to Uncertainty (k=1)' + units)
            ax.set_title(f'Correction Term ${{{i}}}$')
            ax.legend(loc='best')
            out_figs.append(fig)

    return out_figs

def make_correction_factor(
    gc_regressor_rows: configs.CorrectionFactorModelInputs,
    correction_terms: int,
    calc_thermal_weights: bool = False,
    make_plots: bool = True,
    nominals: bool = False,
    cache_s11: bool = True,
    freq_resolution: int = 4,
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
    
    calc_te_power = basic.propagate(rfpower.openloop_thermoelectric_power)
    calc_p_comp = basic.propagate(rfpower.p_comp)
    calc_row = basic.propagate(rfpower.gc_device_row)
    calc_gc = basic.propagate(rfpower.gc_correction_factor)
    concat_along = basic.propagate(concat)
    mean_unq = basic.propagate(numbers.mean_unique_values)
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
    all_union_freqs = []
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
        
        e_off_std = mean_unq(
            configs.RFSweep(parsed_standard['e_off']).load(), 'frequency'
        )
        
        e_on_std = mean_unq(
            configs.RFSweep(parsed_standard['e_on']).load(), 'frequency'
        )
        # try to read the slow dc power from special reflect
        # assume zero if it's not present
        
        # get union of frequency grids
        f1 = p3_fast_std.nom.frequency
        f2 = p3_fast_fs.cov.frequency
        union_f = np.intersect1d(f1,f2)

        if 'p2dc_on' in parsed_special:
            p2dc_on_fs = mean_unq(
                configs.RFSweep(parsed_special['p2dc_on']).load(), 'frequency'
            ).sel(frequency = union_f)
        else:
            p2dc_on_fs = 0
        if 'p2dc_on' in parsed_special:
            p2dc_off_slow_fs = mean_unq(
                configs.RFSweep(parsed_special['p2dc_off_slow'].load(), 'frequency')
            ).sel(frequency = union_f)
        else:
            p2dc_off_slow_fs = 0


        clrm_coeffs = {}
        clrm_coeffs['fs'] = configs.ThermoelectricFitCoefficients(
            gcr['special']['clrm_coeffs']
        ).load()

        clrm_coeffs['std'] = configs.ThermoelectricFitCoefficients(
            gcr['standard']['clrm_coeffs']
        ).load()

        
        # down select frequencies on S1P files
        # interpolate if missing, tell user
        # I think its ok to interpolate on the S1P if
        # it's not provided, but I want to be made aware,
        # since we generally measure S1P files on relatively
        # dense grids

        Gamma_G = configs.S11(gcr['splitter']).load()
        Gamma_std = configs.S11(gcr['standard']['s11']).load()
        Gamma_fs = configs.S11(gcr['special']['s11']).load()

        # reduce down to the union of frequencies
        Gamma_G = try_sel(Gamma_G, f'{row_name}-gamma_g', union_f)
        Gamma_std = try_sel(Gamma_std, f'{row_name}-standard s11', union_f)
        Gamma_fs = try_sel(Gamma_fs, f'{row_name}-special s11', union_f)

        p_comp = calc_p_comp(
            p2_fast_std.sel(frequency = union_f),
            p3_fast_fs.sel(frequency = union_f),
            p3_fast_std.sel(frequency = union_f),
            Gamma_std,
            Gamma_fs,
            Gamma_G
        )
        

        
        # i left this in because I wanted to check some numbers,
        # should be turned off normally.
        debug = False
        if debug:
            p_comp_mismatch_term = (p_comp*p3_fast_std.sel(frequency = union_f))/(p2_fast_std.sel(frequency = union_f)*p3_fast_fs.sel(frequency = union_f))
            
            fig,ax = plt.subplots(1,1)
            ax.plot(union_f, p_comp_mismatch_term.nom)
            fig.suptitle("Mismatch Term on $P^{comp}$\n" + f"{row_name}")
            fig.tight_layout()
            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel("Mismatch Term")
            ...

        if correction_terms == 2:
            raise NotImplementedError('Not implemented.')
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

        # calorimeter coefficients when
        # sensor fs (special) was measured


        # Force calorimeter coefficients to be linearized
        # (if they aren't already) and in units of V/W
        k = {}
        p_off_check = {}
        
        # etimate what the power would be in the off state. If its
        # > 10 mW then the calorimeter is operating in a dc substitution mode
        # and the off measurement is a good estimate of sensitivity.
        # Otherwise, the off measurement is the time varying e0 estimate
        # and will need to be suvtracted from e_on to estimate k
        off_check = {'fs':e_off_fs[0], 'std': e_off_std[0]}
        e_ons = {'fs':e_on_fs, 'std':e_on_std}
        e_offs = {'fs':e_off_fs, 'std':e_off_std}
        p_cal = {}
        for sensor_type in ['fs','std']:
            # get the power estimated from the thermopile voltage in the off
            # state (as a float). This will tell us what operating mode the
            # microcalorimeter is in.
            p_off_check = calc_te_power(
                clrm_coeffs[sensor_type],
                off_check[sensor_type],
                clrm_coeffs[sensor_type].attrs['p_of_e']
                ).nom.values.tolist()
        
            # determine what e value to use for estimating sensitivity
            # if the off power estimate > 10 MW, dc substitution mode and
            # use the off values as the estimate of k
            if p_off_check > 0.01:
                e_k_est = e_ons[sensor_type]

            # other wise we are in an open loop mode and need to pick the 
            # right sensitivity on the curve
            else:
                e_k_est = e_ons[sensor_type] - e_offs[sensor_type]
                
            k[sensor_type] = polyval(clrm_coeffs[sensor_type], e_k_est)
            # k[sensor_type] = clrm_coeffs[sensor_type].sel(deg = 0, drop = True)
            # if k is units of W/V, then invert it
            if clrm_coeffs[sensor_type].attrs['p_of_e']:
                k[sensor_type] = 1/k[sensor_type]
                
            # calculate the p_cal for the flush short (special reflect)
            if sensor_type == 'fs':
                p_cal['fs'] = calc_te_power(
                    clrm_coeffs['fs'],
                    e_ons['fs'] - e_offs['fs'],
                    p_of_e = clrm_coeffs['fs'].attrs['p_of_e']
                    )
                # if in dc substitution mode, subtract off dc power
                if p_off_check > 0.01:
                    p_cal['fs'] = p_cal['fs'] - (p2dc_on_fs - p2dc_off_slow_fs)
            # nonlinear approximation by evaluatiing derivative
            # at the measured power level
            # in the linear case this just resolves to the slope of the linear fit.
            # in the nonlinear case this (might) correct for nonlinearity.
            # derivative = polyderive(clrm_coeffs[sensor_type])
            # if clrm_coeffs[sensor_type].attrs['p_of_e']:
            #     # because of inverse function theorem,
            #     # I can just invert the derivate of P(e)
            #     kinv = polyval(derivative, e_on_fs - e_off_fs)
            #     k[sensor_type] = 1 / kinv
            # else:
            #     E = calc_te_power(clrm_coeffs[sensor_type],e_on_fs - e_off_fs, p_of_e = False)
            #     k[sensor_type] = polyval(derivative, E)
        
        # if not asked to, calculate a traditional
        # correction factor, not krf
        if not calc_thermal_weights:
            k['std'] = 1
            k['fs'] = 1

        row, solution = calc_row(
            p_comp,
            p_cal['fs'],
            zeta_std.sel(frequency = union_f),
            Gamma_std,
            Gamma_fs,
            k_s = k['std'],
            k_x = k['fs'],
            n_correction_terms = correction_terms
        )
        rows.append(row)
        solutions.append(solution)
        all_union_freqs.append(union_f)

    regressor = concat_along(*rows, dim='row', new_coords=np.arange(0, len(rows)))
    regressor_sol = concat_along(
        *solutions, dim='row', new_coords=np.arange(0, len(rows))
    )
    gc = calc_gc(regressor, regressor_sol, n_correction_terms=correction_terms)
    gc.name = f'gc{correction_terms}'
    gc.attrs.update(inputs)
    # frequencies that don't line up will show up as NA, so only keep
    # union of frequency lists
    super_union = all_union_freqs[0]
    if len(all_union_freqs[0]) > 1:
        for fl in all_union_freqs[1:]:
            super_union = np.intersect1d(super_union, fl)
    
    gc = gc.sel(frequency = super_union)
    print(gc.sel(gc = 0)[-1])

    if make_plots:
        figures = review_correction_factor(gc, budget = True)
    # gc.assign_categories_to_all(Origin = 'Correction Factor')
    return gc, figures
