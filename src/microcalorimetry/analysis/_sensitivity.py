# third party dependencies
import matplotlib.pyplot as plt
import numpy as np
from rmellipse.propagators import RMEProp
from rmellipse.uobjects import RMEMeas
from microcalorimetry.math import rfpower, fitting, rmemeas_extras
from pathlib import Path
import microcalorimetry._helpers._intf_tools as clitools
import microcalorimetry.configs as configs
import microcalorimetry.arrays as arrays
import click

__all__ = ['fit_thermoelectric']


@click.command(name='make-sensitivity-coeffs')
@click.argument('parsed-dcsweep', type=Path)
@click.option('--constrain-zero', is_flag=True)
@click.option('--p-of-e', is_flag=True)
@click.option('--deg', type=int)
@click.option('--output-file', '-o', type=Path)
@click.option('--show-plots', is_flag=True)
@click.option('--save-plots', type=Path)
@click.option('--plot-ext', type=str, default='.png')
def _cli_make_k_coeffs(
    *args,
    output_file: Path = None,
    show_plots: bool = False,
    save_plots: Path = None,
    plot_ext: str = 'png',
    **kwargs,
):
    """
    Interface for the command line for parsing DC sweeps.

    Parameters
    ----------
    show_plots : bool, optional
        If true, shows the plots in a gui and freezes the terminal, by default False.
        Path to save group under can
        be provided with the group appended to the hdf5 path (i.e. file.h5/group/path)
    output_file : Path, optional
        If provided, outputs any saveable objects to an HDF5 file, by default None.
    save_plots : Path, optional
        If provided, save plots in this directory
    plot_ext : str, optional
        Save extension for the output plots.
    Returns
    -------
    _type_
        _description_
    """
    kwargs['make_plots'] = show_plots or save_plots

    outputs = clitools.run_and_show_plots(
        fit_thermoelectric,
        *args,
        show_plots=show_plots,
        save_plots=save_plots,
        plot_ext=plot_ext,
        **kwargs,
    )
    if output_file:
        clitools.save_saveable_objects(outputs[0], output_file=output_file)
    return outputs


def fit_thermoelectric(
    parsed_dcsweep: configs.ParsedDCSweep,
    thermometer_corrected: bool = False,
    constrain_zero: bool = False,
    p_of_e: bool = False,
    deg: int = 2,
    min_p: float = 0.001,
    make_plots: bool = True,
) -> tuple[RMEMeas[arrays.KDCTemInd | arrays.KDCTempDep], plt.Figure]:
    r"""
    Fit sensitivity coefficients to a dcsweep measurement of a thermoelectric.

    Takes in a parsed dc measurement and generates sensitivity coefficients.

    Parameters
    ----------
    parsed_dcsweep : microcalorimetry.configs.ParsedDCSweep
        Steps parsed from sensitivity measurement. The first field
        is the applied voltage, the second field is the applied current,
        and the last field is the measure thermopile voltage.
    thermometer_corrected : bool, optional
        Use a thermometer corrected model. Requires a thermometer
        voltage and current to be present in the parsed DC sweep. The default
        is False.
    constrain_zero : bool, optional
        Constrains the fit to zero, only applies to non-thermometer
        corrected models. The default is False.
    p_of_e : bool, optional
        Fit power as a function of thermopile voltage if True. If False,
        fits thermopile voltage as a function of power. Only applies to non
        thermometer corrected models. The default is False.
    deg : int, optional
        Fit degrees of polynomial for non thermometer corrected models.
        Only applies to non thermometer corrected models. The default is 2.
    min_p : float, optional
        The minimum power value to include in the fit. Values below this will be included
        as part of the residuals. The default is 0.001 (1 mW).
    make_plots : bool, optional
        Output plots if True.


    Returns
    -------
    sensitivity : RMEMeas[arrays.KDCTemInd | arrays.KDCTempDep]
        Sensitivity coefficients
    figures : list[plt.Figure]
        List of figures generated (empty if not made). Fit and residuals.
    """
    #  pass through to the underlying function
    propagator = RMEProp(sensitivity=True)
    # wrap any uncertainty functions
    calc_thermopile_sensitivity = propagator.propagate(rfpower.thermopile_sensitivity)
    temperature_corrected_thermoelectric_fit = propagator.propagate(
        rfpower.temperature_corrected_thermoelectric_fit
    )
    parsed_dcsweep = configs.ParsedDCSweep(parsed_dcsweep)
    v = configs.DCSweepLike(parsed_dcsweep.pop('heater_v')).load()
    i = configs.DCSweepLike(parsed_dcsweep.pop('heater_i')).load()
    e = configs.DCSweepLike(parsed_dcsweep['e']).load()

    figures = []

    p = v * i

    ind = p.nom >= min_p
    p_use = p[ind.values]
    e_use = e[ind.values]

    # use the thermometer corrected dataset
    if thermometer_corrected:
        therm_v = configs.DCSweepLike(parsed_dcsweep.pop('therm_v')).load()
        therm_i = configs.DCSweepLike(parsed_dcsweep.pop('therm_i')).load()
        therm_r = therm_v / therm_i
        therm_r_use = therm_r[ind.values]
        coeffs = temperature_corrected_thermoelectric_fit(p_use, therm_r_use, e_use)
    # otherwise use a simple polynomial
    else:

        @propagator.propagate
        def divide(a, b):
            out = a.copy()
            out.values = a.values / b.values
            return out

        therm_r = None
        k = divide(e_use, p_use)
        if p_of_e:
            k = 1 / k
        else:
            pass
        coeffs = calc_thermopile_sensitivity(e_use, k, deg=deg, kunc=k.stdunc().cov)

        coeffs.attrs['p_of_e'] = p_of_e

    coeffs.name = 'coeffs'
    v.name = 'voltage_steps'
    i.name = 'current_steps'
    e.name = 'thermopile_steps'

    # make plots if asked to
    if make_plots:
        figs = plot_coeffs(propagator, p, e, coeffs, p_of_e, temperature=therm_r)
        figures += figs

    return coeffs, figures


_cli_make_k_coeffs = clitools.format_from_npdoc(fit_thermoelectric)(_cli_make_k_coeffs)


def plot_coeffs(propagator, p, e, coeffs, p_of_e, temperature: RMEMeas | None):
    """
    Plots coefficients

    Helper function for make_k_coeffs

    Parameters
    ----------
    propagator : RMEProp
        Propagator to use
    p : RMEMeas
        Heater power
    e : RMEMeas
        Voltage measured at each step
    coeffs : _type_
        Fit coefficients
    p_of_e : _type_
        If fit was done in terms of power (True) or voltage (False)
    temperature: RMEMeas | None
        Optional thermometer readings.

    Returns
    -------
    plt.Figure
        matplotlib figure object
    """

    @propagator.propagate
    def minus(ref, vals):
        out = ref.copy()
        out.values = ref.values - vals.values
        return out

    @propagator.propagate
    def div(ref, vals):
        out = ref.copy()
        out.values = ref.values / vals.values
        return out

    @propagator.propagate
    def interp_by_nom(ref, vals):
        "Assume ref is (N,D) array where N is a umech_id dimension."
        d = ref.dims[-1]
        print()
        ref = ref.rename({d: 'nom'})
        ref = ref.assign_coords({'nom': ref[0, :].values.copy()})
        return ref.interp(nom=vals)

    k = 2

    get_openloop = propagator.propagate(rfpower.openloop_thermoelectric_power)
    polyval2 = propagator.propagate(fitting.polyval2)

    sortind = np.argsort(e.nom.values)
    p_fit = get_openloop(coeffs, e, p_of_e, temperature=temperature)

    delta = minus(p, p_fit)

    e = e[sortind]
    p = p[sortind]
    if temperature is not None:
        temperature = temperature[sortind]
    p_fit = p_fit[sortind]
    delta = delta[sortind]
    upper = p_fit.uncbounds(k=k)[0]
    lower = p_fit.uncbounds(k=-k)[0]

    e_upper = e.stdunc(k=k)[0]
    p_upper = p.stdunc(k=k)[0]
    p_fit_upper = p_fit.stdunc(k=k)[0]

    # plot the measured sensitivity

    labels = []
    handles = []
    fig0, ax = plt.subplots(1, 1)
    eOp = div(e, p)
    ebar = ax.errorbar(
        e.nom * 1e3,
        eOp.nom * 1000,
        xerr=e_upper * 1e3,
        yerr=eOp.stdunc(k=k).cov * 1000,
        color='k',
        marker='o',
        markersize=8,
        linestyle='',
    )
    labels.append('Measured')
    handles.append(ebar)

    # plot the fit sensitivity
    if temperature is None:
        e_fit_vals = np.linspace(float(e.nom.min()), float(e.nom.max()))
        e_fit_vals = np.unique(np.append(e.nom.values, e_fit_vals))
        e_fit = interp_by_nom(e, e_fit_vals).usel(umech_id=[])
        sense = polyval2(coeffs, e_fit)

        if coeffs.attrs['p_of_e']:
            sense = 1 / sense
        else:
            sense = sense

        (line,) = ax.plot(
            e_fit.nom * 1e3,
            sense.nom * 1000,
            color='b',
            markersize=8,
            linestyle='--',
            label='Fit',
        )
        sense_unc = sense.stdunc(k=2).cov

        err = ax.fill_between(
            e_fit.nom * 1e3,
            sense.nom * 1000 + sense_unc * 1000,
            sense.nom * 1000 - sense_unc * 1000,
            color='b',
            alpha=0.2,
        )

        labels.append('Fit (k=2)')
        handles.append((err, line))

    ax.set_xlabel(r'$\left(e - e_0\right)\:\mathrm{\left(mV\right)}$')
    ax.set_ylabel(r'$k_{dc}\:\left(\frac{\mathrm{mV}}{\mathrm{W}}\right)$')

    fig0.suptitle('Measured Sensitivity')
    ax.legend(handles, labels, loc='best')

    # plot the Fit Residuals
    fig_pow, ax_pow = plt.subplots(1, 1)
    fig_res, ax_res = plt.subplots(1, 1)
    ax = [ax_pow, ax_res]

    # power vs e-e0
    handles = []
    labels = []

    handles.append
    # plot fit line if not temperature dependent
    if temperature is None:
        (fit_line,) = ax[0].plot(e.nom * 1e3, p_fit.nom * 1e3, 'k--', lw=2)
        fit_err = ax[0].fill_between(
            e.nom * 1e3,
            lower * 1e3,
            upper * 1e3,
            color='k',
            alpha=0.2,
            # label='Fit (k = 2)',
        )
        handles.append((fit_err, fit_line))
        labels.append('Fit (k = 2)')

        pdat = ax[0].errorbar(
            e.nom * 1e3,
            p.nom * 1e3,
            xerr=e_upper * 1e3,
            yerr=p_upper * 1e3,
            marker='o',
            mfc='k',
            markersize=8,
            capsize=5,
            linestyle='',
            # label='Measured (k = 2)',
            elinewidth=1.5,
        )
        handles.append(pdat)
        labels.append('Measured (k = 2)')

    # temperature dependen fit do a scatter
    else:
        scatter = ax[0].scatter(
            e.nom * 1e3,
            p.nom * 1e3,
            c=temperature.nom / 1000,
            cmap=plt.cm.viridis,
            marker='.',
        )
        # errbars = ax[0].errorbar(e.nom*1e3, p.stdunc(k=2).cov*1e3, fmt = 'none', color = 'k', capsize = 3, zorder = -2)
        fig_pow.colorbar(
            scatter, label=r'$R_{\mathrm{T}}\:\left(\mathrm{k\Omega}\right)$'
        )

    if len(handles) > 0:
        ax[0].legend(handles, labels, loc='best')

    ebar_fmt = 'o'
    mfc_color = 'b'
    ebar_mrker = 'o'
    if temperature is not None:
        ebar_fmt = 'k'
        ebar_mrker = ''

    residuals_line = ax[1].errorbar(
        e.nom * 1e3,
        delta.nom * 1e6,
        xerr=e_upper * 1e3,
        yerr=p_upper * 1e6,
        marker=ebar_mrker,
        color='k',
        mfc=mfc_color,
        markersize=8,
        capsize=5,
        linestyle='',
        label='Measured (k = 2)',
        elinewidth=1.5,
    )

    res_handle = residuals_line
    if temperature is not None:
        residuals_scatter = ax[1].scatter(
            e.nom * 1e3,
            delta.nom * 1e6,
            marker='o',
            c=temperature.nom * 1e-3,
            cmap=plt.cm.viridis,
            zorder=100,
        )
        fig_res.colorbar(
            residuals_scatter, label=r'$R_{\mathrm{T}}\:\left(\mathrm{k\Omega}\right)$'
        )
        res_handle = (residuals_line, residuals_scatter)

    fit_err = ax[1].fill_between(
        e.nom * 1e3,
        -1 * p_fit_upper * 1e6,
        p_fit_upper * 1e6,
        color='k',
        alpha=0.2,
        label='Fit (k = 2)',
    )

    handles = [fit_err, res_handle]
    labels = ['Fit (k = 2)', 'Measured (k = 2)']
    ax[1].legend(handles, labels, loc='best')

    ax[0].set_ylabel(r'$P_{dc}\:\left(\mathrm{mW}\right)$')
    ax[1].set_ylabel(r'$P_{dc}-P_{\mathrm{fit}}\:\left(\mathrm{\mu W}\right)$')
    ax[0].set_xlabel(r'$e-e_0\:\left(\mathrm{mV}\right)$')
    ax[1].set_xlabel(r'$e-e_0\:\left(\mathrm{mV}\right)$')

    # power metering uncertainties
    if temperature is not None:
        e_fit_vals = np.linspace(float(e.nom.min()), float(e.nom.max()))
        e_fit = interp_by_nom(e, e_fit_vals)
        e_fit.make_umechs_unique()
        e_fit.assign_categories_to_all(Origin='DC Metering')
        # need figure out the equivalent of 23.5 her
        Tmean = float(temperature.nom.mean())
        temperature_mean_vals = e_fit_vals * 0 + temperature.nom.mean().values
        temperature_mean = interp_by_nom(temperature, temperature_mean_vals)
        temperature_mean.make_umechs_unique()
        temperature_mean.assign_categories_to_all(Origin='DC Metering (Hypothetical)')

        @propagator.propagate
        def reindex(x, dim):
            return x.assign_coords(**{dim: np.arange(len(x.coords[dim]))})

        temperature_mean = reindex(temperature_mean, 'nom')
        e_fit = reindex(e_fit, 'nom')
        p_check = get_openloop(coeffs, e_fit, p_of_e=None, temperature=temperature_mean)
        t_str = r'{:0.3f}'.format(Tmean / 1000) + r'\:\mathrm{k\Omega}'
        title = 'Evaluated at $' + t_str + '$'
        # plot uncertainties on metered power

    # get power metering estaimte for no temperature dependence
    else:
        e_fit_vals = np.linspace(float(e.nom.min()), float(e.nom.max()))
        e_fit = interp_by_nom(e, e_fit_vals)
        e_fit.make_umechs_unique()
        e_fit.assign_categories_to_all(Origin='DC Metering (Hypothetical)')
        p_check = get_openloop(coeffs, e_fit, p_of_e=coeffs.attrs['p_of_e'])
        title = None

    fig, ax = plt.subplots(1, 1)
    p_nom = p_check.nom
    grouped = rmemeas_extras.categorize_by(
        p_check,
        'Origin',
        apply_origin_filter={
            '*Traceability*': 'Sensitivity DC Instruments',
            '*Noise*': 'Sensitivity DC Noise',
        },
    )
    utot = grouped.stdunc(k=1).cov
    ax.plot(p_nom * 1000, utot / p_nom * 100, 'k-', label='Total')
    for u in grouped.umech_id:
        unc = grouped.usel(umech_id=[u]).stdunc(k=1).cov
        ax.plot(p_nom * 1000, (unc / p_nom) * 100, ls='--', label=u)
    ax.legend(loc='best', title='Uncertainty Mechanism')
    ax.set_ylabel(r'k=1 Uncertainty in $P_{\mathrm{fit}}$ (%)')
    ax.set_xlabel(r'$P_{\mathrm{fit}}\:\left(\mathrm{mW}\right)$')
    ax.set_title(title)

    # msg = 'coeffs (units V/W^i or W/V^i) \n'
    # for i in coeffs.nom.deg:
    #     msg += r'c_' + str(int(i)) + ' = ' + str(float(coeffs.nom.sel(deg=i))) + '\n'
    # ax[0].text(
    #     0.9,
    #     0.01,
    #     msg,
    #     ha='center',
    #     fontsize=12,
    #     bbox={'facecolor': 'orange', 'alpha': 0.5, 'pad': 5},
    # )
    return [fig0, fig, fig_pow, fig_res]
