# third party dependencies
import matplotlib.pyplot as plt
import numpy as np
from rmellipse.propagators import RMEProp
from rmellipse.uobjects import RMEMeas
from microcalorimetry.math import rfpower, fitting
from pathlib import Path
import microcalorimetry._helpers._intf_tools as clitools
import microcalorimetry.configs as configs
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
    constrain_zero: bool = False,
    thermometer_corrected: bool = False,
    p_of_e: bool = False,
    deg: int = 2,
    make_plots: bool = False,
) -> tuple[ RMEMeas, plt.Figure]:
    r"""
    Load and analyze a sensitivity run.

    Takes in a sensitivity experiment and generate sensitivity coefficients
    and other relevant data.

    Parameters
    ----------
    parsed_dcsweep : microcalorimetry.configs.ParsedDCSweep
        Steps parsed from sensitivity measurement. The first field
        is the applied voltage, the second field is the applied current,
        and the last field is the measure thermopile voltage.
    thermometer_corrected : bool, optional
        Use a thermometer corrected model. Requires a thermometer
        voltage and current to be present in the parsed DC sweep.
    constrain_zero : bool, optional
        Constrains the fit to zero, only applies to non thermometer
        corrected models. The default is False.
    p_of_e : bool, optional
        Fit power as a function of thermopile voltage, only applies to non
        thermometer corrected models. The default is False.
    deg : int, optional
        Fit degrees of polynomial, only applies to non
        thermometer corrected models. The default is 2.
    make_plots : bool, optional
        Output plots if True, outputs None in place of figure
        otherwise.

    Returns
    -------
    sensitivity : RMEMeas
        Sensitivity coefficients
    figures : list[plt.Figure]
        List of figures generated (empty if not made). Fit and residuals.
    """
    #  pass through to the underlying function
    propagator = RMEProp(sensitivity=True)
    # wrap any uncertainty functions
    calc_thermopile_sensitivity = propagator.propagate(rfpower.thermopile_sensitivity)
    temperature_corrected_thermoelectric_fit = propagator.propagate(
        rfpower.temperature_corrected_thermoelectric_fit)
    parsed_dcsweep = configs.ParsedDCSweep(parsed_dcsweep)
    v = configs.DCSweep(parsed_dcsweep.pop('heater_v')).load()
    i = configs.DCSweep(parsed_dcsweep.pop('heater_i')).load()
    e = configs.DCSweep(parsed_dcsweep['e']).load()
    
    figures = []

    p = v * i
    
    # use the thermometer corrected dataset
    if thermometer_corrected:
        therm_v = configs.DCSweep(parsed_dcsweep.pop('therm_v')).load()
        therm_i = configs.DCSweep(parsed_dcsweep.pop('therm_i')).load()
        therm_r = therm_v/therm_i
        coeffs = temperature_corrected_thermoelectric_fit(
            p,
            therm_r,
            e
            )
    # otherwise use a simple polynomial
    else:
        therm_r = None
        coeffs = calc_thermopile_sensitivity(
            p,
            e,
            constrain_zero=constrain_zero,
            p_of_e=p_of_e,
            deg=deg,
            punc=p.stdunc().cov,
            eunc=e.stdunc().cov,
        )
        
        coeffs.attrs['constrain_zero'] = constrain_zero
        coeffs.attrs['p_of_e'] = p_of_e


    coeffs.name = 'coeffs'
    v.name = 'voltage_steps'
    i.name = 'current_steps'
    e.name = 'thermopile_steps'


    # make plots if asked to
    if make_plots:
        figs = plot_coeffs(propagator, p, e, coeffs, p_of_e, temperature = therm_r)
        figures+=figs

    return coeffs, figures


_cli_make_k_coeffs = clitools.format_from_npdoc(fit_thermoelectric)(_cli_make_k_coeffs)


def plot_coeffs(
    propagator,
    p,
    e,
    coeffs,
    p_of_e,
    temperature: RMEMeas | None
    ):
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


    k = 2

    
    get_openloop = propagator.propagate(
        rfpower.openloop_thermoelectric_power
        )


    sortind = np.argsort(e.nom.values)
    p_fit = get_openloop(coeffs, e, p_of_e, temperature = temperature)

    delta = minus(p_fit, p)

    e = e[sortind]
    p = p[sortind]
    p_fit = p_fit[sortind]
    delta = delta[sortind]
    upper = p_fit.uncbounds(k=k)[0]
    lower = p_fit.uncbounds(k=-k)[0]

    e_upper = e.stdunc(k=k)[0]
    p_upper = p.stdunc(k=k)[0]
    p_fit_upper = p_fit.stdunc(k=k)[0]
    
    # plot the measured sensitivity
    fig0, ax = plt.subplots(1,1)
    eOp = div(e,p)
    ax.errorbar(
        e.nom * 1e3,
        eOp.nom,
        xerr=e_upper * 1e3,
        yerr= eOp.stdunc(k=k).cov,
        marker='o',
        markersize=8,
        linestyle='',
        label='Measured (k = 2 Uncertainty)',
    )
    ax.set_xlabel('e (mV)')
    ax.set_ylabel(r'$\frac{e}{P_{heater}}\:\left(\frac{\mathrm{V}}{\mathrm{W}}\right)$')
    fig0.suptitle('Measured Sensitivity')

    # plot the Fit Residuals
    fig, ax = plt.subplots(2, 1, sharex = True)

    

    ax[0].fill_between(
        e.nom * 1e3,
        lower * 1e3,
        upper * 1e3,
        color='k',
        alpha=0.2,
        label='k = 2 uncertainty',
    )
    ax[0].plot(e.nom * 1e3, p_fit.nom * 1e3, 'r-', lw=2, label='Fit')

    ax[0].errorbar(
        e.nom * 1e3,
        p.nom * 1e3,
        xerr=e_upper * 1e3,
        yerr=p_upper * 1e3,
        marker='o',
        markersize=8,
        linestyle='',
        label='Measured (k = 2 Uncertainty)',
    )

    ax[1].errorbar(
        e.nom * 1e3,
        delta.nom * 1e6,
        xerr= e_upper * 1e3,
        yerr= p_upper * 1e6,
        marker='.',
        markersize=8,
        capsize=5,
        linestyle='',
        label='Measured (k = 2 Uncertainty)',
    )

    ax[1].fill_between(
        e.nom * 1e3,
        -1 * p_fit_upper * 1e6,
        p_fit_upper * 1e6,
        color='k',
        alpha=0.2,
        label='k = 2 Uncertainty',
    )
    

    ax[0].set_ylabel(r'$P_{heater}\:\left(\mathrm{mW}\right)$')
    ax[1].set_ylabel(r'$P_{fit}-P_{heater}\:\left(\mathrm{\mu W}\right)$')
    ax[1].set_xlabel(r'$e\:\left(\mathrm{mV}\right)$')

    for a in ax:
        a.legend(loc='best')
    fig.suptitle(f'Sensitivity Fit')
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
    return [fig0, fig]
