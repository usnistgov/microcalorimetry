# third party dependencies
import matplotlib.pyplot as plt
import numpy as np
from rmellipse.propagators import RMEProp
from microcalorimetry.math import rfpower, fitting
from pathlib import Path
import microcalorimetry._helpers._intf_tools as clitools
import microcalorimetry.configs as configs
import click

__all__ = ['make_k_coeffs']


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
        make_k_coeffs,
        *args,
        show_plots=show_plots,
        save_plots=save_plots,
        plot_ext=plot_ext,
        **kwargs,
    )
    if output_file:
        clitools.save_saveable_objects(outputs[0], output_file=output_file)
    return outputs


def make_k_coeffs(
    parsed_dcsweep: configs.ParsedDCSweep,
    constrain_zero: bool = False,
    p_of_e: bool = False,
    deg: int = 2,
    make_plots: bool = False,
) -> tuple[dict[configs.ThermoelectricFitCoefficients], plt.Figure]:
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
    constrain_zero : bool, optional
        Constrains the fit to zero. The default is False.
    p_of_e : bool, optional
        Fit power as a function of thermopile voltage.
        The default is False.
    deg : int, optional
        Fit degrees of polynomial, the default is 2.
    make_plots : bool, optional
        Output plots if True, outputs None in place of figure
        otherwise.

    Returns
    -------
    sensitivity : dict[configs.ThermoelectricFitCoefficients]
        Sensitivity coefficients
    figures : list[plt.Figure]
        List of figures generated (empty if not made). Fit and residuals.
    """
    #  pass through to the underlying function
    propagator = RMEProp(sensitivity=True)
    # wrap any uncertainty functions
    calc_thermopile_sensitivity = propagator.propagate(rfpower.thermopile_sensitivity)
    parsed_dcsweep = configs.ParsedDCSweep(parsed_dcsweep)
    v = configs.DCSweep(parsed_dcsweep.pop('v')).load()
    i = configs.DCSweep(parsed_dcsweep.pop('i')).load()
    coeffs_dict = {}
    figures = []
    for ename in parsed_dcsweep.keys():
        e = configs.DCSweep(parsed_dcsweep[ename]).load()

        p = v * i

        coeffs = calc_thermopile_sensitivity(
            p,
            e,
            constrain_zero=constrain_zero,
            p_of_e=p_of_e,
            deg=deg,
            punc=p.stdunc().cov,
            eunc=e.stdunc().cov,
        )

        coeffs.name = 'coeffs'
        v.name = 'voltage_steps'
        i.name = 'current_steps'
        e.name = 'thermopile_steps'

        coeffs.attrs['constrain_zero'] = constrain_zero
        coeffs.attrs['p_of_e'] = p_of_e
        coeffs_dict[ename] = coeffs

        # remove units from the name, don't want thos its confusing.
        ename.replace('(V)', '')

        # make plots if asked to
        fig = None
        if make_plots:
            fig = plot_coeffs(propagator, v, i, e, coeffs, p_of_e, ename)
            figures.append(fig)

    return coeffs_dict, figures


_cli_make_k_coeffs = clitools.format_from_npdoc(make_k_coeffs)(_cli_make_k_coeffs)


def plot_coeffs(propagator, v, i, e, coeffs, p_of_e, ename):
    """
    Plots coefficients

    Helper function for make_k_coeffs

    Parameters
    ----------
    propagator : RMEProp
        Propagator to use
    v : RMEMeas
        Voltage measured at each step
    i : RMEMeas
        Voltage measured at each step
    e : RMEMeas
        Voltage measured at each step
    coeffs : _type_
        Fit coefficients
    p_of_e : _type_
        IF fit was done in terms of power (True) or voltage (False)
    ename : _type_
        Name of e voltage column

    Returns
    -------
    plt.Figure
        matplotlib figure object
    """
    polyval = propagator.propagate(fitting.polyval2)

    @propagator.propagate
    def mult(v, i):
        return v * i

    @propagator.propagate
    def minus(ref, vals):
        out = ref.copy()
        out.values = ref.values - vals.values
        return out

    k = 2
    p = mult(v, i)
    if p_of_e:
        x = e
        xscale = 1e6
        xname = 'e'
        xunits = r'$\mu$V'

        y = p
        yscale = 1e3
        yname = 'Power'
        yunits = 'mW'

    else:
        x = p
        xscale = 1e3
        xname = 'Power'
        xunits = 'mW'
        y = e
        yscale = 1e6
        yname = 'e'
        yunits = r'$\mu$V'

    x_fit = x
    sortind = np.argsort(x_fit.nom.values)
    y_fit = polyval(coeffs, x_fit)

    delta = minus(y_fit, y)

    x_fit = x_fit.isel(steps=sortind)
    x = x.isel(steps=sortind)
    y = y.isel(steps=sortind)
    y_fit = y_fit.isel(steps=sortind)
    delta = delta.isel(steps=sortind)

    fig, ax = plt.subplots(2, 1, sharex = True)
    upper = y_fit.uncbounds(k=k)[0]
    lower = y_fit.uncbounds(k=-k)[0]

    x_upper = x.stdunc(k=k)[0]
    y_upper = y.stdunc(k=k)[0]
    y_fit_upper = y_fit.stdunc(k=k)[0]

    ax[0].fill_between(
        x_fit.nom * xscale,
        lower * yscale,
        upper * yscale,
        color='k',
        alpha=0.2,
        label='k = 2 uncertainty',
    )
    ax[0].plot(x_fit.nom * xscale, y_fit.nom * yscale, 'r-', lw=2, label='Fit')

    ax[0].errorbar(
        x.nom * xscale,
        y.nom * yscale,
        xerr=x_upper * xscale,
        yerr=y_upper * yscale,
        marker='o',
        markersize=8,
        linestyle='',
        label='Measured (k = 2 Uncertainty)',
    )

    ax[1].errorbar(
        x.nom * xscale,
        delta.nom * yscale,
        xerr=x_upper * xscale,
        yerr=y_upper * yscale,
        marker='.',
        markersize=8,
        capsize=5,
        linestyle='',
        label='Measured (k = 2 Uncertainty)',
    )

    ax[1].fill_between(
        x_fit.nom * xscale,
        -1 * y_fit_upper * yscale,
        y_fit_upper * yscale,
        color='k',
        alpha=0.2,
        label='k = 2 Uncertainty',
    )
    ax[0].set_ylabel(f'{yname} ({yunits})')
    ax[1].set_ylabel(f'Fit - Measured ({yunits})')
    ax[1].set_xlabel(f'{xname} ({xunits})')

    for a in ax:
        a.legend(loc='best')
    fig.suptitle(f'Sensitivity Fit {ename}')
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
    return fig
