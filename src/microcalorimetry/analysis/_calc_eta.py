from rmellipse.propagators import RMEProp
from rmellipse.uobjects import RMEMeas

# local packages
from microcalorimetry.math import rfpower, vna
from microcalorimetry._helpers._collections import try_sel, mean_unique_values
import microcalorimetry.configs as configs
import warnings
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import itertools

__all__ = ['make_eta', 'review_eta']


def review_eta(eta: configs.Eta, historical_data: configs.EtaHistorical, k: int = 2):
    """
    Make review plots of effective efficiency data.

    Parameters
    ----------
    eta : configs.Eta
        Effective efficiency being reviewed.
    historical_data : configs.EtaHistorical
        Historical data to review against.
    k : int, optional
        Expansion factor, by default 2

    """
    eta = configs.Eta(eta).load()
    nom = eta.nom
    new_fgrid = nom.frequency
    lb = eta.uncbounds(k=-2).cov
    ub = eta.uncbounds(k=2).cov

    fig, ax = plt.subplots(2, 1, figsize=(8, 8))

    cmap = plt.cm.winter

    # plot the new data uncertainties behind everything
    ax[0].fill_between(
        new_fgrid, lb[..., 0], ub[..., 0], color='k', alpha=0.2, label=f'k = {k}'
    )
    ax[1].fill_between(
        new_fgrid,
        lb[..., 0] - nom[..., 0],
        ub[..., 0] - nom[..., 0],
        color='k',
        alpha=0.2,
    )
    nom_line = ax[0].plot(
        new_fgrid, nom, 'k--', marker='o', lw=3, label='New Nominal', zorder=1000
    )

    for a in ax:
        a.set_prop_cycle(
            plt.cycler('color', cmap(np.linspace(0, 1, len(historical_data))))
        )
    marker = itertools.cycle(
        ('.', 'o', 'v', '^', 'p', '*', 'h', '<', '>', '1', '2', '3', '4', '8', 's')
    )

    # plot historical data as dots
    historical_data = configs.EtaHistorical(historical_data)
    for name, datamodel in historical_data.items():
        hdat = configs.Eta(datamodel).load().nom
        m = next(marker)
        ax[0].plot(hdat.frequency, hdat, m, label=name)
        try:
            ax[1].plot(new_fgrid, hdat - nom, m)
        except ValueError:
            print(f'Warning: Interpolating {name} historical data to plot differences.')
            ax[1].plot(new_fgrid, hdat.interp(frequency=new_fgrid) - nom[..., 0], m)

    ax[0].set_ylabel(r'$\eta$')
    ax[1].set_ylabel(r'$\eta$ - New Nominal')
    fig.legend(*ax[0].get_legend_handles_labels(), loc='lower center', ncol=10)
    fig.subplots_adjust(bottom=0.2)

    fig_hist = fig

    # make the uncertainty budget
    fig_budget, ax_budget = plt.subplots(1, 1, figsize=(8.5, 11))
    utot = eta.stdunc(k=1).cov
    ax_budget.plot(eta.nom.frequency, utot / eta.nom * 100, 'k', lw=2, label='Total')
    ax_budget.set_xlabel('Frequency (GHz)')
    ax_budget.set_ylabel(r'Contributions to Uncertainty in $\eta$ (% of Nominal)')
    try:
        eta2 = eta.categorize_by('Origin')
    except KeyError:
        eta2 = eta
    for ploc in eta2.umech_id:
        unc = eta2.usel(umech_id=str(ploc)).stdunc()[0]
        ax_budget.plot(eta.nom.frequency, unc / eta.nom * 100, '--', lw=2, label=ploc)
    box = ax_budget.get_position()
    ax_budget.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax_budget.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    return fig_hist, fig_budget


def make_eta(
    gc: configs.GC,
    s11: configs.S11,
    parsed_rfsweep: configs.ParsedRFSweep,
    historical_data: configs.EtaHistorical = None,
    thermal_weights: configs.ThermoelectricFitCoefficients = None,
    min_historical_repeats: int = 3,
    uncertainties: bool = True,
    make_plots: bool = True,
) -> tuple[configs.Eta, list[plt.Figure]]:
    """
    Make an effective efficiency measurement.

    Parameters
    ----------
    gc : configs.GC
        Calorimetric correction factor terms.
    s11 : configs.S11
        Reflection coefficient of the sensor.
    parsed_rfsweep : configs.ParsedRFSweep
        Parsed RF sweep output.
    historical_data : configs.EtaHistorical, optional
        Dictionary of key value pairs where values are
        configs.Eta. The default is None.
    thermal_weights : configs.ThermoelectricFitCoefficients, optional
        If provied, assumes gc are thermal weight coefficients and are scaled
        by the provided sensitivity coefficient before calculating
        effective efficiency. These should be thermopile sensitivity
        coefficients of the calorimeter calculated with the same model of
        sensor. The default is None.
    min_historical_repeats : int, optional
        Minium number of repeates (per frequency point) required for a data
        driven statistical model to be made for a particular frequency point.
        If less then this number are present, the statistical variation is
        interpolated from nearby points. The default is 3.
    uncertainties : bool, optional
        Propagate uncertainties during calculation, is faster to turn
        off during debugging or exploratory analysis. The default is True.
    make_plots : bool, optional
        Make plots during the analsysis,otherwise figs will
        be an empty list. The default is True.

    Returns
    -------
    eta : RMEMeas
        Effective efficiency.
    fig : List[plt.Figure]
        Any figures generated.


    """
    # cast as my special dictionary just to be safe
    if historical_data is None:
        historical_data = []
    historical_data = configs.EtaHistorical(historical_data)

    # set up the propagator
    basic = RMEProp(sensitivity=uncertainties)

    effective_efficiency = basic.propagate(rfpower.effective_efficiency)
    mean_unique = basic.propagate(mean_unique_values)

    # load the models into memory from the containers
    parsed_rfsweep = configs.ParsedRFSweep(parsed_rfsweep)
    s11 = configs.S11(s11).load()
    gc = configs.GC(gc).load()
    zeta = configs.RFSweep(parsed_rfsweep['zeta']).load()

    # combine repeats for the same disconnect
    warnings.warn('Averaging duplicates on single connect.')
    zeta = mean_unique(zeta, dim='frequency')

    # select/interp everything down to zeta's grid
    fgrid = zeta.cov.frequency.data
    s11 = try_sel(s11, 'S11', fgrid)
    gc = try_sel(gc, 'gc', fgrid)

    if thermal_weights:
        k = configs.ThermoelectricFitCoefficients(thermal_weights).load()
        if k.attrs['p_of_e']:
            k = 1 / k.sel(deg=1, drop=True)
        else:
            k = k.sel(deg=1, drop=True)
        k = np.abs(k)
        gc = gc / k

    eta_new = effective_efficiency(zeta, s11, gc)

    # build a historical model of repeatability
    # by taking standard deviation across historical data
    # this doesn't account for covariance between historical
    # data, which it should, but I'm not sure how to handle that
    # at the moment because the grids are irregular
    if len(historical_data) > min_historical_repeats:
        hist_noms = {}
        flist = zeta.cov.frequency.values
        for name, datamodel in historical_data.items():
            # load the historical data with nominal only into the config
            data = configs.Eta(datamodel).load().usel(umech_id=[])
            # just need the nominal xarray value for now
            hist_noms[name] = data.nom

        # adds each one historical frequency measurmenet
        # as independent, which isn't great
        all_fdat = []
        type_a_fgrid = []
        dofs = []
        for fi in fgrid:
            fdat = []
            for name, hdat in hist_noms.items():
                try:
                    hfdat = hdat.sel(frequency=fi)
                    fdat.append(hfdat)
                except KeyError:
                    pass
            if len(fdat) > min_historical_repeats:
                all_fdat.append(
                    xr.concat(fdat, dim='frequency').std('frequency', ddof=1)
                )
                type_a_fgrid.append(fi)
                dofs.append(len(fdat))
            else:
                dofs.append(min(dofs))

        # interpolate the type analysis to the frequency grid of the zeta measurements
        # is this ok? probaby not but its the best I can think of at the moment
        type_a = (
            xr.concat(all_fdat, dim='frequency')
            .assign_coords(frequency=type_a_fgrid)
            .interp(frequency=fgrid)
        )
        new_nom = eta_new.nom
        # add each frequency as an independent mechanism
        for i, fi in enumerate(fgrid):
            type_a_fi = type_a.copy()
            type_a_fi[fgrid != fi] = 0
            type_a_fi = new_nom + type_a_fi
            eta_new.add_umech(
                'hist_typeA',
                type_a_fi,
                add_uid=True,
                dof=dofs[i],
                category={'Origin': 'Historical Repeatability'},
            )

    # cast as a DataModelContainer and maybe
    # generate review plots
    fig = None
    if make_plots:
        fig = review_eta(eta_new, historical_data)

    return list(fig), eta_new
