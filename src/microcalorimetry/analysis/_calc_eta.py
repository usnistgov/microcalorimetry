from rmellipse.propagators import RMEProp
from rmellipse.uobjects import RMEMeas

# local packages
from microcalorimetry.math import rfpower, vna, numbers, fitting, rmemeas_extras
from microcalorimetry._helpers._collections import try_sel
import microcalorimetry.configs as configs
import microcalorimetry._gwex as _gwex
import warnings
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import itertools
from typing import Callable

__all__ = [
    'make_eta',
    'review_eta',
    'make_eta_repeatability_model',
    'make_classical_eta_unc_model',
    'dc_lead_correction',
    'apply_uncertainty_model',
]


def review_eta(
    eta: configs.Eta,
    historical_data: configs.EtaHistorical = None,
    repeatability_model: configs.Eta = None,
    k: int = 2,
):
    """
    Make review plots of effective efficiency data.

    Parameters
    ----------
    eta : configs.Eta
        Effective efficiency being reviewed.
    historical_data : configs.EtaHistorical
        Historical data to review against.
    repeatability_model : configs.Eta
    k : int, optional
        Expansion factor, by default 2

    """
    eta = configs.Eta(eta).load()
    nom = eta.nom
    new_fgrid = nom.frequency
    lb = eta.uncbounds(k=-2).cov
    ub = eta.uncbounds(k=2).cov

    fig, ax = plt.subplots(2, 1)

    cmap = plt.cm.winter

    # get historical data
    if historical_data is None:
        historical_data = {}
    historical_data = configs.EtaHistorical(historical_data)

    nominals = historical_data.load_nominals()
    # i don't want to deal with this one anymore
    # so throw it awway
    del historical_data

    if len(nominals) > 0:
        ref = numbers.greedy_average(*(list(nominals.values()) + [nom]))
    else:
        ref = nom.copy()

    # plot the new data uncertainties behind everything
    ax[0].fill_between(
        new_fgrid, lb[..., 0], ub[..., 0], color='k', alpha=0.2, label=f'Utot k = {k}'
    )
    ax[1].fill_between(
        new_fgrid,
        lb[..., 0] - ref.loc[new_fgrid, 0],
        ub[..., 0] - ref.loc[new_fgrid, 0],
        color='k',
        alpha=0.2,
        label=f'Utot k = {k}',
    )
    nom_line = ax[0].plot(
        new_fgrid, nom, 'k--', marker='o', lw=3, label='New Nominal', zorder=1000
    )
    ax[1].plot(
        new_fgrid,
        nom - ref.sel(frequency=nom.frequency),
        'k',
        marker='o',
        lw=3,
        label='New Nominal',
        zorder=1000,
    )
    # plot repeatability model centered around
    # the new measurement
    if repeatability_model is not None:
        center = nom - ref.sel(frequency=nom.frequency)
        repmodel = configs.Eta(repeatability_model).load()
        ub = center + repmodel.stdunc(k=k).cov[:, 0].interp(frequency=nom.frequency)
        lb = center + repmodel.stdunc(k=-k).cov[:, 0].interp(frequency=nom.frequency)
        ax[1].plot(ub.frequency, ub, 'k-.', label='Repeatability Model', zorder=1000)
        ax[1].plot(lb.frequency, lb, 'k-.', zorder=1000)

    for a in ax:
        a.set_prop_cycle(plt.cycler('color', cmap(np.linspace(0, 1, len(nominals)))))
    marker = itertools.cycle(
        ('.', 'o', 'v', '^', 'p', '*', 'h', '<', '>', '1', '2', '3', '4', '8', 's')
    )
    rev_nominals = [(name, datamodel) for name, datamodel in nominals.items()][::-1]
    for name, datamodel in rev_nominals:
        hdat = nominals[name]
        m = next(marker)
        ax[0].plot(hdat.frequency, hdat, m, label=name)
        ax[1].plot(
            hdat.frequency, hdat - ref.sel(frequency=hdat.frequency), m, label=name
        )

    ax[0].set_ylabel(r'$\eta$')
    ax[1].set_ylabel(r'$\eta$ - New Nominal')
    ax[1].set_xlabel('Frequency (GHz)')
    fig.tight_layout()
    fig.subplots_adjust(right=0.7)
    fig.legend(
        *ax[1].get_legend_handles_labels(),
        loc='upper left',
        ncol=1,
        bbox_to_anchor=(0.7, 1.0),
        bbox_transform=fig.transFigure,
    )

    fig_hist = fig

    # make the uncertainty budget
    fig_budget, ax_budget = plt.subplots(1, 1)
    utot = eta.stdunc(k=1).cov
    ax_budget.plot(eta.nom.frequency, utot * 100, 'k', lw=2, label='Total (k=1)')
    ax_budget.set_xlabel('Frequency (GHz)')
    ax_budget.set_ylabel(r'Uncertainty in $\eta$ (%)')
    try:
        eta2 = rmemeas_extras.categorize_by(eta, 'Origin')
    except KeyError:
        eta2 = eta

    for ploc in eta2.umech_id:
        unc = eta2.usel(umech_id=str(ploc)).stdunc()[0]
        ax_budget.plot(eta.nom.frequency, unc * 100, '--', lw=2, label=ploc)
    box = ax_budget.get_position()
    ax_budget.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax_budget.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    return fig_hist, fig_budget


def make_classical_eta_unc_model(
    frequency: np.array,
    uA_model: configs.PythonFunction,
    uB_model: configs.PythonFunction,
) -> tuple[configs.Eta, plt.Figure]:
    """
    Generate a classical uncertainty model for an eta measurement.

    uA_model and uB_model are python functions that take in a frequency
    list and output a standard uncertainty (Type A and B respectivley).

    This model can be used to apply uncertainties to an eta calculate
    after it has been calculated. Is is zero nominal, so uncertainties are
    added to an eta measurmeent by simply adding it to to an effective
    efficiency measurement.

    Uncertainties are assumed to be independent across frequency.

    Parameters
    ----------
    frequency : np.array
        Frequency in GHz.
    uA_model : configs.PythonFunction
        Python function that outputs type A uncertainty.
    uB_model : configs.PythonFunction
        Python function that outputs type B uncertainty.

    Returns
    -------
    eta_unc : configs.Eta
        Eta configuration object with zero nominal and uncertainties
        derived from uA_model and uB_model.
    fig : plt.Figure
        Matplotlib figure object generated.

    """
    uA = uA_model(frequency)
    uB = uB_model(frequency)

    data = xr.DataArray(
        np.zeros(uA.shape), dims=('frequency',), coords={'frequency': frequency}
    ).expand_dims({'eta': [0]}, axis=-1)

    data = _gwex.as_format(data, _gwex.eff)

    data = RMEMeas.from_nom(f'eta_uncertainty', data)

    def origin_str(model_fun):
        return f'{model_fun.__name__}'

    for i, f in enumerate(frequency):
        ub_pert = data.nom.copy()
        ua_pert = data.nom.copy()
        ub_pert.loc[{'frequency': f}] += uB[i]
        ua_pert.loc[{'frequency': f}] += uA[i]
        data.add_umech(
            f'uA_{f}', ua_pert, category={'Type': 'A', 'Origin': origin_str(uA_model)}
        )

        data.add_umech(
            f'uB_{f}', ub_pert, category={'Type': 'B', 'Origin': origin_str(uB_model)}
        )

    # add uncertainties
    fig, ax = plt.subplots(1, 1)
    grouped = rmemeas_extras.categorize_by(data, 'Type')
    for u in grouped.umech_id:
        unc = grouped.usel(umech_id=[u]).stdunc().cov
        ax.plot(frequency, unc, label=f'u{u}')
    ax.plot(frequency, data.stdunc().cov, label='UTot', color='k')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel(r'Uncertainty in $\eta$ (k=1)')
    ax.legend(loc='best')
    fig.tight_layout()

    return data, fig


def make_eta_repeatability_model(
    historical_data: list[configs.EtaHistorical],
    make_plots: bool = True,
    coverage: float = 2,
    min_points: int = 3,
) -> tuple[configs.Eta, list[plt.Figure], RMEMeas]:
    """
    Generate a historical repeatability model from sensor data.

    Parameters
    ----------
    historical_data : list[configs.EtaHistorical]
        List of EtaHistorical configuration objects for different sensors.
    make_plots : bool, optional
        Generate figures if True.
    coverage : int, optional
        Attempt to have the repeatability uncertainty meet this coverage
        factor. Applied frequency point by frequency point.
    min_points : int, optional
        Minimum number of required samples per frequency points.

    Returns
    -------
    hist_model : RMEMeas
        RMEMeas object of eff centered at 0 with uncertainties describing
        historical repeatability. Can be added to an effective efficiency
        measurement to add repeatability uncertainties.
    fig : plt.Figure
        Figure reporting the model. None if no figure
    coeffs : RMEMeas
        Fit coefficients generated from the repeatability model.

    """

    # for each sensor, generate a
    def markers():
        out = itertools.cycle(
            ('.', 'o', 'v', '^', 'p', '*', 'h', '<', '>', '1', '2', '3', '4', '8', 's')
        )
        return out

    colors = plt.cm.viridis(np.linspace(0, 1, len(historical_data)))
    colors = itertools.cycle(colors)
    prp = RMEProp(sensitivity=True)

    fig = None
    if make_plots:
        fig, ax = plt.subplots(1, 1, sharex=True)
    # values that will be averaged in the end
    zero_averaged_sensors = {}
    for i, sensor_data in enumerate(historical_data):
        historical = configs.EtaHistorical(sensor_data)
        sensor_noms = {k: configs.Eta(v).load().nom for k, v in historical.items()}
        sensor_avg = numbers.greedy_average(*list(sensor_noms.values()))
        this_sensor_zero = {
            k: v - sensor_avg.sel(frequency=v.frequency) for k, v in sensor_noms.items()
        }
        zero_averaged_sensors |= this_sensor_zero
        if make_plots:
            marker_cycle = markers()
            color = next(colors)
            for k in sensor_noms:
                marker = next(marker_cycle)

                nom = sensor_noms[k]
                znom = zero_averaged_sensors[k]

                # included points
                line = ax.plot(
                    znom.frequency,
                    znom,
                    marker=marker,
                    color=color,
                    ls='',
                )
            keys = list(this_sensor_zero.keys())
            line[0].set_label('{},..., {}'.format(keys[0], keys[-1]))

    u = coverage * numbers.greedy_std(
        *list(zero_averaged_sensors.values()), ddof=1, min_points=min_points
    )

    def expanding_uncertainty(freq, fstop, quadratic, offset):
        out = np.zeros(freq.shape)
        out[freq < fstop] = offset
        out[freq > fstop] = quadratic * (freq[freq > fstop] - fstop) ** 2 + offset
        # print(out)
        return out

    fit = u[:, 0].curvefit('frequency', expanding_uncertainty)

    print(fit.curvefit_coefficients)
    coeffs = fit.curvefit_coefficients

    u_fit_data = expanding_uncertainty(
        u.frequency.data, *fit.curvefit_coefficients.data
    )
    u_fit = u.copy()
    u_fit[:, 0] = u_fit_data

    model = u * 0

    model = RMEMeas.from_nom('repeatability_model', model)
    model.add_umech(
        'Repeatabliity',
        u_fit,
        dof=len(zero_averaged_sensors),
        category={'Type': 'A', 'Origin': 'Historical Repeatablity'},
    )

    ub = model.stdunc().cov[..., 0]
    ax.fill_between(
        model.nom.frequency,
        ub * 2,
        ub * -2,
        color='k',
        alpha=0.2,
        label='Historical Repeatability Model (k=2)',
        zorder=1000,
    )
    ax.legend(loc='best')
    ax.legend(loc='best')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel(r'Inerpolated Values of $\eta$')
    ax.set_ylabel(r'Variation in $\eta$ from Sensor Average')

    fig.tight_layout()

    return model, fig, coeffs


def dc_lead_correction(
    eta: configs.Eta,
    R_lead: float,
    R_bolo: float,
) -> tuple[configs.Eta]:
    """
    Corrects an effective efficiency measurmeent for losses in DC leads.

    Parameters
    ----------
    eta : configs.Eta
        _description_
    R_lead : float
        Sum of lead resistance for Force and Sense leads of sensor.
    R_bolo : float
        Bolometer resistance, usually 200 ohms for bolometer sensors.

    Returns
    -------
    eta
        Corrected eta value

    """

    eta = configs.Eta(eta).load()

    # set up the propagator
    linear = RMEProp()

    bias_correct = linear.propagate(rfpower.dcbias_eta_correction)
    corrected = bias_correct(eta, R_lead, R_bolo)

    return corrected


def make_eta(
    gc: configs.GC,
    s11: configs.S11,
    parsed_rfsweep: configs.ParsedRFSweep,
    historical_data: configs.EtaHistorical = None,
    thermal_weights: configs.ThermoelectricFitCoefficients = None,
    uncertainties: bool = True,
    make_plots: bool = True,
) -> tuple[list[plt.Figure] | None, configs.Eta]:
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
    uncertainties : bool, optional
        Propagate uncertainties during calculation, is faster to turn
        off during debugging or exploratory analysis. The default is True.
    make_plots : bool, optional
        Make plots during the analsysis,otherwise figs will
        be an empty list. The default is True. Plots are made by passing
        off to the review eta function.

    Returns
    -------

    fig : List[plt.Figure] | None
        Any figures generated.
    eta : RMEMeas
        Effective efficiency.



    """
    # cast as my special dictionary just to be safe
    if historical_data is None:
        historical_data = {}
    historical_data = configs.EtaHistorical(historical_data)

    # set up the propagator
    basic = RMEProp(sensitivity=uncertainties)

    effective_efficiency = basic.propagate(rfpower.effective_efficiency)
    mean_unique = basic.propagate(numbers.mean_unique_values)

    # load the models into memory from the containers
    parsed_rfsweep = configs.ParsedRFSweep(parsed_rfsweep)
    s11 = configs.S11(s11).load()
    gc = configs.GC(gc).load()
    zeta = configs.RFSweep(parsed_rfsweep['zeta']).load()

    # average repeats for a single connect, zeta repeatability
    # will be taken care of with historical repeatability
    # models, and there is unlikely to be enough frequency points
    # to make a claim about uncertainty at this point.
    e_on = configs.RFSweep(parsed_rfsweep['e_on']).load()
    e_off = configs.RFSweep(parsed_rfsweep['e_off']).load()
    zeta = mean_unique(zeta, dim='frequency')
    e_on = mean_unique(e_on, dim='frequency')
    e_off = mean_unique(e_off, dim='frequency')

    # select/interp everything down to zeta's grid
    fgrid = zeta.cov.frequency.data
    s11 = try_sel(s11, 'S11', fgrid)
    gc = try_sel(gc, 'gc', fgrid)

    if thermal_weights:
        k = configs.ThermoelectricFitCoefficients(thermal_weights).load()
        calc_te_power = basic.propagate(rfpower.openloop_thermoelectric_power)

        polyderive = basic.propagate(fitting.polyderive)
        polyval = basic.propagate(fitting.polyval2)
        derivative = polyderive(k)
        # evaluate the sensitivity at the power
        # levels being measureed
        if k.attrs['p_of_e']:
            # because of inverse function theorem,
            # I can just invert the derivate of P(e)
            kinv = polyval(derivative, e_on - e_off)
            k = 1 / kinv
        else:
            E = calc_te_power(k, e_on - e_off, p_of_e=False)
            k = polyval(derivative, E)
        k = np.abs(k)

        # if k.attrs['p_of_e']:
        #     k = 1 / k.sel(deg=1, drop=True)
        # else:
        #     k = k.sel(deg=1, drop=True)
        k = np.abs(k)
        gc = gc / k

    eta_new = effective_efficiency(zeta, s11, gc)

    # cast as a DataModelContainer and maybe
    # generate review plots
    fig = None
    if make_plots:
        fig = review_eta(eta_new, historical_data)
        fig = list(fig)

    return fig, eta_new


def apply_uncertainty_model(eta: configs.Eta, model: configs.Eta) -> configs.Eta:
    """
    Applies uncertainty models of effective efficiency measurements.

    Uncertainties are applied by adding model to eta, assuming model is
    zero nominal.

    Parameters
    ----------
    eta : configs.Eta
        Eta that will have unertainties applied.
    model : configs.Eta
        Has uncertainties that will be applied on to eta.

    Returns
    -------
    eta : configs.Eta
        Same nominal as the input eta, but with new uncertainties.

    """
    basic = RMEProp(sensitivity=True)
    model = configs.Eta(model).load()
    model = model.interp(
        frequency=eta.nom.frequency,
        method='linear',
        kwargs={'fill_value': 'extrapolate'},
    )
    # add together
    eta = eta + model
    return eta
