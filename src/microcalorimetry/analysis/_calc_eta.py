from rmellipse.propagators import RMEProp
from rmellipse.uobjects import RMEMeas

# local packages
from microcalorimetry.math import rfpower, vna
from microcalorimetry._helpers._collections import try_sel, mean_unique_values
import microcalorimetry.configs as configs
import microcalorimetry._gwex as _gwex
import warnings
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import itertools
from typing import Callable

__all__ = ['make_eta', 'review_eta', 'make_eta_historical_model', 'make_classical_eta_unc_model']


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


def interp_eta_hist(
    hist: configs.EtaHistorical,
    frequency_bounds_tol: float = 0,
    grad_tol: float = None,
    min_required: int = 3,
    nominal_only: bool = True,
) -> tuple[configs.EtaHistorical, configs.EtaHistorical]:
    """
    Interpolates a set of historical Eta measurements.

    Selects measurements that span the super set of freqeuncy ranges
    within a specified tolerance.

    Parameters
    ----------
    hist : configs.EtaHistorical
        List of historical data to down select.
    frequency_bounds_tol : float, optional
        Bounds by which to check the input data spans the
        range. The default is 0.5.

    Returns
    -------
    no_interp : configs.EtaHistorical
        Input values not interpolated
    interp : configs.EtaHistorical
        Output values interpolated down.
    grad_tol : float, optional
        Requires interpolated values to be within
        the same gradient tolerance, this is to avoid interpolate
        values smoothing over values where a peak is known to exist.
    min_required : int, optional
        Per frequency point, requires at least this any points
        to meet the gradient tolerance to include the point in the model.
    nominal_only : bool, optional
        Removes uncertainty mechanism if asked to.
        Default is True
    """
    sensor_data = configs.EtaHistorical(hist)

    # if nominal only is set,
    # this will reduce it down
    # to only a nominal value
    sel = None
    if nominal_only:
        sel = []
    noms_only = {
        k: configs.Eta(v).load().usel(umech_id=sel) for k, v in sensor_data.items()
    }
    flist = [n.nom.frequency for n in noms_only.values()]
    flist = np.unique(np.concat(flist))
    noms_only_interp = {}
    ub = max(flist) - frequency_bounds_tol
    lb = min(flist) + frequency_bounds_tol
    for i, (k, v) in enumerate(noms_only.items()):
        fmin = np.min(v.nom.frequency)
        fmax = np.max(v.nom.frequency)
        ub_ok = fmax >= ub
        lb_ok = fmin <= lb
        if ub_ok and lb_ok:
            noms_only_interp[k] = v.interp(
                frequency=flist,
                method='quadratic',
                kwargs={'fill_value': 'extrapolate'},
            )
        else:
            msg = f"Skipping {k}, frequencies don't span bounds to tolerance."
            if not ub_ok:
                msg += f'\n   (fmax of {float(fmax)} > {max(flist)} - {frequency_bounds_tol} is False)'
            if not lb_ok:
                msg += f'\n   (fmin of {float(fmin)} < {min(flist)} + {frequency_bounds_tol} is False)'
            print(msg)
    # evaluate concavity
    # to determine if it's acceptable to interpolate
    if grad_tol:
        absgradients = []
        noms = []
        for i, (k, v) in enumerate(noms_only_interp.items()):
            absgradients.append(np.abs(np.gradient(v.nom[:, 0])))
            noms.append(v.nom[:, 0])
        noms = np.array(noms)
        absgradients = np.array(absgradients)
        max_gradients = np.max(absgradients, axis=0)
        drop_f = []
        for i, f in enumerate(flist):
            # bin |gradients| by a tolerance
            bins = np.linspace(
                0, max_gradients[i], int(np.ceil(max_gradients[i] / grad_tol))
            )
            # everypoint fits into the gradient tolerance+-0, continue
            if len(bins) == 1:
                continue
            # the maximum gradient is > tolerance so more then
            # 1 bin, see how the different f points fit into it
            fvals = absgradients[:, i]
            inds = np.digitize(fvals, bins=bins, right=True)
            # if true, every single frequency point fits
            # into the same bin (index) and we can continue
            if len(np.unique(inds)) == 1:
                continue
            # if we made it here some of the points have a smaller gradient
            # and need to be dealt with
            # pick the last bin(largest gradient)
            good = np.where(inds == max(inds))[0]
            bad = np.where(inds != max(inds))[0]

            # if not enough points satisfy the gradient tolerance,
            # add the point to the drop list
            if len(good) < min_required:
                drop_f.append(f)
                # print(f"Dropping frequency {f} (only {len(good)} valid points)")
                continue

            # enough points do statisfy the gradient point, so replace the
            # bad points with values samples on a gaussian distribution
            # that preserves the distribution of the good points
            std_good = np.std(noms[good, i], ddof=1)
            mu_good = np.mean(noms[good, i])
            resamples = np.random.normal(loc=mu_good, scale=std_good, size=len(bad))
            old = noms[bad, i]
            noms[bad, i] = resamples
        # replace ineterpolated output values with resampled
        # values
        for i, k in enumerate(noms_only_interp):
            diff = np.sum(noms_only_interp[k].cov[:, :, 0] - noms[i, :])
            # print(float(diff))
            noms_only_interp[k].cov[:, :, 0] = noms[i, :]
        # remove and frequency points that didn't have enough acceptable data
        # to sample
        if drop_f:
            new_flist_ind = np.logical_not(np.isin(flist, drop_f))
            new_flist = flist[new_flist_ind]
            for i, k in enumerate(noms_only_interp):
                noms_only_interp[k] = noms_only_interp[k].sel(frequency=new_flist)

    return noms_only, noms_only_interp

def make_classical_eta_unc_model(
    frequency: np.array,
    uA_model: configs.PythonFunction,
    uB_model: configs.PythonFunction
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
        np.zeros(uA.shape),
        dims = ('frequency',),
        coords = {'frequency':frequency}).expand_dims({'eta':[0]}, axis = -1)


    data = _gwex.as_format(data, _gwex.eff)

    data = RMEMeas.from_nom(f'eta_uncertainty',data)

    def origin_str(model_fun):
        return f'{model_fun.__name__}'

    for i,f in enumerate(frequency):
        ub_pert = data.nom.copy()
        ua_pert = data.nom.copy()
        ub_pert.loc[{'frequency':f}] += uB[i]
        ua_pert.loc[{'frequency':f}] += uA[i]
        data.add_umech(
            f'uA_{f}',
            ua_pert,
            category = {'Type':'A','Origin':origin_str(uA_model)}
            )
    
        data.add_umech(
            f'uB_{f}',
            ub_pert,
            category = {'Type':'B','Origin':origin_str(uB_model)}
            )

    # add uncertainties
    fig,ax = plt.subplots(1,1)
    grouped = data.categorize_by('Type')
    for u in grouped.umech_id:
        unc = grouped.usel(umech_id = [u]).stdunc().cov
        ax.plot(frequency, unc, label = f'u{u}')
    ax.plot(frequency, data.stdunc().cov, label = 'UTot', color = 'k')
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel('Uncertainty in $\eta$ (k=1)')
    ax.legend(loc = 'best')
    fig.tight_layout()

    return data, fig


def make_eta_historical_model(
    historical_data: list[configs.EtaHistorical],
    make_plots: bool = True,
    frequency_bounds_tol: float = 0.5,
    grad_tol: int = 3e-4,
    min_points: int = 5,
    k: float = 2.0,
) -> tuple[configs.Eta, list[plt.Figure]]:
    """
    Generate a historical repeatability model from sensor data or a function.

    Each configuration object should be for an individual sensor.
    Historical models are generated by comparing the nominal values
    of each connect.

    Missing frequency points are interpolated onto a frequency list that
    represents a "super set" of all unique frequencies on the list.

    Parameters
    ----------
    historical_data : list[configs.EtaHistorical]
        List of EtaHistorical configuration objects for different sensors.
    make_plots : bool, optional
        Generate figures if True.
    frequency_bounds_tol : float, optional
        A historical measurement must have a frequqency range that spanse
        the minimum/maximum frequency range
    grad_tol : float, optional
        An frequency point must have an interpolated value within the maximum
        tolerance of this value to be allowed into the combination.
        Will be resampled it it doesn't.
    min_points : int, optional
        Minimum number of acceptable interpolated measurements per
        frequency point to be accepted into the model.
    k : float, optional
        Expansion factor for plotting the uncertainty bounds

    Returns
    -------
    hist_model : RMEMeas
        RMEMeas object of eff centered at 0 with uncertainties describing
        historical repeatability. Can be added to an effective efficiency
        measurement to add repeatability uncertainties.
    fig : plt.Figure
        Figure reporting the model. None if no figure

    """
    # for each sensor, generate a
    def markers():
        out = itertools.cycle(
            ('.', 'o', 'v', '^', 'p', '*', 'h', '<', '>', '1', '2', '3', '4', '8', 's')
        )
        return out

    expansion_factor = k
    colors = plt.cm.viridis(np.linspace(0, 1, len(historical_data)))
    colors = itertools.cycle(colors)
    prp = RMEProp(sensitivity=True)

    fig = None
    if make_plots:
        fig, ax = plt.subplots(2, 1, sharex=True)
    # values that will be averaged in the end
    zero_averaged_sensors = []
    for i, sensor_data in enumerate(historical_data):
        if len(configs.EtaHistorical(sensor_data)) < min_points:
            raise Exception(
                f'config {i}, not enough sensors to meet mininum {min_points}'
            )
            continue
        noms_only, noms_only_interp = interp_eta_hist(
            sensor_data,
            frequency_bounds_tol=frequency_bounds_tol,
            grad_tol=grad_tol,
            min_required=min_points,
        )
        interp_flist = list(noms_only_interp.values())[0].nom.frequency
        mean = np.mean([v.nom for v in noms_only_interp.values()], axis=0)
        zero_averaged_sensors += [v - mean for v in noms_only_interp.values()]

        if make_plots:
            marker_cycle = markers()
            color = next(colors)
            for k, v in noms_only_interp.items():
                marker = next(marker_cycle)

                line = ax[0].plot(
                    v.nom.frequency,
                    v.nom,
                    marker=marker,
                    color=color,
                    ls='-',
                )[0]
                # included points
                fuse_ind = np.isin(noms_only[k].nom.frequency, interp_flist)
                fuse_ind2 = np.isin(interp_flist, noms_only[k].nom.frequency)
                fuse = noms_only[k].nom.frequency[fuse_ind]
                ax[1].plot(
                    noms_only[k].nom.frequency.sel(frequency=fuse),
                    noms_only[k].nom.sel(frequency=fuse) - mean[fuse_ind2],
                    marker=marker,
                    color=color,
                    ls='',
                )[0]
            keys = list(noms_only_interp.keys())
            line.set_label('{},..., {}'.format(keys[0], keys[-1]))

        del noms_only, noms_only_interp, mean, interp_flist

    # interpolate zero averaged sensors
    flist = [n.nom.frequency for n in zero_averaged_sensors]
    flist = np.unique(np.concat(flist))
    interp_zerod = [
        z.interp(
            frequency=flist, method='linear', kwargs=dict(fill_value='extrapolate')
        )
        for z in zero_averaged_sensors
    ]

    # # combine across individual sensors
    model = prp.combine(
        *interp_zerod,
        combine_categories={'Type': 'A', 'Origin': 'Historical Repeatability'},
    )
    # set nominal to zero jsut to get rid of floating point
    # stuff
    model.cov[0, ...] = 0

    ub = model.uncbounds(k=expansion_factor).cov[:, 0]
    lb = model.uncbounds(k=-expansion_factor).cov[:, 0]
    ax[1].fill_between(
        model.nom.frequency,
        lb,
        ub,
        color='k',
        alpha=0.2,
        label='Historical Repeatability Model (k={:0.2f})'.format(expansion_factor),
        zorder=1000,
    )
    ax[0].legend(loc='best')
    ax[1].legend(loc='best')
    ax[1].set_xlabel('Frequency (GHz)')
    ax[0].set_ylabel('Inerpolated Values of $\eta$')
    ax[1].set_ylabel('Variation in $\eta$ from Sensor Average')

    fig.tight_layout()

    return model, fig


def make_eta(
    gc: configs.GC,
    s11: configs.S11,
    parsed_rfsweep: configs.ParsedRFSweep,
    historical_model: configs.Eta = None,
    historical_data: configs.EtaHistorical = None,
    thermal_weights: configs.ThermoelectricFitCoefficients = None,
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
    historical_model : configs.Eta, optional
        A zero-nominal effective efficiency dataset that is used to apply
        uncertainties related to historical repeatability of the
        calorimeter.
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
        be an empty list. The default is True.

    Returns
    -------

    fig : List[plt.Figure]
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

    # if historical model was provided
    # interpolate it to the right grid and
    # add it to the new value. Historical model should
    # be nominally zero
    if historical_model is not None:
        historical_model = configs.Eta(historical_model).load()
        historical_model = historical_model.interp(
            frequency=eta_new.nom.frequency,
            method='linear',
            kwargs={'fill_value': 'extrapolate'},
        )
        eta_new = eta_new + historical_model

    # cast as a DataModelContainer and maybe
    # generate review plots
    fig = None
    if make_plots:
        fig = review_eta(eta_new, historical_data)

    return list(fig), eta_new
