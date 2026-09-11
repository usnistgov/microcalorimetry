from rmellipse.propagators import RMEProp
from rmellipse.uobjects import RMEMeas

# local packages
from microcalorimetry.math import rfpower, numbers, fitting, rmemeas_extras
from microcalorimetry._helpers._collections import try_sel
from pathlib import Path
import matplotlib.patches as mpatches
import microcalorimetry.configs as configs
import microcalorimetry._gwex as _gwex
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import itertools
import microcalorimetry.arrays as arrays

__all__ = [
    'make_eta',
    'review_eta',
    'make_eta_repeatability_model',
    'make_classical_eta_unc_model',
    'dc_lead_correction',
    'apply_uncertainty_model',
]


def review_eta(
    eta: configs.EtaLike,
    historical_data: configs.EtaHistorical = None,
    repeatability_model: configs.EtaLike = None,
    k: int = 2,
    frequency_precision: int = 4,
    max_hist_legend: int = None,
    hist_avg: bool = True,
) -> tuple[plt.Figure, plt.Figure]:
    """
    Make review plots of effective efficiency data.

    Parameters
    ----------
    eta : configs.EtaLike
        Effective efficiency being reviewed.
    historical_data : configs.EtaHistorical
        Historical data to review against.
    repeatability_model : configs.EtaLike
        Model of calorimeter's repeatability to plot.
    k : int, optional
        Expansion factor, by default 2
    frequency_precision : int, optional
        Precision (decimal places) to round off frequencies to before averaging and comparing
        measurements.
    max_hist_legend : int, optional
        Maxmium number of historical measurmeents to include in the legend.
        If None, will include all. Many historical measurements will result
        in large legends.
    hist_avg : bool, optional
        If True, include the average of historical measurements in the plot.

    Returns
    -------
    fig_hist:
        Figure of effectice efficiency measurement plotted against reference
        data provided.
    fig_budget:
        Figure of uncertainty budget for effective efficiency measurement.

    """
    # get historical data
    if historical_data is None:
        historical_data = {}

    historical_data = configs.EtaHistorical(historical_data)

    nominals = historical_data.load_nominals()

    # average frequency points for each measurement
    # after rounding off frequency values.
    nominals = {
        k: numbers.mean_unique_values(
            v.assign_coords(frequency=np.round(v.frequency, frequency_precision)),
            dim='frequency',
        )
        for k, v in nominals.items()
    }
    # i don't want to deal with this one anymore
    # so throw it awway
    del historical_data

    eta = configs.EtaLike(eta).load()
    nom = eta.nom
    new_fgrid = nom.frequency
    lb = eta.uncbounds(k=-k).cov
    ub = eta.uncbounds(k=k).cov

    cmap = plt.cm.winter

    ydiff_scale = 100

    fig, ax = plt.subplots(2, 1, sharex=True)

    handles = []
    labels = []

    ref = nom.copy()

    # plot the new data uncertainties behind everything
    ax[0].fill_between(new_fgrid, lb[..., 0], ub[..., 0], color='k', alpha=0.2)
    new_err = ax[1].fill_between(
        new_fgrid,
        (lb[..., 0] - nom[:, 0]) * ydiff_scale,
        (ub[..., 0] - nom[:, 0]) * ydiff_scale,
        color='k',
        alpha=0.2,
    )

    (nom_line,) = ax[0].plot(
        new_fgrid, nom, 'k-', marker='o', label='New Nominal', zorder=1000
    )

    handles.append((new_err, nom_line))
    labels.append(r'$\eta^{\mathrm{\left(new\right)}}$' + f' (k={k})')

    ax[1].plot(
        new_fgrid,
        (nom - ref.sel(frequency=nom.frequency)) * ydiff_scale,
        'k',
        lw=3,
        zorder=-900,
    )
    # plot average of reference data with std
    if len(nominals) > 0:
        average_hist = numbers.greedy_average(*list(nominals.values()))
    else:
        average_hist = eta.nom
    use_freqs = np.intersect1d(average_hist.frequency, ref.frequency)

    if hist_avg:
        (hist_avg_line,) = ax[1].plot(
            use_freqs,
            (average_hist.sel(frequency=use_freqs) - ref.sel(frequency=use_freqs))
            * ydiff_scale,
            'r-',
            zorder=1000,
        )
        handles.append(hist_avg_line)
        labels.append(f'Average of History')

    # plot repeatability model centered around
    # the new measurement
    if repeatability_model is not None:
        center = nom - ref.sel(frequency=nom.frequency)
        repmodel = configs.EtaLike(repeatability_model).load()
        ub_rep = center + repmodel.stdunc(k=k).cov[:, 0].interp(frequency=nom.frequency)
        lb_rep = center + repmodel.stdunc(k=-k).cov[:, 0].interp(
            frequency=nom.frequency
        )
        (rep_line,) = ax[1].plot(
            ub_rep.frequency,
            ub_rep * ydiff_scale,
            '-.',
            label='Repeatability Model',
            zorder=1000,
            color='orange',
        )
        ax[1].plot(
            lb_rep.frequency,
            lb_rep * ydiff_scale,
            '-.',
            zorder=900,
            color='orange',
        )

        handles.append(rep_line)
        labels.append(f'Repeatability Model (k=2)')

    for a in ax:
        a.set_prop_cycle(plt.cycler('color', cmap(np.linspace(0, 1, len(nominals)))))
    marker = itertools.cycle(
        ('.', 'o', 'v', '^', 'p', '*', 'h', '<', '>', '1', '2', '3', '4', '8', 's')
    )
    rev_nominals = [(name, datamodel) for name, datamodel in nominals.items()][::-1]
    for i, (name, datamodel) in enumerate(rev_nominals):
        hdat = nominals[name]
        # plot differences

        m = next(marker)
        ax[0].plot(hdat.frequency, hdat, m, label=name)

        use_freqs = np.intersect1d(hdat.frequency, ref.frequency)

        (this_line,) = ax[1].plot(
            use_freqs,
            (hdat.sel(frequency=use_freqs) - ref.sel(frequency=use_freqs))
            * ydiff_scale,
            m,
            label=name,
        )
        if not max_hist_legend:
            handles.append(this_line)
            labels.append(name.replace('_', ' ').title())
        else:
            if i < max_hist_legend - 1:
                handles.append(this_line)
                labels.append(name.replace('_', ' ').title())

    if max_hist_legend:
        # add ...
        empty_handle = mpatches.Patch(color='none', label='asdas')
        handles.append(empty_handle)
        labels.append('...')
        # last item
        handles.append(this_line)
        labels.append(name.replace('_', ' ').title())

    ax[0].set_ylabel(r'$\eta$')
    ax[1].set_ylabel(
        r'$\left[\eta-\eta^{\mathrm{\left(new\right)}}\right]\cdot' + f'{ydiff_scale}$'
    )
    ax[1].set_xlabel('Frequency (GHz)')
    fig.tight_layout()
    fig.subplots_adjust(right=0.7)
    fig.legend(
        handles,
        labels,
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
    model: configs.PythonFunction,
    u_type: str,
    correlate_frequencies: bool = True,
    make_plots: bool = True,
) -> tuple[RMEMeas[arrays.Eta], plt.Figure]:
    """
    Generate a classical uncertainty model for an eta measurement.

    uA_model and uB_model are python functions that take in a frequency
    list and output a standard uncertainty (Type A and B respectively).

    This model can be used to apply uncertainties to an eta calculate
    after it has been calculated. Is is zero nominal, so uncertainties are
    added to an eta measurement by simply adding it to to an effective
    efficiency measurement.

    Uncertainties are assumed to be independent across frequency.

    Parameters
    ----------
    frequency : np.array
        Frequency in GHz.
    model : configs.PythonFunction
        Python function that outputs type B uncertainty.
    u_type : str
        Type of uncertainty (A or B)
    correlate_frequencies : bool, optional
        Treat uncertainties as correlated if True. The default is True.
    make_plots : bool, optional
        If True, make plots.

    Returns
    -------
    eta_unc : RMEMeas[arrays.Eta]
        Eta configuration object with zero nominal and uncertainties
        derived from uA_model and uB_model.
    fig : plt.Figure | None
        Matplotlib figure object generated.

    """
    model = configs.PythonFunction(model)
    u = model(frequency)

    # make an array of zeros to be the nominal
    data = xr.DataArray(
        np.zeros(u.shape), dims=('frequency',), coords={'frequency': frequency}
    ).expand_dims({'eta': [0]}, axis=-1)

    data = _gwex.as_format(data, _gwex.eff)

    # turn into an RMEMeas object with a nominal zero
    data = RMEMeas.from_nom(f'eta_uncertainty', data)

    if correlate_frequencies:
        pert = data.nom.copy()
        pert[:, 0] += u
        data.add_umech(
            model.name,
            pert,
            category={'Type': u_type, 'Origin': model.name},
            add_uid=True,
        )

    else:
        for i, f in enumerate(frequency):
            u_pert = data.nom.copy()
            u_pert.loc[{'frequency': f}] += u[i]
            data.add_umech(
                f'u{u_type}_{f}',
                u_pert,
                category={'Type': u_type, 'Origin': model.name},
                add_uid=True,
            )

    # add uncertainties
    fig = None
    if make_plots:
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
    historical_data: dict[configs.EtaHistorical],
    make_plots: bool = True,
    dist: str = 'gaussian',
    coverage: float = 2,
    ddof: int = 1,
    min_points: int = 3,
    model_type: str = 'expanding_quadratic',
    categories: dict = {'Type': 'A', 'Origin': 'Repeatability'},
    umech_id_basename: str = 'Repeatability',
    frequencies: np.array = None,
) -> tuple[RMEMeas[arrays.Eta], list[plt.Figure], RMEMeas]:
    """
    Generate a historical repeatability model from sensor data.

    Parameters
    ----------
    historical_data : dict[configs.EtaHistorical]
        List of EtaHistorical configuration objects for different sensors. The
        key of each entry is the name of a sensor, and the values are the
        EtaHistorical objects.
    make_plots : bool, optional
        Generate figures if True.
    dist : str, optional
        Assume a type of distribution. Options are 'guassian' or 'uniform'.
        By default 'gaussian'.
    coverage : int, optional
        Attempts to have 1 std uncertainty meet this coverage
        factor. The default is 2.
    ddof : int, optional
        Delta degrees of freedom when calculating standard uncertainty
        frequency by frequency, only relevant for gaussian distributions.
    min_points : int, optional
        Minimum number of required samples per frequency point to be included
        as part of the fit. The default is 3.
    model_type : str, optional, {'expanding_quadratic','expanding_quadratic_linear','quadratic','constant','line'}
        Type of repeatability model to use. Expanding models are a constant value
        up to a cut off frequency, at which point they transition to a increasing
        function. One of 'expanding_quadratic','expanding_quadratic_linear','quadratic','constant','line'.
    categories: dict, optional
        Create category metadata for uncertainy mechnisms. The default is
        {'Type': 'A', 'Origin':'Repeatability'}.
    umech_id_basename: str, optional
        Basename for the umech_id generated.
    frequencies: np.array, optional
        If provided, will apply the repeatability model to the provided
        frequencies.

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
    for i, (sensor_name, sensor_data) in enumerate(historical_data.items()):
        historical = configs.EtaHistorical(sensor_data)
        sensor_noms = {k: configs.EtaLike(v).load().nom for k, v in historical.items()}
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

                # drop points = zero, they would be the only point
                znom = znom.where(znom != 0, drop=True)

                # included points
                line = ax.plot(
                    znom.frequency,
                    znom * 100,
                    marker=marker,
                    color=color,
                    ls='',
                )
            keys = list(this_sensor_zero.keys())
            line[0].set_label(sensor_name)
    valid_dists = ['gaussian', 'uniform']
    if dist not in valid_dists:
        raise ValueError(f'Expect dist to be one of {valid_dists} not {dist}')

    # calculate the uncertainty estimate frequency by frequency

    # assumes a gaussian distribution
    if dist == 'gaussian':
        u = coverage * numbers.greedy_std(
            *list(zero_averaged_sensors.values()), ddof=ddof, min_points=min_points
        )
        u = u.where(u != 0, drop=True).dropna(dim='frequency')

    # assumes a uniform distribution
    elif dist == 'uniform':
        u = xr.concat(list(zero_averaged_sensors.values()), dim='new dim', join='inner')
        u = u.where(u != 0, drop=True).dropna(dim='frequency')
        diff = u.max(dim='new dim') - u.min(dim='new dim')

        u = diff / np.sqrt(12) * coverage
    else:
        raise ValueError(f'Expect dist to be one of {valid_dists} not {dist}')

    def expanding_quadratic(freq, fstop, quadratic, offset):
        out = np.zeros(freq.shape)
        out[freq <= fstop] = offset
        out[freq > fstop] = quadratic * (freq[freq > fstop] - fstop) ** 2 + offset
        # print(out)
        return out

    def expanding_linear(freq, fstop, slope, offset):
        out = np.zeros(freq.shape)
        out[freq <= fstop] = offset
        out[freq > fstop] = slope * (freq[freq > fstop] - fstop) + offset
        # print(out)
        return out

    def quadratic(freq, offset, quadratic):
        out = offset + quadratic * freq**2
        # print(out)
        return out

    def constant(freq, offset):
        return offset

    def line(freq, offset, slope):
        return offset + slope * freq

    fit_funs = [expanding_quadratic, expanding_linear, quadratic, constant, line]
    fit_funs = {f.__name__: f for f in fit_funs}

    spline_funs = ['makima']
    if frequencies is None:
        freqs = u.frequency.data
    else:
        freqs = frequencies
    caught = None
    if model_type in fit_funs:
        try:
            fit = u[:, 0].curvefit('frequency', fit_funs[model_type])

            u_fit_data = fit_funs[model_type](freqs, *fit.curvefit_coefficients.data)
            u_fit = u.copy().interp(
                frequency=freqs, kwargs=dict(fill_value='extrapolate')
            )
            u_fit[:, 0] = u_fit_data

            print(fit.curvefit_coefficients)
            coeffs = fit.curvefit_coefficients
        except Exception as e:
            caught = e
            print('Fit failed for - {e}')
    elif model_type in spline_funs:
        u_fit = u.interp(frequency=freqs, method=model_type)
        coeffs = None

    else:
        raise ValueError(
            f'model_type {model_type} not recognized. Must be one of {[k for k in fit_funs] + spline_funs}'
        )

    # coerce into the expected structure of an eta file
    model = (u_fit[:, 0] * 0).expand_dims('eta', axis=-1)

    model = RMEMeas.from_nom('repeatability_model', model)
    model.add_umech(
        umech_id_basename,
        u_fit,
        dof=len(zero_averaged_sensors),
        add_uid=True,
        category=categories,
    )
    pretty_model_name = model_type.replace('_', ' ').title()
    ub = model.stdunc().cov[..., 0]
    
    
    ax.plot(
        u.frequency,
        u*-100,
        color = 'red',
    )
    ax.plot(
        u.frequency,
        u*100,
        color = 'red',
        label = f'${coverage}\sigma$ of {dist.title()}'
    )
        
    
    ax.fill_between(
        model.nom.frequency,
        ub * 2 * 100,
        ub * -2 * 100,
        color='k',
        alpha=0.2,
        label=f'{pretty_model_name} Model (k=2)',
        zorder=1000,
    )


    h, l = ax.get_legend_handles_labels()

    ax.legend(loc='upper left')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel(r'$\left[\eta - \bar{\eta}_{\mathrm{sensor}}\right]\cdot100$')

    fig.tight_layout()

    if caught is not None:
        raise caught from caught

    return model, fig, coeffs


def dc_lead_correction(
    eta: configs.EtaLike, R_lead: float, R_bolo: float, R_lead_max: float = 0.14
) -> tuple[RMEMeas[arrays.Eta]]:
    """
    Applies a dc lead correction.

    Corrects for heat lost between the dc connector and the
    4-wire connection of a resistor inside the sensor.
    See ERROR IN CALORIMETRIC EFFECTIVE EFFICIENCY MEASUREMENTS DUE TO DC LOSSES, T.P
    Crowley and B.F. Riddle for more details.

    Parameters
    ----------
    eta : configs.EtaLike
        Path to an effective efficiency measurement to apply the correction to.
    R_lead : float
        Sum of resistance for Force and Sense leads of sensor.
    R_bolo : float
        Bolometer resistance, usually 200 ohms for bolometer sensors.
    R_lead_max : float
        This is the maximum allowed resistance for force and sense leads of the sensor.
        Uncertainty is only applied for R_lead > R_lead_max, and only excess uncertainty
        is applied. It is assumed uncertainty
        associated with lead resitances less than or equal to R_lead_max have
        already been accounted for. Set to zero to count all the uncertainty associated
        with your lead resistance. The default value is 0.14 Ohms, which is the typical
        allowable value for CN mounts in NIST's Type N microcalorimeter.

    Returns
    -------
    eta : RMEMeas[arrays.Eta]
        Corrected eta value

    """

    eta = configs.EtaLike(eta).load()

    # set up the propagator
    linear = RMEProp(sensitivity=True)

    bias_correct = linear.propagate(rfpower.dcbias_eta_correction)
    corrected = bias_correct(eta, R_lead, R_bolo)

    if R_lead_max < R_lead:
        print('Excess lead resistance, adding uncertainty')
        assumed_u_already = R_lead_max / 400 / np.sqrt(3)
        total_u = R_lead / 400 / np.sqrt(3)
        extra = np.sqrt(total_u**2 - assumed_u_already**2)
        corrected.add_umech(
            name='eta_dclead_excess',
            value=corrected.nom + extra,
            dof=np.inf,
            category={'Type': 'B', 'Origin': 'Excess DC Lead Resistance'},
            add_uid=True,
        )

    return corrected


def make_eta(
    gc: configs.GCLike,
    s11: configs.S11Like,
    parsed_rfsweep: configs.ParsedRFSweep,
    clrm_sensitivity: configs.KDCLike = None,
    repeatability_model: configs.EtaLike = None,
    extra_eta_uncertainties: list[Path] = None,
    historical_data: configs.EtaHistorical = None,
    legacy_uncertainties: bool = False,
    make_plots: bool = True,
) -> tuple[list[plt.Figure] | None, RMEMeas[arrays.Eta]]:
    """
    Make an effective efficiency measurement.

    Parameters
    ----------
    gc : configs.GCLike
        Calorimetric correction factor terms.
    s11 : configs.S11Like
        Reflection coefficient of the sensor.
    parsed_rfsweep : configs.ParsedRFSweep
        Parsed RF sweep output.
    clrm_sensitivity : configs.KDCLike, optional
        If provided, assumes the correction factor is weighted by the
        calorimeters sensitivity and provide in units of (V/W).
        These coefficents will be used to calculate the dimensionless
        correction factor. The default is None.
    repeatability_model : configs.EtaLike, optional
        Supply a repeatability model of the microcalorimeter to apply as
        uncertainty mechanisms and use in review charts. The default is None.
    extra_eta_uncertainties : list[Path], optional
        Supply additional uncertainty models of eta that should be applied
        after the correction. The default is None.
    historical_data : configs.EtaHistorical, optional
        Supply a historical Eta configuration object to use a reference
        measurements in the review charts. The default is None.
    legacy_uncertainties : bool, optional
        If legacy uncertainties are selected, than uncertainties will not
        be carried forward from the supplied reflection coefficient,
        correction factor, dc measurements, and or sensitivity model.
        All uncertainties are expected to be supplied via the
        extra_eta_uncertainties or repeatability_model options. This
        is to support legacy microcalorimeter systems which do not
        forward propagate uncertainties from their measurement. The
        default is False.
    make_plots : bool, optional
        Make plots during the analsysis,otherwise figs will
        be an empty list. The default is True. Plots are made by passing
        off to the review eta function.

    Returns
    -------

    fig : List[plt.Figure] | None
        Any figures generated.
    eta : RMEMeas[arrays.Eta]
        Effective efficiency.



    """
    # cast as my special dictionary just to be safe
    if historical_data is None:
        historical_data = {}
    historical_data = configs.EtaHistorical(historical_data)

    # set up the propagator
    # if not using legacy uncertainties, then propagate
    # uncertainties forward.
    basic = RMEProp(sensitivity=not legacy_uncertainties)

    effective_efficiency = basic.propagate(rfpower.effective_efficiency)
    mean_unique = basic.propagate(numbers.mean_unique_values)
    calc_te_power = basic.propagate(rfpower.openloop_thermoelectric_power)
    polyval = basic.propagate(fitting.polyval2)

    # load the models into memory from the containers
    parsed_rfsweep = configs.ParsedRFSweep(parsed_rfsweep)
    s11 = configs.S11Like(s11).load()
    gc = configs.GCLike(gc).load()
    zeta = configs.RFSweepLike(parsed_rfsweep['zeta']).load()

    # average repeats for a single connect, zeta repeatability
    # will be taken care of with historical repeatability
    # models, and there is unlikely to be enough frequency points
    # to make a claim about uncertainty at this point.
    e_on = configs.RFSweepLike(parsed_rfsweep['e_on']).load()
    e_off = configs.RFSweepLike(parsed_rfsweep['e_off']).load()
    zeta = mean_unique(zeta, dim='frequency')
    e_on = mean_unique(e_on, dim='frequency')
    e_off = mean_unique(e_off, dim='frequency')

    # select/interp everything down to zeta's grid
    fgrid = zeta.cov.frequency.data
    s11 = try_sel(s11, 'S11', fgrid)
    gc = try_sel(gc, 'gc', fgrid)

    if clrm_sensitivity:
        sense = configs.KDCLike(clrm_sensitivity).load()
        # # Old version from when coefficients were e as function of p estimate coefficients
        #
        # if k.attrs['p_of_e']:
        #     k = 1 / k.sel(deg=1, drop=True)
        # else:
        #     k = k.sel(deg=1, drop=True)
        # # k = np.abs(k)
        #

        p_off_check = calc_te_power(
            sense, e_off[1], sense.attrs['p_of_e']
        ).nom.values.tolist()

        # determine what e value to use for estimating sensitivity
        # if the off power estimate > 10 MW, dc substitution mode and
        # use the on values as the estimate of k
        if p_off_check > 0.01:
            e_k_est = e_on

        # other wise we are in an open loop mode and off values are estimates
        # of e_0 the time varying offset
        else:
            e_k_est = e_on - e_off

        k = polyval(sense, e_k_est)
        # k[sensor_type] = clrm_coeffs[sensor_type].sel(deg = 0, drop = True)
        # if k is units of W/V, then invert it
        if sense.attrs['p_of_e']:
            k = 1 / k

        # print(k)

        gc = gc / k

    eta_new = effective_efficiency(zeta, s11, gc)

    # apply model of statistical repeatability
    if repeatability_model:
        eta_new = apply_uncertainty_model(
            eta_new, repeatability_model, make_unique=True
        )

    if extra_eta_uncertainties:
        for eu in extra_eta_uncertainties:
            eta_new = apply_uncertainty_model(eta_new, eu, make_unique=True)

    # cast as a DataModelContainer and maybe
    # generate review plots
    fig = None
    if make_plots:
        fig = review_eta(
            eta_new, historical_data, repeatability_model=repeatability_model
        )
        fig = list(fig)

    return fig, eta_new


def apply_uncertainty_model(
    eta: configs.EtaLike, model: configs.EtaLike, make_unique: bool = False
) -> RMEMeas[arrays.Eta]:
    """
    Applies uncertainty models of effective efficiency measurements.

    Uncertainties are applied by adding model to eta, assuming model is
    zero nominal.

    Parameters
    ----------
    eta : configs.EtaLike
        Eta that will have unertainties applied.
    model : configs.EtaLike
        Model of uncertainties that will be applied to eta. Should be nominaly
        zero.
    make_unique : bool, optional
        If True, renames all uncertainty mechanisms so they are unique and
        independent from all other uncertainty mechanisms. Should be set
        to True for repeatability models.

    Returns
    -------
    eta : RMEMeas[arrays.Eta]
        Same nominal as the input eta, but with new uncertainties.

    """
    basic = RMEProp(sensitivity=True, set_active=False)

    @basic.propagate
    def add(x, y):
        return x + y

    eta = configs.EtaLike(eta).load()
    model = configs.EtaLike(model).load()
    model = model.interp(
        frequency=eta.nom.frequency,
        method='linear',
        kwargs={'fill_value': 'extrapolate'},
    )
    if make_unique:
        model.make_umechs_unique(same_uid=True)

    # add together
    eta = add(eta, model)
    return eta
