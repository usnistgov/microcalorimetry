# -*- coding: utf-8 -*-
"""Analysis functions used in RF power and calorimetery."""

import xarray as xr
import microcalorimetry._gwex as dfm
import numpy as np
from microcalorimetry.math import fitting


def dcbias_eta_correction(
    eta: xr.DataArray,
    R_lead: float,
    R_bolo: float,
):
    """
    Correct an effective efficiency measurment for DC biases.

    From CPEM 2010 paper.

    Parameters
    ----------
    eta : xr.DataArray
        Effective efficiency measurement.
    R_lead : float
        Lead resistance of the Force and Sense leads of the sensor
        added together.
    R_bolo : float
        Resistance of the bolometer, typically 200 Ohms for thermistor
        sensors.
    """
    epsDC = R_lead / (2 * R_bolo)
    return eta / (1 + epsDC)


def openloop_thermoelectric_power(
    coeffs: xr.DataArray,
    e: xr.DataArray,
    p_of_e: bool,
    temperature: xr.DataArray | None = None
) -> xr.DataArray:
    """
    Calculate the openloop thermoelectric power from fit coefficients.

    This function can accept models that provide power as a polynomial
    function of voltage (or vice versa) or can be provided a temperature
    sensitive model of the thermoelectric if temperature readings are 
    also provided.

    Parameters
    ----------
    coeffs : xr.DataArray
        Quadratic fit coefficients. Fit coefficients are along
        last dimension called "deg", coordinates are integer
        values corresponding to the polynomail power (i.e.
        deg = 2 corresponds to c_2 for y = c_2*x**^2 ). Fit
        coefficients can be power in terms of voltage (W/V^n) or
        voltage in terms of power (V/W^n). 
        
        Alternativley, can take in temperature sensitive fit coefficients
        and a temperature reading.
    thermoelectric_voltage : xr.DataArray
        Array of thermoelectric voltages to evaluate power at.
    p_of_e : bool
        If true, assumes fit is power in terms
        of the thermolectric voltage, by default True.
    temperature: xr.DataArray | None, optional
        If provided, assumed to be a temperature dependent thermoelectric model.
        Otherwise assumed to a straight forward polynomial.

    Returns
    -------
    xr.DataArray
        Evaluated thermoelectric power.
    """

    # if temperature readings were given then do a temperature fit and
    # bounce out
    if temperature is not None:
        p = apply_temperature_dependent_fit(
            e, temperature, coeffs)
        return p

    # otherwise a singlae variable polynmial model
    # if power fit in terms of thermoelectric voltage,
    # just evaluate the fit coefficients, this is easiest
    if p_of_e:
        p = fitting.polyval2(coeffs, e)

    else:
        # deg 2, solve root
        # this is faster than polyroot
        # deg 2, solve root
        # this is faster than polyroot
        if max(coeffs.deg) == 2:
            p = e.copy()
            a = coeffs.sel(deg=[2]).data
            b = coeffs.sel(deg=[1]).data
            try:
                c = coeffs.sel(deg=[0]).data - e.data
            except KeyError:
                c = coeffs.sel(deg=[1]).data * 0 - e.data
            root1 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
            root2 = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
            if np.mean(b) > 0:
                p.data[..., :] = root1
            else:
                p.data[..., :] = root2
        # cubic root formula
        elif max(coeffs.deg) == 3:
            p = e.copy()
            a = coeffs.sel(deg=[3]).data
            b = coeffs.sel(deg=[2]).data
            c = coeffs.sel(deg=[1]).data
            # if the 0th term isnt there, assume its zero
            try:
                d = coeffs.sel(deg=[0]).data - e.data
            except KeyError:
                d = coeffs.sel(deg=[1]).data * 0 - e.data
            # intermediate values
            delta_0 = (b**2 - 3 * a * c).astype(complex)
            delta_1 = (2 * b**3 - 9 * a * b * c + 27 * a**2 * d).astype(complex)

            C_root = np.sqrt(delta_1**2 - 4 * delta_0**3)
            C = np.power((delta_1 + C_root) / 2, 1 / 3)
            if np.isclose(np.min(C), 0):
                C = np.power((delta_1 - C_root) / 2, 1 / 3)

            # if zero at this point, then fraction is zero
            x = [np.nan] * 3
            zeta = (-1.0 + (-3.0) ** (1.0 / 2)) / 2.0
            for k in range(3):
                if np.isclose(np.min(C), 0):
                    print(0)
                    xk = -1 / (3 * a) * (b + zeta**k * C + 0)
                else:
                    xk = -1 / (3 * a) * (b + zeta**k * C + delta_0 / (zeta**k * C))
                x[k] = xk
            # check for the real root between 0 and 100
            # that should be the one that indicates the power
            # reading
            # pick the right root:
            # iterate over every single value and do the thing
            # this could be optimizid probably idk
            out_data = np.zeros(p.data.shape)
            for idx, _ in np.ndenumerate(out_data):
                for k in range(3):
                    xi = x[k][idx]
                    # solution shoul be aproimatley real
                    # and provide a value that is close
                    # to what we expect to see
                    if np.isclose(np.imag(xi), 0, atol=1e-10):
                        if np.real(xi) > -1e-3 and np.real(xi) < 50e-3:
                            good_root = k
                            out_data[idx] = np.real(x[k][idx])
                            # print(x[k][idx])

                if good_root is None:
                    raise Exception('Failed to pick a root.')
            p.data[..., :] = out_data

        else:
            p = fitting.polyroot2(coeffs, y=e)

    return p

def _meannorm(x: xr.DataArray, mean: float | xr.DataArray, std: float | xr.DataArray):
    return (x - mean) / std

def temperature_corrected_thermoelectric_fit(
        P: xr.DataArray, 
        T: xr.DataArray, 
        V: xr.DataArray):
    """
    Fits power, voltage, and temperature to a temperature sensitive model.
    
    Power is in Watts and voltage is is Volts. Temperature can be any units
    provided it is proprtional to temperture and the same unit is units
    when using the fit later on. 
    
    It is typically resistance for a resistance thermometer

    Parameters
    ----------
    P : xr.DataArray
        Power (Watts).
    T : xr.DataArray
        Temperatue
    V : xr.DataArray
        Volts (v).

    Returns
    -------
    xr.DataArray
        Array of fit coefficients with names othe coefficients along
        a dimension called 'col'.
    """

    def get_meanstd(Input):    
        mean = np.mean(Input)
        std = np.std(Input)
        return mean, std

    VMEAN, VSTD = get_meanstd(V.values[0])
    VNorm = _meannorm(V.values, VMEAN, VSTD)
    TMEAN, TSTD = 0,1
    if T is not None:
        TMEAN, TSTD = get_meanstd(T.values[0])
        TNorm = _meannorm(T.values, TMEAN, TSTD)

    cs = []
    ats = []
    bts = []
    ds = []
    aps = []
    bps = []
    
    for i in range(0, len(VNorm)):
        if T is not None:
            A = np.column_stack(
                [
                    np.ones(len(VNorm[i].flatten())), 
                    TNorm[i].flatten()**2, 
                    TNorm[i].flatten(), 
                    TNorm[i].flatten()*VNorm[i].flatten(), 
                    VNorm[i].flatten()**2, 
                    VNorm[i].flatten()
                ]
            )
        else:
            A = np.column_stack(
                [
                    np.ones(len(VNorm[i].flatten())), 
                    VNorm[i].flatten()**2, 
                    VNorm[i].flatten()
                ]
            )
        
        
        B = P[i].values.flatten() / V[i].values.flatten()
        result, _, _, _ = np.linalg.lstsq(A, B)
        if T is not None:
            c, at, bt, d, ap, bp = result
        else:
            c, ap, bp = result
            d, at, bt = (0,0,0)
        
        cs.append(c)
        ats.append(at)
        bts.append(bt)
        ds.append(d)
        aps.append(ap)
        bps.append(bp)
    
    cs = np.array(cs)    
    ats = np.array(ats)    
    bts = np.array(bts)    
    ds = np.array(ds)    
    aps = np.array(aps)    
    bps = np.array(bps)    

    VArray = np.zeros(shape=(len(cs), 10))
    VArray[:,0] = aps
    VArray[:,1] = bps
    VArray[:,2] = ats
    VArray[:,3] = bts
    VArray[:,4] = ds
    VArray[:,5] = cs
    VArray[:,6] = TMEAN
    VArray[:,7] = VMEAN
    VArray[:,8] = TSTD
    VArray[:,9] = VSTD


    VXRs = xr.DataArray(
        VArray,
        dims = ['umech_id', 'col'],
        coords = {'umech_id': P.umech_id, 'col': ['ap', 'bp', 'at', 'bt', 'd', 'c', 'MeanT', 'MeanV', 'STDT', 'STDV']}
    )

    return VXRs


def apply_temperature_dependent_fit(
        e: xr.DataArray,
        temperature: xr.DataArray,
        coeffs: xr.DataArray
        ):
    """
    Apply a temperature dependent fit.
    
    Shapes are assumed to be 

    Parameters
    ----------
    e : xr.DataArray
        Thermopile voltage shape (...,N)
    temperature : xr.DataArray
        Temperature readings shape (...,N).
    coeffs : xr.DataArray
        Fit coefficients shape (...,10).
        Last dimension is called 'col' and contains the fit coefficients.

    Returns
    -------
    xr.DataArray
        Array of power readings with the same shape as 
    """
    fit = coeffs
    V = e
    T = temperature

    VsNorm = (V - fit.sel (col = ['MeanV']).data) / fit.sel (col = ['STDV']).data
    TsNorm = (T - fit.sel (col = ['MeanT']).data) / fit.sel (col = ['STDT']).data    
    return V * (fit.sel (col = ['at']).values*TsNorm**2 +
                   fit.sel (col = ['bt']).values*TsNorm +
                   fit.sel (col = ['ap']).values*VsNorm**2 +
                   fit.sel (col = ['bp']).values*VsNorm +
                   fit.sel (col = ['d']).values*VsNorm*TsNorm +
                   fit.sel (col = ['c']).values)


def thermopile_sensitivity(
    p: xr.DataArray,
    e: xr.DataArray,
    constrain_zero: bool = False,
    p_of_e: bool = True,
    deg: int = 2,
    punc: np.array = None,
    eunc: np.array = None,
) -> xr.DataArray:
    """
    Calculate the sensitivity of thermopile element using DC power.

    v, i, and e are typically estimates of the steady state values of a step
    response. Function performs a polynomial of requested degree. v, i, and e
    are assuemd to be of shape (..., n) where all coordinates and dimensions are
    the same. The fitting is done across the n dimensions, and repeated across
    (...).

    Parameters
    ----------
    v : xr.DataArray
        DC voltage applied to a sensor.
    i : xr.DataArray
        DC current applied to a sensor.
    e : xr.DataArray
        DC voltage measured across the thermopile.
    constrain_zero : bool, optional
        Constrain the offset to cross zero.
    p_of_e: bool, optional
        If true, fits power in terms of thermopile voltage, if false fits
        thermopile voltage in terms of power.
    deg: int, optional
        Degree of polynomial fit. The default is 2.
    punc: np.array, optional
        Uncertainty on power values. The default is None. If provided,
        used as weights for the ordinary least squares fitting.
    eunc: np.array, optional
        Uncertainty on power values. The default is None. If provided,
        used as weights for the ordinary least squares fitting.


    Returns
    -------
    sense : xr.DataArray
        Resulting coefficients of the fit.

    """
    if p_of_e:
        y = p
        yunc = punc
        x = e

    else:
        y = e
        yunc = eunc
        x = p

    coeffs = fitting.polyfit2(x, y, deg=deg, constrain_zero=constrain_zero, yunc=yunc)

    return coeffs


def zeta_general(
    e_on: xr.DataArray,
    e_off: xr.DataArray,
    cal_k: xr.DataArray,
    p_of_e: bool,
    P2: xr.DataArray,
    P_dc_on_slow: xr.DataArray = 0,
    P_dc_off_slow: xr.DataArray = 0,
) -> xr.DataArray:
    """
    Calculate uncorrected effective efficiency for any sensor.

    Requires knowledge of calorimeter's sensitivity, but doesn't
    assume linearity as it can be corrected for using
    higher order fit coefficients. This formulation is sensor agnostic.

    Parameters
    ----------
    e_on : xr.DataArray
        Thermopile voltage in the RF on state (of calorimeter).
    e_off : xr.DataArray
        Thermopile voltage in the RF off state (of calorimeter).
    cal_k : xr.DataArray
        Thermopile sensitivity coefficients of the calorimeter.
        Last dimension called 'deg' with integer coordinates
        corresponding to the polynomial order.
    p_of_e : bool
        True means cal_k fits power in terms of thermopile
        voltage. False means cal_k fits thermopile voltage
        in terms of power. V/W or W/V.
    P2 : xr.DataArray
        Metered power of the sensor in the calorimeter
    P_dc_on_slow : xr.DataArray
        DC power in the RF on state evalutated on the
        long term at the point P2 was taken.
    P_dc_off_slow : xr.DataArray
        DC power in the RF off state evaluated on the
        long term at the time P2 was taken (usually via
        interpolation).

    Returns
    -------
    xr.DataArray
        Uncorrected effective efficiency.

    """
    # E_off = openloop_thermoelectric_power(cal_k, e_off, p_of_e)

    E  = openloop_thermoelectric_power(cal_k, e_on-e_off, p_of_e)

    return P2 / (E - P_dc_on_slow + P_dc_off_slow)


def zeta_dcsub(
    e_on: xr.DataArray,
    e_off: xr.DataArray,
    V_on: xr.DataArray,
    V_off: xr.DataArray,
    V_off_slow: xr.DataArray,
    I_on: xr.DataArray = None,
    I_off: xr.DataArray = None,
    I_off_slow: xr.DataArray = None,
) -> xr.DataArray:
    """
    Calculate uncorrected effective efficiency for a DC substitution type sensor.

    All in inputs are assumed to be data arrays of the same shape, and share
    common coordinates.

    Parameters
    ----------
    e_on : xr.DataArray
        Thermopile voltage in the RF on state.
    e_off : xr.DataArray
        Thermopile voltage in the RF off state.
    V_on : xr.DataArray
        DC voltage in the RF on state.
    V_off : xr.DataArray
        DC voltage in the RF off state.
    V_off_slow : xr.DataArray
        DC voltage in the RF off state. (estimated from equilibrium)
    I_on : xr.DataArray, optional
        DC current in the RF on state if power was measured with an SMU. If not
        provided only the voltage measurement will be used.
    I_off : xr.DataArray, optional
        DC current in the RF off state if power was measured with an SMU. If not
        provided only the voltage measurement will be used. The default is None.
    I_off_slow : xr.DataArray, optional
        DC current in the RF off state if power was measured with an SMU
        (in equilibrium). If not provided only the voltage measurement will
        be used. The default is None.
    Returns
    -------
    xr.DataArray
        Uncorrected effective efficiency.

    """
    if I_on is not None:
        P_on = V_on * I_on
    else:
        P_on = V_on**2

    if I_off is not None:
        P_off = V_off * I_off
    else:
        P_off = V_off**2

    if I_off_slow is not None:
        P_off_slow = V_off_slow * I_off
    else:
        P_off_slow = V_off_slow**2

    numerator = (P_off / P_off_slow) - (P_on / P_off_slow)
    denom = e_on / e_off - (P_on / P_off_slow)
    return numerator / denom

def correction_factor(
    gamma_s: xr.DataArray,
    gc: xr.DataArray,
) -> xr.DataArray:
    """
    Get the correction factor.

    Parameters
    ----------
    gamma_s : xr.DataArray
        Reflection coefficient of the standard in s1p_c format.
    gc : xr.DataArray
        Correction factor, either 1 term or 4 term model.

    Raises
    ------
    Exception
        If not a 1 term correction factor, until support added..

    Returns
    -------
    xr.DataArray
        Effective efficiency.

    """
    # 1 term model
    if gc.shape[-1] == 1:
        gam = gamma_s.loc[..., 'S11']
        g = 1 + gc * (1 + np.abs(gam) ** 2) / (1 - np.abs(gam) ** 2)

    elif gc.shape[-1] == 2:
        gam = gamma_s.loc[..., 'S11']
        num = gc[..., 0] * (1 + np.abs(gam) ** 2) - 2 * np.real(gam) * gc[..., 1]
        g = 1 + num / (1 - np.abs(gam) ** 2)

    elif gc.shape[-1] == 4:
        gam = gamma_s.sel(s='S11', drop=True)
        gc1 = gc.sel(gc=0, drop=True)
        gc2 = gc.sel(gc=1, drop=True)
        gc3 = gc.sel(gc=2, drop=True)
        gc4 = gc.sel(gc=3, drop=True)
        num = (
            gc1
            + gc2 * np.abs(gam) ** 2
            - 2 * gc3 * np.real(gam)
            + 2 * gc4 * np.imag(gam)
        )
        g = 1 + num / (1 - np.abs(gam) ** 2)

    elif gc.shape[-1] == 3:
        gam = gamma_s.sel(s='S11', drop=True)
        gc1 = gc.sel(gc=0, drop=True)
        gc3 = gc.sel(gc=1, drop=True)
        gc4 = gc.sel(gc=2, drop=True)
        num = (
            gc1 * (1 + np.abs(gam) ** 2)
            - 2 * gc3 * np.real(gam)
            + 2 * gc4 * np.imag(gam)
        )
        g = 1 + num / (1 - np.abs(gam) ** 2)

    else:
        raise ValueError(f'{gc.shape[-1]} term correction factor not supported')


    return g


def effective_efficiency(
    uncorrected_eta: xr.DataArray,
    gamma_s: xr.DataArray,
    gc: xr.DataArray,
) -> xr.DataArray:
    """
    Calculate effective efficiency.

    Currently only supports a 1 term model.

    Parameters
    ----------
    uncorrected_eta : xr.DataArray
        Uncorrected effective efficiency of a calorimetric measurement.
    gamma_s : xr.DataArray
        Reflection coefficient of the standard in s1p_c format.
    gc : xr.DataArray
        Correction factor, either 1 term or 4 term model.

    Raises
    ------
    Exception
        If not a 1 term correction factor, until support added..

    Returns
    -------
    xr.DataArray
        Effective efficiency.

    """
    # 1 term model
    if gc.shape[-1] == 1:
        gam = gamma_s.loc[..., 'S11']
        g = 1 + gc * (1 + np.abs(gam) ** 2) / (1 - np.abs(gam) ** 2)
        eta = uncorrected_eta * g
        # put into ....,frequency,eta dimensions
        eta = eta.rename({eta.dims[-1]: 'eta'})
        eta = eta.assign_coords({'eta': [0]})
    elif gc.shape[-1] == 2:
        gam = gamma_s.loc[..., 'S11']
        num = gc[..., 0] * (1 + np.abs(gam) ** 2) - 2 * np.real(gam) * gc[..., 1]
        g = 1 + num / (1 - np.abs(gam) ** 2)
        eta = uncorrected_eta * g
        eta = eta.expand_dims({'eta': [0]}, axis=-1)
        eta = eta.drop_vars('s')
    elif gc.shape[-1] == 4:
        gam = gamma_s.sel(s='S11', drop=True)
        gc1 = gc.sel(gc=0, drop=True)
        gc2 = gc.sel(gc=1, drop=True)
        gc3 = gc.sel(gc=2, drop=True)
        gc4 = gc.sel(gc=3, drop=True)
        num = (
            gc1
            + gc2 * np.abs(gam) ** 2
            - 2 * gc3 * np.real(gam)
            + 2 * gc4 * np.imag(gam)
        )
        g = 1 + num / (1 - np.abs(gam) ** 2)
        eta = uncorrected_eta * g
        eta = eta.expand_dims({'eta': [0]}, axis=-1)
        eta = eta.rename({eta.dims[-1]: 'eta'})

    elif gc.shape[-1] == 3:
        gam = gamma_s.sel(s='S11', drop=True)
        gc1 = gc.sel(gc=0, drop=True)
        gc3 = gc.sel(gc=1, drop=True)
        gc4 = gc.sel(gc=2, drop=True)
        num = (
            gc1 * (1 + np.abs(gam) ** 2)
            - 2 * gc3 * np.real(gam)
            + 2 * gc4 * np.imag(gam)
        )
        g = 1 + num / (1 - np.abs(gam) ** 2)
        eta = uncorrected_eta * g
        eta = eta.expand_dims({'eta': [0]}, axis=-1)
        eta = eta.rename({eta.dims[-1]: 'eta'})
    else:
        raise ValueError(f'{gc.shape[-1]} term correction factor not supported')

    eta = dfm.make_into(eta, dfm.eff)
    return eta


def calorimetric_power_delta_general(
    e_on: xr.DataArray,
    e_off: xr.DataArray,
    cal_k: xr.DataArray,
    p_of_e: bool,
    P_dc_on_slow: xr.DataArray = 0,
    P_dc_off_slow: xr.DataArray = 0,
) -> xr.DataArray:
    """
    Calculate power delta.

    Requires knowledge of calorimeter's sensitivity, but doesn't
    assume linearity as it can be corrected for using
    higher order fit coefficients. This formulation is sensor agnostic.

    Parameters
    ----------
    e_on : xr.DataArray
        Thermopile voltage in the RF on state (of calorimeter).
    e_off : xr.DataArray
        Thermopile voltage in the RF off state (of calorimeter).
    cal_k : xr.DataArray
        Thermopile sensitivity coefficients of the calorimeter.
        Last dimension called 'deg' with integer coordinates
        corresponding to the polynomial order.
    p_of_e : bool
        True means cal_k fits power in terms of thermopile
        voltage. False means cal_k fits thermopile voltage
        in terms of power. V/W or W/V.
    P_dc_on_slow : xr.DataArray
        DC power in the RF on state evalutated on the
        long term at the point P2 was taken.
    P_dc_off_slow : xr.DataArray
        DC power in the RF off state evaluated on the
        long term at the time P2 was taken (usually via
        interpolation).

    Returns
    -------
    xr.DataArray
        Uncorrected effective efficiency.

    """
    E  = openloop_thermoelectric_power(cal_k, e_on - e_off, p_of_e)
    # E_on = openloop_thermoelectric_power(cal_k, e_on, p_of_e)
    # E = E_on - E_off

    return E - P_dc_on_slow + P_dc_off_slow


def calorimetric_power_delta_dcsub(
    e_on: xr.DataArray,
    e_off: xr.DataArray,
    k: xr.DataArray,
    p_of_e: bool,
    dc_sub: xr.DataArray = 0,
):
    """
    Calculate  Delta_x for a calorimeter run.

    Delta_x is the difference in power measured by the calorimeters thermopile
    element and the power metered by the sensor. It is a term used in calcualting
    the correction factor of the calorimeter.

    k is assumed to be formatted from the thermopile_sensitivity method.

    If DC current is provided, then the DC power is calculated using V*I, if
    a set point resistance is provided then DC power is calculated using V**2/R.
    One of ether the setpoint OR the DC current must be provided.

    Parameters
    ----------
    e_on : xr.DataArray
        Thermopile voltage in the RF on state.
    e_off : xr.DataArray
        Thermopile voltage in the RF off state.
    k : xr.DataArray
        Sensitivity cofficients that describe the power at the sensitivity
        of the thermopile element. Assumed to be in the mck data format.
    p_of_e : bool
        Whether or not he coefficients (k) are power in terms of e (True)
        or e in terms of power (False).
    dc_sub : xr.DataArray
        DC substituted power.

    Returns
    -------
    delta : xr.DataArray
        Power delta.

    """
    pe_off = openloop_thermoelectric_power(k, e_off, p_of_e)
    pe_on = openloop_thermoelectric_power(k, e_on, p_of_e)
    delta = pe_on - pe_off + dc_sub
    return delta


def calorimetric_alpha_xs(
    dcsub_s: xr.DataArray,
    dcsub_3x: xr.DataArray,
    dcsub_3s: xr.DataArray,
    gamma_s: xr.DataArray,
    gamma_x: xr.DataArray,
    gamma_g: xr.DataArray,
) -> xr.DataArray:
    """
    Calculate the 'alpha' term, proportional to the power incident on sensor x.

    Used in the correction factor algorithm.

    Parameters
    ----------
    dcsub_s : xr.DataArray
        DC substituted power measured by the normal standard in the calorimeter.
    dcsub_3x : xr.DataArray
        DC subtituted power measured by the side arm with sesnsor x in the
        calorimeter.
    dcsub_3s : xr.DataArray
        DC substituted power measured by the sid-arm with the normal standard
        in the calorimeter.
    gamma_s : xr.DataArray
        Reflection coefficient of standard s in s1p_c format.
    gamma_x : xr.DataArray
        Reflection coefficient of sensor x in s1p_c format.
    gamma_g : xr.DataArray
        Port-2-port mismatch of the spliter being used.

    Returns
    -------
    xr.DataArray
        alpha_xs.

    """
    num = dcsub_s * dcsub_3x * np.abs(1 - gamma_g * gamma_s) ** 2
    den = (1 - np.abs(gamma_s) ** 2) * dcsub_3s * np.abs(1 - gamma_g * gamma_x) ** 2
    return num / den


def dc_substituted_power(
    V_on: xr.DataArray,
    V_off: xr.DataArray,
    R: xr.DataArray = None,
    I_on: xr.DataArray = None,
    I_off: xr.DataArray = None,
) -> xr.DataArray:
    """
    Calculate dc substituted power.

    R OR (I_on and I_off) must be provided to calculate the dc substituted
    power.

    Parameters
    ----------
    V_on : xr.DataArray
        DC voltage in the RF on state.
    V_off : xr.DataArray
        DC voltage in the RF off state.
    R : xr.DataArray, optional
        Set-point resistance of the standard. The default is None.
    I_on : xr.DataArray, optional
        DC current in the RF on state. The default is None.
    I_off : xr.DataArray, optional
        DC current in the RF off state. The default is None.

    Raises
    ------
    Exception
        If neither R nor Current measurements are provided, or if both are.

    Returns
    -------
    xr.DataArray
        dc substituted power.

    """
    if R is None and I_on is not None and I_off is not None:
        P_on = V_on * I_on
        P_off = V_off * I_off
    elif R is not None and I_on is None and I_off is None:
        P_on = V_on**2 / R
        P_off = V_off**2 / R
    else:
        raise Exception(
            'Either provide a resistance setting OR on/off current measurements.'
        )

    return abs(P_on - P_off)


def gc_device_row(
    alpha_xs: xr.DataArray,
    delta_x: xr.DataArray,
    zeta_s: xr.DataArray,
    gamma_s: xr.DataArray,
    gamma_x: xr.DataArray,
    n_correction_terms: int = 1,
) -> xr.DataArray:
    """
    Build a row in the correction factor regressor matrix.

    Returns (....,r,c) shaped data

    Parameters
    ----------
    alpha_xs : xr.DataArray
        alpha_xs term in the correction factor model for standard x.
    delta_x : xr.DataArray
        power delta term in the correctionf factor model for sensor x.
    zeta_s : xr.DataArray
        uncorrected effective efficiency of standard s.
    gamma_s : xr.DataArray
        reflection coefficient of standard s in the s1p_c format.
    gamma_x : xr.DataArray
        reflection coefficient of standard x in the s1p_c format.
    n_correction_terms : int, optional
        Number of correction terms being calculated. The default is 1.

    Returns
    -------
    xr.DataArray
        Row in the correction factor regressor matrix
        with shape (...,1, n_correction_terms)
    xr.DataArray
        Solution to the equation for the resulting row
        with shape (..., 1, n_correction_terms)

    """

    gx = 1 - np.abs(gamma_x) ** 2
    gs = 1 - np.abs(gamma_s) ** 2

    solution = gs * (zeta_s * delta_x - gx * alpha_xs)
    old_d = solution.dims[-1]
    solution = solution.rename({old_d: 'row'})
    solution = solution.assign_coords({'row': [0]})
    solution = solution.expand_dims({'col': [0]}, axis=-1)
    # drop unused scalar coords, dont want them.
    for k in solution.coords:
        if k not in solution.dims:
            solution = solution.drop_vars(k)

    # common function for building a single observation in matrix
    def obs(cxi, csi, i):
        num = gs * alpha_xs * cxi - zeta_s * delta_x * csi
        out = num

        # recast into mor useful coordinate space
        old_d = out.dims[-1]
        out = out.rename({old_d: 'row'})
        out = out.assign_coords({'row': [0]})
        out = out.expand_dims({'col': [i]}, axis=-1)

        # drop unused dimensions
        # sometimes they stick around.
        for k in out.coords:
            if k not in out.dims:
                out = out.drop_vars(k)
        return out

    if n_correction_terms == 1:
        cx1 = 1 + np.abs(gamma_x) ** 2
        cs1 = 1 + np.abs(gamma_s) ** 2
        row = obs(cx1, cs1, 1)

    elif n_correction_terms == 2:
        cx1 = 1 + np.abs(gamma_x) ** 2
        cs1 = 1 + np.abs(gamma_s) ** 2
        cx2 = -2 * np.real(gamma_x)
        cs2 = -2 * np.real(gamma_s)
        r11 = obs(cx1, cs1, 1)
        r12 = obs(cx2, cs2, 2)
        row = xr.concat((r11, r12), dim='col')

    elif n_correction_terms == 3:
        cx1 = 1 + np.abs(gamma_x) ** 2
        cs1 = 1 + np.abs(gamma_s) ** 2
        r11 = obs(cx1, cs1, 1)

        cx2 = -2 * np.real(gamma_x)
        cs2 = -2 * np.real(gamma_s)
        r12 = obs(cx2, cs2, 2)

        cx3 = 2 * np.imag(gamma_x)
        cs3 = 2 * np.imag(gamma_s)
        r13 = obs(cx3, cs3, 3)
        row = xr.concat((r11, r12, r13), dim='col')

    elif n_correction_terms == 4:
        cx1 = 1
        cs1 = 1
        r11 = obs(cx1, cs1, 1)

        cx2 = np.abs(gamma_x) ** 2
        cs2 = np.abs(gamma_s) ** 2
        r12 = obs(cx2, cs2, 2)

        cx3 = -2 * np.real(gamma_x)
        cs3 = -2 * np.real(gamma_s)
        r13 = obs(cx3, cs3, 3)

        cx4 = 2 * np.imag(gamma_x)
        cs4 = 2 * np.imag(gamma_s)
        r14 = obs(cx4, cs4, 4)
        row = xr.concat((r11, r12, r13, r14), dim='col')

    else:
        raise Exception('Correction terms ', n_correction_terms, ' not supported')

    return row, solution


def gc_correction_factor(
    regressor: xr.DataArray, solution: xr.DataArray, n_correction_terms: int = 1
) -> xr.DataArray:
    """
    Calculate calorimetric correction factor.

    Parameters
    ----------
    regressor : xr.DataArray
        Rows in the correction factor regressor matrix.
    solution : xr.DataArray
        Solutions to the regressor matrix
    n_correction_term: int,optional
        Number of correction factor terms to calculate.

    Returns
    -------
    xr.DataArray
        Correction factor model.

    """
    out = fitting.ordinary_least_squares(regressor, solution)

    coords = ['gc' + str(i) for i in range(n_correction_terms)]

    out = out.assign_coords({'coeff': coords})
    out = out.rename({'coeff': 'gc'})

    out_formats = {1: dfm.gc1, 2: dfm.gc2, 3: dfm.gc3, 4: dfm.gc4}

    out = dfm.make_into(out, out_formats[n_correction_terms])
    return out


if __name__ == '__main__':
    pass
