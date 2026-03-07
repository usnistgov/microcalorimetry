# -*- coding: utf-8 -*-
"""
This module contains functions that define uncertainy models for
"""

import numpy as np


def uA_24mm(f):
    """
    Historical repeatability of NIST's 2.4 mm Type A uncertainty.

    Accepts frequency in GHz.

    Taken from the calorimeter matlab code "effcalc.m" on March 2026

    Parameters
    ----------
    f : array
        Frequency in GHz.

    Returns
    -------
    unc : array
        Type A uncertainty same shape as f.

    """
    out = f.copy()
    f_hi = f[f > 29.8]
    out[f <= 29.8] = 0.001
    out[f > 29.8] = 7e-4 + 9e-6 * (f_hi - 24) ** 2
    return out


def _unc_typeN(
    frequencies, RLead: float = 0, RThermistor: float = 200
) -> tuple[np.array]:
    """
    Get the uncertainties associated with the Type N calorimeter.

    This function is envoked by the underlying models.

    This function taken from pat of the effcalc.m matlab function
    on March 2026.

    Parameters
    ----------
    frequencies : array
        Array of frequencies
    RLead : float, optional
        Lead resistance of the sensor. The default is 0.
    RThermistor :float , optional
        Resistance of the thermistor. The default is 200.

    Returns
    -------
    uA : np.array
        Type A uncertainty.
    uB : np.array
        Type B uncertainty.
    Utot : np.array
        k = 2 total uncertainty.

    """
    FreqAll = frequencies

    # URefFlag means the DC resistance was measured
    # calculate DCLoss multiplying factor from RLead
    epsDC = RLead / (2 * RThermistor)  # equation(2) in CPEM 2010 paper
    # this makes efficiency appear too large and therefore to correct for it,
    # need to divide by 1 + epsDC(epsDC is positive)
    # gg = gg / (1 + epsDC)  # include this effect in the correction factor

    # eta = eta_unc * gg

    uDCLoss50 = RLead / 400 / np.sqrt(3)
    UNotDC = 1.94e-3
    U50 = np.sqrt(UNotDC**2 + (2 * uDCLoss50) ** 2)
    RefFreqPointer = FreqAll == 0.05
    OtherFreqPointer = np.logical_not(RefFreqPointer)

    # get uncertainty as combination of the 50 MHz and general rule
    UtotNot50 = 2e-3 + 5.0e-5 * FreqAll + 4.8e-6 * FreqAll**2
    Utot = UtotNot50 * OtherFreqPointer + U50 * RefFreqPointer
    # DC Loss uncertainty from Feb 2012 memo
    uDCLossNot50 = RLead / 400 / np.sqrt(3)
    # uDCLoss = uDCLossNot50 * OtherFreqPointer + uDCLoss50 * RefFreqPointer
    uDCLoss = uDCLossNot50

    # determine uA
    # Type A nanovoltmeter uncertainties from Fred Clague that I am retaining, ue is from g
    # measurements and unVM is from standard runs
    # ue = 0.00016; unVM = 0.00015;
    # random uncertainty from Feb 2012 memo:
    # Random uncertainty at 0 GHz, 18 GHz, and freq where it goes from flat
    # to linear

    uRan0 = 0.00025
    uRan18 = 0.0006
    fBreak = 5
    fNmax = 18
    testN1 = FreqAll <= fBreak
    testN2 = np.logical_not(testN1)
    uRanm = (uRan18 - uRan0) / (fNmax - fBreak)
    uRanb = uRan18 - uRanm * fNmax
    uRanLin = uRanm * FreqAll + uRanb
    uRan = testN1 * uRan0 + testN2 * uRanLin

    # uncertainty in determining g for new calorimeters. It will be used with
    # the old calorimeter as well just to make things simpler.
    # ugsfit = 0.0005 / np.sqrt(3)

    # Next line is really reproducibility, not uA
    # uA = np.sqrt(ue**2 + unVM**2 + ugsfit ^ 2 + uRan**2 + uDCLoss**2)
    uA = np.sqrt(uRan**2 + uDCLoss**2)

    # The code originally calculated the type B uncertainties based on the
    # type A uncertainty and the total uncertainty.
    # So, if you increase uA, uB would decrease to keep Utot the same.
    # I want to assign higher uncertainties bases on less repeateable
    # dc connections.So, to estimate uB, I need to calculate the old
    # value of uA(from the 2012 memo).

    uDCLoss2012 = 0.00035 / np.sqrt(3)
    uA2012 = np.sqrt(uRan**2 + uDCLoss2012**2)

    # Removed special case for calorimeter N0
    # k = 1, not independently specified in memo
    uB = np.sqrt((Utot / 2) ** 2 - uA2012**2)
    Utot = 2 * np.sqrt(uA**2 + uB**2)
    return uA, uB, Utot


def uA_CN(f):
    """
    Type A uncertainty of the Type N Calorimeter in use at NIST.

    Parameters
    ----------
    f : np.ndarray
        Array of frequencies

    Returns
    -------
    unc: array
        Type A uncertainty same shape as f.

    """
    uA, _, _ = _unc_typeN(f)
    return uA


def uB_CN(f):
    """
    Type B uncertainty of the Type N Calorimeters in use at NIST.

    Parameters
    ----------
    f : np.ndarray
        Array of frequencies

    Returns
    -------
    unc: array
        Type B uncertainty same shape as f.

    """
    _, uB, _ = _unc_typeN(f)
    return uB
