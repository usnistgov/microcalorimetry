# -*- coding: utf-8 -*-
"""
This module tests some math functions.

The primary target of these tests should be the
microcalorimetry.math.rfpower submodule.

"""

import microcalorimetry.measurements.dcsweep as dcsweep
import microcalorimetry.math.rfpower as rfpower
import microcalorimetry.configs as configs
from rmellipse.uobjects import RMEMeas
from rmellipse.propagators import RMEProp
from pathlib import Path
from dataclasses import dataclass
import xarray as xr
import numpy as np
from copy import copy

LOCAL = Path(__file__).parents[0]
REF_DATA = LOCAL / 'test_analysis_scripted_refs'


@dataclass
class TECoeffTestCase:
    """
    Coefs = ax^3 + bx^2 + cx + d
    """

    a: float
    b: float
    c: float
    d: float
    volts: float
    p_of_e: bool


def make_coeffs(coeffs: list[float], meas=False) -> xr.DataArray:
    """
    Make a coeffs array [xn, xn-1, ..., x0]

    Parameters
    ----------
    coeffs : float
        list of coeffs.

    Returns
    -------
    coeffs
        Coefficients array

    """
    data = xr.DataArray(
        np.array(coeffs[::-1], float),
        dims=('deg'),
        coords={'deg': np.arange(len(coeffs))},
    )
    data = RMEMeas.from_nom(name='coeffs', nom=data)
    data.add_umech('umech_0', value=data.nom + 0.0001)
    return data


def test_openloop_te_power():
    # make some test cases
    test_cases = [
        TECoeffTestCase(a=1, b=1, c=1, d=1e-3, volts=1e-3, p_of_e=False),
        TECoeffTestCase(a=-1, b=1, c=1, d=1e-3, volts=1e-3, p_of_e=False),
        TECoeffTestCase(a=1, b=-1, c=1, d=-1e-3, volts=1e-3, p_of_e=False),
    ]

    for tc in test_cases:
        test_coeffs = [tc.a, tc.b, tc.c, tc.d]
        test_arr = make_coeffs(test_coeffs)
        test_volts = xr.DataArray([tc.volts])
        test_power = rfpower.openloop_thermoelectric_power(
            test_arr.nom, test_volts, p_of_e=tc.p_of_e
        )

        ref_coeffs = copy(test_coeffs)
        ref_coeffs[-1] -= tc.volts
        ref_roots = np.roots(ref_coeffs)
        # pick the real value
        ref_power_i = np.argmin(np.abs(np.imag(ref_roots)))
        ref_power = ref_roots[ref_power_i]
        print(ref_power, float(test_power))
        assert np.isclose(np.real(ref_power), np.real(test_power))
    pass


if __name__ == '__main__':
    test_openloop_te_power()
