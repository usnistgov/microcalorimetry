"""
This module contains functions for exporting effective efficiency datasets.
"""

from __future__ import annotations
import microcalorimetry.configs as configs
from microcalorimetry.math import rmemeas_extras
from microcalorimetry._tkquick.gui_dtypes import SaveAsPath
from rmellipse.uobjects import RMEMeas
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime

__all__ = ['as_doteff']


def group_typed_uncertainties(params: RMEMeas) -> RMEMeas:
    """
    Group uncertainty mechanisms in params by Type.

    Any mechanism with a combine_id is assumed to be Type A.

    Any mechanisms that doesn't have a type assigned is assumed to be

    Parameters
    ----------
    params : RMEMeas
        Meas object to group uncertainty mechanisms.

    Returns
    -------
    grouped : RMEMeas
        Meas object with grouped uncertainties.

    """
    # treat anything with a combine ID as type A, derived from statistical means
    use = params.copy()
    cc = use.covcats
    if 'combine_id' in params.covcats.categories:
        cc.loc[{'categories': 'Type'}][cc.loc[{'categories': 'combine_id'}] != ''] = 'B'
    # anything that hasn't been assigned a Type yet is Type B

    not_a_or_b = np.logical_and(
        cc.loc[{'categories': 'Type'}] != 'A', cc.loc[{'categories': 'Type'}] != 'B'
    )
    if not_a_or_b.any():
        print(
            'Some uncertainty mechanisms not identified as type A or B, assigning as B.'
        )
        cc.loc[{'categories': 'Type'}][not_a_or_b] = 'B'

    # assume anything with a combine id is Type A
    # things that didn't have one would be grouped under uncategorized
    grouped = rmemeas_extras.categorize_by(use, 'Type')
    return grouped


def as_doteff(
    output_path: SaveAsPath,
    eta: configs.Eta,
    s11: configs.S11,
    sensor_name: str = None,
    connect_number: int = None,
    expansion_factor: int | float = 2,
    frequency_decimals: int = 2,
    eta_decimals: int = 4,
    s11_abs_decimals: int = 4,
    s11_angle_decimals: int = 2,
    columns: int = 7,
):
    """
    Generate a .eff file from an effective efficiency measurement.

    These files contain an effective efficiency measurement, as well
    as a reflection coefficient and an expression of uncertainty. Type A and
    Type B uncertainties will be inferred from the objects covariance
    metadata.

    Exporting as a .eff file will lose
    covariance information about the measurement.


    Parameters
    ----------
    output_path : SaveAsPath
        File path to save to.
    eta : configs.Eta
        Effective efficiency of the sensor.
    s11 : configs.S11
        S11 data of the sensor.
    sensor_name : str
        Name of the sensor. If not provided, won't be included
        in the output file.
    connect_number : int
        Connect number of the sensor. If not provided, won't be included
        in the comment line.
    expansion_factor : int, optional
        Expansion factor of the total uncertainty. If >= 1, will use that
        as the expansion factor. If < 1, then will calculate the confidence
        interval of the provided fraction (e.g 0.95 will calculate the
        95% confidence interval) using metadata of the uncertainty mechanisms.
    frequency_decimals : int, optional
        Decimals for reporting frequency, by default 2
    eta_decimals : int, optional
        Decimals for reporting the eta parameters, by default 4.
    s11_abs_decimals : int, optional
        Decimals for reporting the S11 absolute value, by default 4.
    s11_angle_decimals : int, optional
        Decimals for reporting the S11 angle vakule, by default 2.
    columns : int, optional
        Number of columns in the eff file. By default 7.
    """

    eta_dat = configs.Eta(eta).load()
    s11_dat = configs.S11(s11).load()

    flist = eta_dat.cov.frequency
    try:
        s11_dat = s11_dat.sel(frequency=flist)
    except KeyError:
        print('Warning: missing frequencies in S11 data. Interpolating.')
        s11_dat = s11_dat.interp(frequency=flist)

    now = datetime.now()
    now = now.strftime('%d-%b-%Y')

    eta_dat = eta_dat[..., 0]
    eta_nom = eta_dat.nom.copy()
    eta_grouped = group_typed_uncertainties(eta_dat)
    eta_uA = eta_grouped.usel(umech_id=['A']).stdunc(k=1).cov
    eta_uB = eta_grouped.usel(umech_id=['B']).stdunc(k=1).cov
    # if using the DOFs, the expansion factor might be slightly different
    # for each frequency point. So, I'm picking the worst case scenario
    # to report here.
    if expansion_factor < 1:
        eta_utot = eta_dat.confint(expansion_factor)[1] - eta_nom
        expansion_factor = float(
            np.max(eta_utot / eta_dat.stdunc(k=expansion_factor).cov)
        )
    # re calculate with the worst case
    eta_utot = eta_dat.stdunc(k=expansion_factor).cov
    del eta_dat, eta_grouped

    # only need the nominal here
    s11_nom = s11_dat.nom[:, 0]
    s11_abs = np.abs(s11_nom)
    s11_angle = np.angle(s11_nom, deg=True)
    del s11_dat, s11_nom

    data_line = [f'{{:{3 + frequency_decimals}.{frequency_decimals}f}}']
    data_line += [f'{{:{2 + s11_abs_decimals}.{s11_abs_decimals}f}}']
    data_line += [f'{{:{5 + s11_angle_decimals}.{s11_angle_decimals}f}}']
    data_line += [f'{{:{2 + eta_decimals}.{eta_decimals}f}}'] * 4
    data_line = (' ' * 7).join(data_line)
    # print(data_line)
    # calculate eta uncertainty
    with open(output_path, 'w') as f:

        def line(line: str, nl='\n'):
            f.write(line + nl)

        if sensor_name:
            line(f'# Gamma and Effective Efficiency of {sensor_name}')
        if connect_number:
            line(f'# Eta: Connect {connect_number}')
        line(f'# COL={columns}, File created {now}')
        line(f'# ExpansionFactor={expansion_factor:.2f}')
        line(
            '#  freq        |Gam|       arg(Gam)      eta           uA           uB         Utot'
        )
        for i, fi in enumerate(flist):
            # print(s11_angle[i])
            dli = data_line.format(
                fi,
                s11_abs[i],
                s11_angle[i],
                eta_nom.sel(frequency=fi),
                eta_uA.sel(frequency=fi),
                eta_uB.sel(frequency=fi),
                eta_utot.sel(frequency=fi),
            )
            # print(dli)
            if i == len(flist) - 1:
                term = ''
            else:
                term = '\n'
            line(dli, nl=term)
