"""
This module contains functions for exporting effective efficiency datasets.
"""

from __future__ import annotations
import microcalorimetry.configs as configs
from microcalorimetry.math import rmemeas_extras, vna
from microcalorimetry._tkquick.gui_dtypes import SaveAsPath
from rmellipse import RMEProp, RMEMeas
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime

__all__ = ['as_doteff', 'as_dotdut_s11']


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


def as_dotdut_s11(
    s11: configs.S11Like,
    path: SaveAsPath,
    device: str,
    expansion_factor: float = 2.0,
    ):
    """
    Export a device to a .dut file for 1 port s-parameters.
    
    This is a cal services 


    Parameters
    ----------
    s11 : configs.S11Like
        Path to output file.
    path : SaveAsPath
        RMEMeas measurement s1p_c data.
    device : str
        Name of device.
    expansion_factor : float, optional
        If >1 will use the provided expansion factor. 
        If < 1, will calculate the confidence interval 
        (i,e 0.95 will calculate the 95% confidence interval.

    Returns
    -------
    Path
        Path to exported DUT file.
    """
    data = configs.S11Like(s11).load()

    # assign everythin got type B as default
    data.create_empty_categories(['Type'])
    data.covcats.loc[:, 'Type'] = 'B'

    # assign type a inherted from primary standards
    a_markers = ['Drift', 'trace', 'cables', 'cable', 'Cable', 'PCA', 'additive']
    type_a_umechs = data.cov.umech_id[
        [False] + [any([am in oi for am in a_markers]) for oi in data.umech_id]
    ]
    data.covcats.loc[type_a_umechs, 'Type'] = 'A'

    # propagate to real/imaginary
    prop = RMEProp(sensitivity=True)
    get_mag = prop.propagate(vna.linmag)
    get_phs = prop.propagate(vna.phase)

    # calculate mag and phase, than 
    mag = group_typed_uncertainties(get_mag(data)).sel(s='S11')
    phs = group_typed_uncertainties(get_phs(data)).sel(s='S11') * 180 / np.pi
    assert 'uncategorized' not in mag.umech_id

    if expansion_factor < 1:
        mag_expansion = max(
            (mag.confint(expansion_factor, rad=True)[1] - mag.nom) / mag.stdunc(rad=True).cov
        )
        phase_expansion = max(
            (phs.confint(expansion_factor, rad=True)[1] - phs.nom) / phs.stdunc(rad=True).cov
        )
        expansion_factor = float(np.round(mag_expansion, 2))
        print(f'{device} .dut export using expansion factor {expansion_factor}')

    # calculate phase and magnitude uncertainties
    utot_mag = mag.stdunc(k=expansion_factor).cov
    ua_mag = mag.usel(umech_id=['A']).stdunc(k=1).cov
    ub_mag = mag.usel(umech_id=['B']).stdunc(k=1).cov
    uc_mag = ub_mag * 0

    utot_phs = phs.stdunc(k=expansion_factor, deg=True).cov
    ua_phs = phs.usel(umech_id=['A']).stdunc(k=1, deg=True).cov
    ub_phs = phs.usel(umech_id=['B']).stdunc(k=1, deg=True).cov
    uc_phs = ub_mag * 0

    with open(path, 'w') as f:
        now = datetime.today().strftime('%d %b %Y')
        f.write(f'! {device}    {now}\n')
        f.write(f'! Utot expansion factor = {expansion_factor}\n')
        f.write(f'! ')
        f.write(
            '! Freq, |Gamma|, Arg(Gamma), Ub|G|, Un|G|, Uc|G|, Utot|G|, UbA(G), UnA(G), UcA(G), UtotA(G)\n'
        )
        fmt = '{:9.6f} {:7.5f} {:8.3f} {:7.5f} {:7.5f} {:7.5f} {:7.5f} {:8.3f} {:8.3f} {:8.3f} {:8.3f}\n'
        for freq in mag.nom.frequency:
            f.write(
                fmt.format(
                    freq,
                    mag.nom.sel(frequency=freq),
                    phs.nom.sel(frequency=freq),
                    ub_mag.sel(frequency=freq),
                    ua_mag.sel(frequency=freq),
                    uc_mag.sel(frequency=freq),
                    utot_mag.sel(frequency=freq),
                    ub_phs.sel(frequency=freq),
                    ua_phs.sel(frequency=freq),
                    uc_phs.sel(frequency=freq),
                    utot_phs.sel(frequency=freq),
                )
            )


def as_doteff(
    output_path: SaveAsPath,
    eta: configs.EtaLike,
    s11: configs.S11Like,
    sensor_name: str = None,
    connect_number: int = None,
    expansion_factor: float = 2,
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
    eta : configs.EtaLike
        Effective efficiency of the sensor.
    s11 : configs.S11Like
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

    eta_dat = configs.EtaLike(eta).load()
    s11_dat = configs.S11Like(s11).load()

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
