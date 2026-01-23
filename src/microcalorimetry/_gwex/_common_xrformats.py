# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 10:19:37 2024

@author: dcg2
"""

import pandas as _pd
import numpy as _np
import xarray as _xr
import uuid
import microcalorimetry._gwex as gwex
from datetime import datetime

# %% Format Definitions
# generates a format of size N of frequency dependent variables

# %% w1p_ri


def _from_csv_w1p_ri(path, delimiter=' ', comment='!'):
    d = _pd.read_csv(path, delimiter=delimiter, comment=comment, header=None)
    d = d.to_numpy()
    f = d[:, 0]
    d = d[:, 1:5]
    xcoords = {'frequency': f}

    meta = {}
    header = gwex.get_header(path, comment)
    meta['header'] = header
    return xcoords, d, meta


def _to_csv_w1p_ri(xr, path):
    d = xr.to_pandas()
    d.to_csv(path, sep=r' ', header=None)


w1p_ri = gwex.DataFormat(
    name='w1p_ri',
    xdims=['frequency'],
    ycoords={'w': ['Re(a11)', 'Im(a11)', 'Re(b11)', 'Im(b11)']},
    dtype=float,
    category='s-parameters',
    csv_file_extension='.w1p',
    from_csv=_from_csv_w1p_ri,
    to_csv=_to_csv_w1p_ri,
    description='1 port real/imaginary  wave params.',
)

# %% w1p_c

w1p_c = gwex.DataFormat(
    name='w1p_c',
    xdims=['frequency'],
    ycoords={'w': ['a11', 'b11']},
    dtype=complex,
    category='s-parameters',
    description='1 port complex  wave params.',
)


# %% w2p_ri
def _from_csv_w2p_ri(path, comment='#', skiprows=0):
    with open(path, 'r') as file:
        search = True
        while search:
            pos = file.tell()
            line = file.readline()
            if line.startswith('#') or line.startswith('!'):
                pass
            else:
                file.seek(pos)
                search = False
        d = _pd.read_csv(
            file, sep=r'\s+', comment=comment, header=None, skiprows=skiprows
        )
        d = d.to_numpy()
        f = d[:, 0]
        d = d[:, 1:]
        xcoords = {'frequency': f}

    meta = {}
    header = gwex.get_header(path, comment)
    meta['header'] = header
    return xcoords, d, meta


def _to_csv_w2p_ri(xr, path):
    d = xr.to_pandas()
    d.to_csv(path, sep=r' ', header=None)


w2p_ri = gwex.DataFormat(
    name='w2p_ri',
    xdims=['frequency'],
    ycoords={
        'w': [
            'Re(a11)',
            'Im(a11)',
            'Re(b11)',
            'Im(b11)',
            'Re(a21)',
            'Im(a21)',
            'Re(b21)',
            'Im(b21)',
            'Re(a12)',
            'Im(a12)',
            'Re(b12)',
            'Im(b12)',
            'Re(a22)',
            'Im(a22)',
            'Re(b22)',
            'Im(b22)',
        ]
    },
    dtype=float,
    category='s-parameters',
    csv_file_extension='.w2p',
    from_csv=_from_csv_w2p_ri,
    to_csv=_to_csv_w2p_ri,
    description='2 port real/imaginary wave params.',
)

# %% w2p_c

w2p_c = gwex.DataFormat(
    name='w2p_c',
    xdims=['frequency'],
    ycoords={
        'w': [
            'a11',
            'b11',
            'a21',
            'b21',
            'a12',
            'b12',
            'a22',
            'b22',
        ]
    },
    dtype=complex,
    category='s-parameters',
    description='2 port real/imaginary wave params.',
)

# %% s1p_ri


def _from_csv_s1p_ri(path, comment='#'):
    d = _pd.read_csv(path, sep=r'\s+', comment=comment, header=None)
    d = d.to_numpy()
    f = d[:, 0]
    d = d[:, 1:5]
    xcoords = {'frequency': f}

    meta = {}
    header = gwex.get_header(path, comment)
    meta['header'] = header
    return xcoords, d, meta


def _to_csv_s1p_ri(xr, path):
    f = _np.expand_dims(xr.frequency.to_numpy(), axis=1)
    d = xr.to_numpy()
    out = _np.concatenate((f, d), axis=1)
    _np.savetxt(path, out, fmt='%.18e', delimiter='\t')


s1p_ri = gwex.DataFormat(
    name='s1p_ri',
    xdims=['frequency'],
    ycoords={'s': ['Re(S11)', 'Im(S11)']},
    dtype=float,
    category='s-parameters',
    csv_file_extension='.s1p',
    from_csv=_from_csv_s1p_ri,
    to_csv=_to_csv_s1p_ri,
    description='1 port real/imaginary s-params.',
)

# %% dut_s1p
# Some files have these uncertainties, some do not


def _from_csv_duts1p(path, comment='!'):
    d = _pd.read_csv(path, header=None, sep=r'\s+', comment=comment)
    d = d.to_numpy()
    nom = _np.expand_dims(d[:, [1, 2]], 0)
    index = d[:, 0]
    if len(d[0]) == 11:
        plocs = ['Ub|G|', 'Un|G|', 'Uc|G|', 'UbA(G)', 'UnA(G)', 'UcA(G)']
        pilocs = [3, 4, 5, 7, 8, 9]
        perts = _np.zeros([6] + list(d[:, [1, 2]].shape))
    elif len(d[0]) == 3:
        plocs = []
        pilocs = []
        perts = _np.zeros([0] + list(d[:, [1, 2]].shape))

    for i, (p, pi) in enumerate(zip(plocs, pilocs)):
        perts[i, :, :] = nom
        if '|' in p:
            perts[i, :, 0] += d[:, pi]
        else:
            perts[i, :, 1] += d[:, pi]
    plocs = ['nominal'] + plocs
    data = _np.concatenate([nom, perts])
    xcoords = {'frequency': index, 'umech_id': plocs}

    meta = {}
    header = gwex.get_header(path, comment)
    meta['header'] = header
    return xcoords, data, meta


def _to_csv_duts1p(xr, path, k=1):
    print('USING K = ' + str(k) + ' Be careful out there')
    d = _np.zeros(shape=tuple([xr.shape[1]] + [11]))
    d[:, 0] = _np.round(
        xr['frequency'].values, 6
    )  # To get rid of floating point errors, am unsure if it rounds digits or decimals (so to 1 MHz). numpy's format_float_positional might be useful
    d[:, 1] = xr.loc[..., '|Gamma|'][0].values
    d[:, 2] = xr.loc[..., 'Arg(Gamma)'][0].values
    d[:, 3] = _np.transpose(
        (k * (((xr - xr[0, ...]) ** 2).sum(dim='umech_id')) ** 0.5)
        .sel(col='|Gamma|')
        .values
    )
    d[:, 4] = _np.zeros(shape=d[:, 4].shape)
    d[:, 5] = _np.zeros(shape=d[:, 5].shape)
    d[:, 6] = _np.transpose(
        (k * (((xr - xr[0, ...]) ** 2).sum(dim='umech_id')) ** 0.5)
        .sel(col='|Gamma|')
        .values
    )
    d[:, 7] = _np.transpose(
        (k * (((xr - xr[0, ...]) ** 2).sum(dim='umech_id')) ** 0.5)
        .sel(col='Arg(Gamma)')
        .values
    )
    d[:, 8] = _np.zeros(shape=d[:, 8].shape)
    d[:, 9] = _np.zeros(shape=d[:, 9].shape)
    d[:, 10] = _np.transpose(
        (k * (((xr - xr[0, ...]) ** 2).sum(dim='umech_id')) ** 0.5)
        .sel(col='Arg(Gamma)')
        .values
    )

    _np.savetxt(path, d, fmt='%.10e', delimiter=' ')

    # For legacy GammaG
    # d = _np.zeros(shape=tuple([xr.shape[1]] + [3]))
    # d[:, 0] = _np.round(xr['frequency'].values, 6)  # To get rid of floating point errors, am unsure if it rounds digits or decimals (so to 1 MHz). numpy's format_float_positional might be useful
    # d[:, 1] = xr.loc[..., '|Gamma|'][0].values
    # d[:, 2] = xr.loc[..., 'Arg(Gamma)'][0].values
    # d = _np.round(d, precision=5)
    # _np.savetxt(path, d, '%f', delimiter=' ')


dut_s1p = gwex.DataFormat(
    name='dut_s1p',
    xdims=['umech_id', 'frequency'],
    ycoords={'col': ['|Gamma|', 'Arg(Gamma)']},
    dtype=float,
    from_csv=_from_csv_duts1p,
    to_csv=_to_csv_duts1p,
    csv_file_extension='.dut',
    category='s-parameters',
    description='cal services mag/arg 1-port s parameters.',
)

# %% asc_s1p
asc_s1p = gwex.DataFormat(
    name='asc_s1p',
    xdims=['umech_id', 'frequency'],
    ycoords={'col': ['|Gamma|', 'Arg(Gamma)']},
    dtype=float,
    from_csv=_from_csv_duts1p,
    csv_file_extension='.asc',
    category='s-parameters',
    description='cal services mag/arg 1-port s parameters.',
)

# %% s1p_c

s1p_c = gwex.DataFormat(
    name='s1p_c',
    xdims=['frequency'],
    ycoords={'s': ['S11']},
    dtype=complex,
    category='s-parameters',
    description='1 port complex s-params.',
)

# %% s2p_ri


def _from_csv_s2p_ri(path, comment='#'):
    d = _pd.read_csv(path, delimiter='\t', header=None, comment=comment)
    d = d.to_numpy()
    index = d[:, 0]
    data = d[:, 1:]
    xcoords = {'frequency': index}

    meta = {}
    header = gwex.get_header(path, comment)
    meta['header'] = header
    return xcoords, data, meta


def _to_csv_s2p_ri(xr, path):
    f = _np.expand_dims(xr.frequency.to_numpy(), axis=1)
    d = xr.to_numpy()
    out = _np.concatenate((f, d), axis=1)
    f = open(path, 'w')  # Open file for write, overwriting file if it exits
    f.write('# GHz S RI R 50\n')  # TouchStone header line
    f.close()  # Release the file
    _np.savetxt(path, out, fmt='%.18e', delimiter='\t')


s2p_ri = gwex.DataFormat(
    name='s2p_ri',
    xdims=['frequency'],
    ycoords={
        's': [
            'Re(S11)',
            'Im(S11)',
            'Re(S12)',
            'Im(S12)',
            'Re(S21)',
            'Im(S21)',
            'Re(S22)',
            'Im(S22)',
        ]
    },
    dtype=float,
    category='s-parameters',
    csv_file_extension='.s2p',
    from_csv=_from_csv_s2p_ri,
    to_csv=_to_csv_s2p_ri,
    description='2 port real/imaginary s-params.',
)


# %% s2p_c

s2p_c = gwex.DataFormat(
    name='s2p_c',
    xdims=['frequency'],
    # ydims=['srow', 'scol'],
    ycoords={'srow': [1, 2], 'scol': [1, 2]},
    # The below comment might be solved already, not sure - Zenn
    # This one will be weird, the shape is 2, 2 but was told it will always be
    # len(ycoords[key]). I can make it overwritable if its actually 2x2. The above one is not
    # yshape=(2, 2),
    dtype=complex,
    category='s-parameters',
    description='2 port complex s-params.',
)


# %% t2p_baab
t2p_baab = gwex.DataFormat(
    name='t2p_baab',
    xdims=['frequency'],
    # ydims=['srow', 'scol'],
    ycoords={'trow': [1, 2], 'tcol': [1, 2]},
    # This one will be weird, the shape is 2, 2 but was told it will always be
    # len(ycoords[key]). I can make it overwritable if its actually 2x2. The above one is not
    # yshape=(2, 2),
    dtype=complex,
    category='s-parameters',
    description='2 port complex t-params, baab format.',
)

# %% t2p_abab
t2p_abab = gwex.DataFormat(
    name='t2p_abab',
    xdims=['frequency'],
    # ydims=['srow', 'scol'],
    ycoords={'trow': [1, 2], 'tcol': [1, 2]},
    # This one will be weird, the shape is 2, 2 but was told it will always be
    # len(ycoords[key]). I can make it overwritable if its actually 2x2. The above one is not
    # yshape=(2, 2),
    dtype=complex,
    category='s-parameters',
    description='2 port complex t-params, abab format.',
)

# %% errorbox_1p
errbox_1p = gwex.DataFormat(
    name='errbox_1p',
    xdims=['frequency'],
    ycoords={'errterm': ['E11', 'E22', 'delta']},
    dtype=complex,
    category='s-parameters',
    description='1 port error box model.',
)

# %% correlated error box


def _from_errbox4pRMAwave(path, comment='#'):
    d = _pd.read_csv(path, sep=r'\s+', comment=comment, header=None)
    d = d.to_numpy()
    f = d[:, 0]
    d = d[:, 1:]
    xcoords = {'frequency': f}

    meta = {}
    header = gwex.get_header(path, comment)
    meta['header'] = header
    return xcoords, d, meta


errbox_4p_RMAwave = gwex.DataFormat(
    name='errbox_4p_RMAwave',
    xdims=['frequency'],
    ycoords={'col': ['Re', 'Im']},
    dtype=float,
    category='s-parameters',
    description='4 port errbox model with all the terms in an errorbox stacked on top of one another. Specialy format from RMAwave.',
    from_csv=_from_errbox4pRMAwave,
    csv_file_extension='.complex',
)

# %% calorimeter_sensitivity


def _from_csv_mck(path, comment='#'):
    d = _pd.read_csv(path, delimiter=r'\s+', comment=comment, header=None)
    d = (d.to_numpy())[:, 0]

    meta = {}
    header = gwex.get_header(path, comment)
    meta['header'] = header
    return {}, d, meta


def _to_csv_mck(xr, path):
    d = xr.to_numpy()
    _np.savetxt(path, d, fmt='%.18e', delimiter='\t')


mck = gwex.DataFormat(
    name='mck',
    xdims=[],
    ycoords={'k': ['quadratic', 'slope', 'offset']},
    dtype=float,
    csv_file_extension='.mck',
    from_csv=_from_csv_mck,
    to_csv=_to_csv_mck,
    category='power',
    description='microcalorimeter sensitivity coefficients.',
)

# %% calorimeter historical matlab format


def _from_calml(path: str, comment='"'):
    """
    Read in a calorimeter matlab file.

    Notes from the original matlab code:
    % The standard file columns are time, frequency, vbias, and vpile
    % A 2nd bias voltage is added between vbias and vpile (i.e. col 4).  When there are two
    %    bias voltages, the one in column 3 was used to level the power in the Type 2 loop
    %    while the one in column 4 is the other.  Data files since spring have a "calo" or
    %    "side" in the header to indicate whether the bias voltage in column 3 represents the
    %    calorimeter or sidearm voltage
    % An EIP frequency is added as the last column.
    % Both additions were done this way for historical reasons.  (i.e. we threw thing
    %    together and hoped it worked)

    Parameters
    ----------
    path : str
        Path to the file.

    Returns
    -------
    xcoords : dict
        xarray coordinates of the x-dimensions, key matches label.
    out : ndarray
        numpy array of the values for the xarray that will be created.

    """
    d = _pd.read_csv(path, comment=comment).to_numpy()
    out = d
    # standard calorimeter run, put zeros for sidearm and eip voltage
    if d.shape[1] == 4:
        dummy = _np.zeros((d.shape[0]))
        out = _np.insert(d, 3, dummy, axis=1)
        out = _np.insert(out, 5, dummy, axis=1)

    # could be a side
    else:
        raise Exception(
            'Ability to parse historical matlab files with sidearms/EIP frequency not yet added.'
        )

    xcoords = {'timestamp': out[:, 0]}

    meta = {}
    header = gwex.get_header(path, comment)
    meta['header'] = header
    return xcoords, out[:, 1:], meta


calml = gwex.DataFormat(
    name='calml',
    xdims=['timestamp'],
    ycoords={'col': ['frequency', 'v_bias', 'v3_bias', 'e', 'EIP Freq']},
    dtype=float,
    csv_file_extension='',
    from_csv=_from_calml,
    description='legacy microcalorimeter matlab format, no sidearm',
)

# %% eff_7 columns


def _from_csv_eff_7(path, comment='#'):
    d = _pd.read_csv(path, header=None, sep=r'\s+', comment=comment)
    d = d.to_numpy()
    nom = _np.expand_dims(d[:, [1, 2, 3]], 0)
    index = d[:, 0]
    plocs = ['uA', 'uB']
    pilocs = [4, 5]
    perts = _np.zeros([2] + list(d[:, [1, 2, 3]].shape))
    for i, pi in enumerate(pilocs):
        perts[i, :, :] = nom
        perts[i, :, 2] += d[:, pi]
    plocs = ['nominal'] + plocs
    data = _np.concatenate([nom, perts])
    xcoords = {'frequency': index, 'umech_id': plocs}

    meta = {}
    header = gwex.get_header(path, comment)
    meta['header'] = header
    return xcoords, data, meta


def _to_csv_eff_7(xr, path, k=1):
    print(
        'For Eff7 using k = '
        + str(k)
        + ', it is a direct std*k cause idk how to use the dof thing in this function'
    )
    d = _np.zeros(shape=tuple([xr.shape[1]] + [xr.shape[2] + 4]))
    d[:, 0] = _np.round(
        xr['frequency'].values, 6
    )  # To get rid of floating point errors
    d[:, 1] = xr.loc[..., '|Gamma|'][0]
    d[:, 2] = xr.loc[..., 'Arg(Gamma)'][0]
    d[:, 3] = xr.loc[..., 'eta'][0]
    d[:, 4] = _np.zeros(shape=d[:, 3].shape)
    d[:, 5] = (
        _np.transpose(
            ((((xr - xr[0, ...]) ** 2).sum(dim='umech_id')) ** 0.5)
            .sel(col='eta')
            .values
        )
        * k
    )
    d[:, 6] = (
        _np.transpose(
            ((((xr - xr[0, ...]) ** 2).sum(dim='umech_id')) ** 0.5)
            .sel(col='eta')
            .values
        )
        * k
    )

    _np.savetxt(path, d, fmt='%.18e', delimiter=' ')


eff_7 = gwex.DataFormat(
    name='eff_7',
    xdims=['umech_id', 'frequency'],
    ycoords={'col': ['|Gamma|', 'Arg(Gamma)', 'eta']},
    dtype=float,
    from_csv=_from_csv_eff_7,
    to_csv=_to_csv_eff_7,
    csv_file_extension='.eff',
    category='power',
    description='cal services effective efficiency 7 column; device gamma present without uncertainties',
)

# %% eff_5 columns


def _from_csv_eff_5(path, comment='#'):
    d = _pd.read_csv(path, header=None, sep=r'\s+', comment=comment)
    d = d.to_numpy()
    nom = _np.expand_dims(d[:, [1]], 0)
    index = d[:, 0]
    plocs = ['uA', 'uB']
    pilocs = [2, 3]
    perts = _np.zeros([2] + list(d[:, [1]].shape))
    for i, pi in enumerate(pilocs):
        perts[i, :, :] = nom
        perts[i, :, 0] += d[:, pi]
    plocs = ['nominal'] + plocs
    data = _np.concatenate([nom, perts])
    xcoords = {'frequency': index, 'umech_id': plocs}

    meta = {}
    header = gwex.get_header(path, comment)
    meta['header'] = header
    return xcoords, data, meta


eff_5 = gwex.DataFormat(
    name='eff_5',
    xdims=['umech_id', 'frequency'],
    ycoords={'col': ['eta']},
    dtype=float,
    from_csv=_from_csv_eff_5,
    csv_file_extension='.eff',
    category='power',
    description='cal services effective efficiency 5 column; device gamma present without uncertainties',
)
# %% eff_2 columns


def _from_csv_eff_2(path, comment='#'):
    d = _pd.read_csv(path, header=None, sep=r'\s+', comment=comment)
    d = d.to_numpy()
    nom = _np.expand_dims(d[:, [1]], 0)
    index = d[:, 0]
    xcoords = {'frequency': index, 'umech_id': ['nominal']}

    meta = {}
    header = gwex.get_header(path, comment)
    meta['header'] = header
    return xcoords, nom, meta


eff_2 = gwex.DataFormat(
    name='eff_2',
    xdims=['umech_id', 'frequency'],
    ycoords={'col': ['eta']},
    dtype=float,
    from_csv=_from_csv_eff_2,
    csv_file_extension='.eff',
    category='power',
    description='cal services effective efficiency 2 column; device gamma present without uncertainties',
)

# %% eff standard computation format for eff
eff = gwex.DataFormat(
    name='eff',
    xdims=['frequency'],
    ycoords={'eta': [0]},
    dtype=float,
    category='power',
    description='Standard effective efficiency format for computation',
)


# %% 1 term calorimeteric gc model

gc1 = gwex.DataFormat(
    name='gc1',
    xdims=['frequency'],
    ycoords={'gc': [0]},
    dtype=float,
    category='power',
    description='1-term microcalorimeter correction model.',
)

gc2 = gwex.DataFormat(
    name='gc2',
    xdims=['frequency'],
    ycoords={'gc': [0, 1]},
    dtype=float,
    category='power',
    description='2-term microcalorimeter correction model.',
)

gc3 = gwex.DataFormat(
    name='gc3',
    xdims=['frequency'],
    ycoords={'gc': [0, 1, 2]},
    dtype=float,
    category='power',
    description='3-term microcalorimeter correction model.',
)

gc4 = gwex.DataFormat(
    name='gc4',
    xdims=['frequency'],
    ycoords={'gc': [0, 1, 2, 3]},
    dtype=float,
    category='power',
    description='4-term microcalorimeter correction model.',
)

# %% ASCII files for check standards


def _from_csv_ascii_power(path, comment='#'):
    d = _pd.read_csv(path, header=None, sep=r'\s+', comment=comment)
    d = d.to_numpy()
    nom_obj = _np.expand_dims(d[:, [2, 3, 4, 5]], 0)
    nom = nom_obj.astype(float)
    data = nom
    index = d[:, 1].astype(float)
    plocs = ['nominal']
    if len(d[0]) == 9:
        perts = _np.zeros([3] + list(d[:, [2, 3, 4, 5]].shape))
        uncs = ['uGMB', 'uGAB', 'uEB']
        perts[:, :, :] = nom
        perts[0, :, 1] += d[:, 6].astype(float)
        perts[1, :, 2] += d[:, 7].astype(float)
        perts[2, :, 3] += d[:, 8].astype(float)
        plocs = ['nominal'] + uncs
        data = _np.concatenate([nom, perts])
    xcoords = {'frequency': index, 'umech_id': plocs}

    meta = {}
    header = gwex.get_header(path, comment)
    meta['header'] = header
    return xcoords, data, meta


ascii_power = gwex.DataFormat(
    name='ascii_power',
    xdims=['umech_id', 'frequency'],
    ycoords={'col': ['repeats', '|Gamma|', 'Arg(Gamma)', 'eta']},
    dtype=float,
    from_csv=_from_csv_ascii_power,
    csv_file_extension='',
    category='power',
    description='average ascii files for power (1 - port) check standards',
)


def _from_csv_ascii_s1p(path, comment='#'):
    d = _pd.read_csv(path, header=None, sep=r'\s+', comment=comment)
    d = d.to_numpy()
    nom_obj = _np.expand_dims(d[:, [2, 3, 4]], 0)
    nom = nom_obj.astype(float)
    data = nom
    index = d[:, 1].astype(float)
    plocs = ['nominal']
    if len(d[0]) == 7:
        perts = _np.zeros([2] + list(d[:, [2, 3, 4]].shape))
        uncs = ['uGMB', 'uGAB']
        perts[:, :, :] = nom
        perts[0, :, 1] += d[:, 5].astype(float)
        perts[0, :, 2] += d[:, 6].astype(float)
        plocs = ['nominal'] + uncs
        data = _np.concatenate([nom, perts])
    xcoords = {'frequency': index, 'umech_id': plocs}

    meta = {}
    header = gwex.get_header(path, comment)
    meta['header'] = header
    return xcoords, data, meta


ascii_s1p = gwex.DataFormat(
    name='ascii_s1p',
    xdims=['umech_id', 'frequency'],
    ycoords={'col': ['repeats', '|Gamma|', 'Arg(Gamma)']},
    dtype=float,
    from_csv=_from_csv_ascii_s1p,
    csv_file_extension='',
    category='s-parameters',
    description='average ascii files for (1 - port) check standards',
)


# %% ASC (cumulative) files for direct comparison
def _from_csv_asc_power(path, comment='#', skiprows=24):
    d = _pd.read_csv(path, header=None, sep=r'\s+', comment=comment, skiprows=skiprows)
    d = d.to_numpy()
    nom = _np.expand_dims(d[:, [1, 2, 9, 10, 11]], 0)
    index = d[:, 0]
    plocs = [
        'udc',
        'us',
        'ua',
        'upm',
        'uad',
        'Mgtb',
        'Mgsn',
        'Mgsc',
        'Agtb',
        'Agsn',
        'Agsc',
    ]
    pilocs = [3, 4, 5, 6, 7, 12, 13, 14, 16, 17, 18]
    perts = _np.zeros([len(pilocs)] + list(nom.shape)[1:3])

    for i, (p, pi) in enumerate(zip(plocs, pilocs)):
        perts[i, :, :] = nom
        if '|' in p:
            perts[i, :, 0] += d[:, pi]
        else:
            perts[i, :, 1] += d[:, pi]
    plocs = ['nominal'] + plocs
    data = _np.concatenate([nom, perts])
    xcoords = {'frequency': index, 'umech_id': plocs}

    meta = {}
    header = gwex.get_header(path, comment)
    meta['header'] = header
    return xcoords, data, meta


asc_power = gwex.DataFormat(
    name='asc_power',
    xdims=['umech_id', 'frequency'],
    ycoords={'col': ['eta', 'Cf', '|Gamma|', 'Arg(Gamma)', 'Pdut']},
    dtype=float,
    from_csv=_from_csv_asc_power,
    csv_file_extension='',
    category='power',
    description='cumulative files for direct comparison',
)

# %% Single historical file NIST


def _from_hist_s2p(path, comment='#', skiprows=21):
    d = _pd.read_csv(path, header=None, comment=comment, skiprows=skiprows).to_numpy()

    connects = _np.unique(d[:, 2]).astype(int)
    frequencies = _np.unique(d[:, 0])
    data = _np.zeros((len(connects), len(frequencies), 4))
    try:
        for c in connects:
            # print(c)
            data[c - 1, :, :] = d[d[:, 2].astype(int) == c, 3:]
        xcoords = {'connect': connects, 'frequency': frequencies}
    except ValueError as e:
        raise Exception(
            ' failed to read hist_s2p. Most likely a repeat frequency in the file due to rounding.'
        )

    meta = {}
    header = gwex.get_header(path, comment)
    meta['header'] = header
    return xcoords, data, meta


hist_s2p = gwex.DataFormat(
    name='hist_s2p',
    xdims=['connect', 'frequency'],
    ycoords={
        'col': [
            'Port-1 |Gamma|',
            'Port-1 Arg(Gamma)',
            'Port-2 |Gamma|',
            'Port-2 Arg(Gamma)',
        ]
    },
    dtype=float,
    from_csv=_from_hist_s2p,
    csv_file_extension='',
    category='s-parameters',
    description="historical file for single measurement of s1p. Files are used to make average ascii's",
)


# %% Data Format Conversions

# %% converting to w1p_ri


def _w1pc_to_w1pri(out_data, *in_data):
    out_data.loc[..., 'Re(a11)'] = _np.real(in_data[0].loc[..., 'a11'])
    out_data.loc[..., 'Im(a11)'] = _np.real(in_data[0].loc[..., 'a11'])
    out_data.loc[..., 'Re(b11)'] = _np.real(in_data[0].loc[..., 'b11'])
    out_data.loc[..., 'Im(b11)'] = _np.real(in_data[0].loc[..., 'b11'])
    return out_data


w1p_ri.add_converter(_w1pc_to_w1pri, w1p_c)


# %% converting to w1p_c

# TODO
# def _w2pri_to_w2pc(out_data, in_data):
#     strs = ["a11", "b11", "a21", "b21", "a12", "b12", "a22", "b22"]
#     for s in strs:
#         re = 'Re(' + s + ')'
#         im = 'Im(' + s + ')'
#         out_data.loc[..., s] = in_data[0].loc[..., re] + 1.0j * in_data[0].loc[..., im]
#     return out_data

# w2p_c.add_converter(_w2pri_to_w2pc, w2p_ri)

# %% converting to w2p_c


def _w2pri_to_w2pc(out_data, in_data):
    strs = ['a11', 'b11', 'a21', 'b21', 'a12', 'b12', 'a22', 'b22']
    for s in strs:
        re = 'Re(' + s + ')'
        im = 'Im(' + s + ')'
        out_data.loc[..., s] = in_data[0].loc[..., re] + 1.0j * in_data[0].loc[..., im]
    return out_data


w2p_c.add_converter(_w2pri_to_w2pc, w2p_ri)


# %% converting to t2p_baab


def _s2pcd_to_t2p_baab(out_data, in_data):
    S11 = in_data.loc[..., 1, 1]
    S22 = in_data.loc[..., 2, 2]
    S12 = in_data.loc[..., 1, 2]
    S21 = in_data.loc[..., 2, 1]

    out_data.loc[..., 1, 1] = -(S11 * S22 - S21 * S12) / S21
    out_data.loc[..., 1, 2] = S11 / S21
    out_data.loc[..., 2, 1] = -S22 / S21
    out_data.loc[..., 2, 2] = 1.0 / S21
    return out_data


t2p_baab.add_converter(_s2pcd_to_t2p_baab, s2p_c)


# %% converting to t2p_abab


def _s2pc_to_t2p_abab(out_data, in_data):
    S11 = in_data.loc[..., 1, 1]
    S22 = in_data.loc[..., 2, 2]
    S12 = in_data.loc[..., 1, 2]
    S21 = in_data.loc[..., 2, 1]

    out_data.loc[..., 1, 1] = -S22 / S21
    out_data.loc[..., 1, 2] = 1.0 / S21
    out_data.loc[..., 2, 1] = (S12 * S21 - S11 * S22) / S21
    out_data.loc[..., 2, 2] = S11 / S21
    return out_data


t2p_abab.add_converter(_s2pc_to_t2p_abab, s2p_c)


# %% converting to eff_7


def _effands1pc_to_eff7(out_data, *in_data):
    if in_data[0].dfm.dataformat.name == 'eff':
        eff_data = in_data[0]
        gamma_data = in_data[1]
    else:
        eff_data = in_data[1]
        gamma_data = in_data[0]
    out_data.loc[..., 'eta'] = eff_data.values[..., 0]
    out_data.loc[..., '|Gamma|'] = abs(gamma_data.loc[..., 'S11'])
    out_data.loc[..., 'Arg(Gamma)'] = (
        _np.angle(gamma_data.loc[..., 'S11']) / _np.pi * 180
    )

    return out_data


eff_7.add_converter(_effands1pc_to_eff7, eff, s1p_c)


# %% converting to duts1p


def _s1pc_to_duts1p(out_data, *in_data):
    out_data.loc[..., '|Gamma|'] = abs(in_data[0].loc[..., 'S11'])
    out_data.loc[..., 'Arg(Gamma)'] = (
        _np.angle(in_data[0].loc[..., 'S11']) / _np.pi * 180
    )
    return out_data


dut_s1p.add_converter(_s1pc_to_duts1p, s1p_c)


def _eff7_to_duts1p(out_data, *in_data):
    out_data.loc[..., '|Gamma|'] = in_data[0].loc[..., '|Gamma|']
    out_data.loc[..., 'Arg(Gamma)'] = in_data[0].loc[..., 'Arg(Gamma)']
    return out_data


dut_s1p.add_converter(_eff7_to_duts1p, eff_7)


# %% converting to s1p_ri


def _s1pc_to_s1pri(out_data, *in_data):
    out_data.loc[..., 'Re(S11)'] = _np.real(in_data[0].loc[..., 'S11'])
    out_data.loc[..., 'Im(S11)'] = _np.imag(in_data[0].loc[..., 'S11'])
    return out_data


s1p_ri.add_converter(_s1pc_to_s1pri, s1p_c)


# %% converting to s1p_c


def _w1pri_to_s1pc(out_data, *in_data):
    a11 = in_data[0].loc[..., 'Re(a11)'] + 1.0j * in_data[0].loc[..., 'Im(a11)']
    b11 = in_data[0].loc[..., 'Re(b11)'] + 1.0j * in_data[0].loc[..., 'Im(b11)']
    out_data.loc[..., 'S11'] = b11 / a11
    return out_data


s1p_c.add_converter(_w1pri_to_s1pc, w1p_ri)


def _w1pc_to_s1pc(out_data, *in_data):
    out_data.loc[..., 'S11'] = in_data[0].loc[..., 'b11'] / in_data[0].loc[..., 'a11']
    return out_data


s1p_c.add_converter(_w1pc_to_s1pc, w1p_c)


def _s2pc_to_s1pc(out_data, in_data):
    out_data.loc[..., 'S11'] = in_data.loc[..., 1, 1]
    return out_data


s1p_c.add_converter(_s2pc_to_s1pc, s2p_c)


def _s1pri_to_s1pc(out_data, *in_data):
    out_data.loc[..., 'S11'] = (
        in_data[0].loc[..., 'Re(S11)'] + 1.0j * in_data[0].loc[..., 'Im(S11)']
    )
    return out_data


s1p_c.add_converter(_s1pri_to_s1pc, s1p_ri)


def _s2pri_to_s1pc(out_data, *in_data):
    S11 = in_data[0].loc[..., 'Re(S11)'] + 1.0j * in_data[0].loc[..., 'Im(S11)']

    out_data[..., 0] = S11
    return out_data


s1p_c.add_converter(_s2pri_to_s1pc, s2p_ri)


def _duts1p_to_s1pc(out_data, *in_data):
    mag = in_data[0].loc[..., '|Gamma|']
    arg = _np.deg2rad(in_data[0].loc[..., 'Arg(Gamma)'])

    re = mag * _np.cos(arg)
    im = mag * _np.sin(arg)

    out_data.loc[..., 'S11'] = re + 1.0j * im
    return out_data


s1p_c.add_converter(_duts1p_to_s1pc, dut_s1p)


def _asciipower_to_s1pc(out_data, *in_data):
    mag = in_data[0].loc[..., '|Gamma|']
    arg = _np.deg2rad(in_data[0].loc[..., 'Arg(Gamma)'].values)

    re = mag * _np.cos(arg)
    im = mag * _np.sin(arg)

    out_data.loc[..., 'S11'] = re + 1.0j * im
    return out_data


s1p_c.add_converter(_asciipower_to_s1pc, ascii_power)


def _asciis1p_to_s1pc(out_data, *in_data):
    mag = in_data[0].loc[..., '|Gamma|']
    arg = _np.deg2rad(in_data[0].loc[..., 'Arg(Gamma)'].values)

    re = mag * _np.cos(arg)
    im = mag * _np.sin(arg)

    out_data.loc[..., 'S11'] = re + 1.0j * im
    return out_data


s1p_c.add_converter(_asciis1p_to_s1pc, ascii_s1p)


def hists2p_to_s1pc(out_data, *in_data):
    out_data.loc[..., 'S11'] = in_data[0].loc[..., 'Port-1 |Gamma|'] * _np.exp(
        1.0j * in_data[0].loc[..., 'Port-1 Arg(Gamma)'] * _np.pi / 180
    )
    return out_data


s1p_c.add_converter(hists2p_to_s1pc, hist_s2p)


# %% converting to s2p_ri


def _s2pc_to_s2pri(out_data, *in_data):
    out_data.loc[..., 'Re(S11)'] = _np.real(in_data[0].loc[..., 1, 1])
    out_data.loc[..., 'Im(S11)'] = _np.imag(in_data[0].loc[..., 1, 1])
    out_data.loc[..., 'Re(S12)'] = _np.real(in_data[0].loc[..., 1, 2])
    out_data.loc[..., 'Im(S12)'] = _np.imag(in_data[0].loc[..., 1, 2])
    out_data.loc[..., 'Re(S21)'] = _np.real(in_data[0].loc[..., 2, 1])
    out_data.loc[..., 'Im(S21)'] = _np.imag(in_data[0].loc[..., 2, 1])
    out_data.loc[..., 'Re(S22)'] = _np.real(in_data[0].loc[..., 2, 2])
    out_data.loc[..., 'Im(S22)'] = _np.imag(in_data[0].loc[..., 2, 2])
    return out_data


s2p_ri.add_converter(_s2pc_to_s2pri, s2p_c)


# %% converting to s2p_c


def _s1pc_to_s2pc(out_data, in_data):
    # make the s12/s21 small so t-parameters work
    out_data.loc[..., 1, 2] = 1e-9
    out_data.loc[..., 2, 1] = 1e-9
    out_data.loc[..., 1, 1] = in_data.loc[..., 'S11']
    out_data.loc[..., 2, 2] = in_data.loc[..., 'S11']
    return out_data


s2p_c.add_converter(_s1pc_to_s2pc, s1p_c)


def _err1term_to_s2pc(out_data, in_data):
    E00 = in_data.loc[..., 'E11']
    E11 = in_data.loc[..., 'E22']
    Delta = in_data.loc[..., 'delta']

    E01E10 = E00 * E11 - Delta
    E01 = _np.sqrt(E01E10)
    E10 = _np.sqrt(E01E10)

    out_data.loc[..., 1, 1] = E00
    out_data.loc[..., 1, 2] = E01
    out_data.loc[..., 2, 1] = E10
    out_data.loc[..., 2, 2] = E11

    return out_data


s2p_c.add_converter(_err1term_to_s2pc, errbox_1p)


def _s2pc_to_err1term(out_data, in_data):
    S11 = in_data.loc[..., 1, 1]
    S22 = in_data.loc[..., 2, 2]
    S12 = in_data.loc[..., 1, 2]
    S21 = in_data.loc[..., 2, 1]

    E00 = S11
    E11 = S22

    E01E10 = S12**2
    Delta = E00 * E11 - E01E10

    out_data.loc[..., 'E11'] = E00
    out_data.loc[..., 'E22'] = E11
    out_data.loc[..., 'delta'] = Delta

    return out_data


errbox_1p.add_converter(_s2pc_to_err1term, s2p_c)


def _s2pri_to_s2pc(out_data, *in_data):
    S11 = in_data[0].loc[..., 'Re(S11)'] + 1.0j * in_data[0].loc[..., 'Im(S11)']
    S12 = in_data[0].loc[..., 'Re(S12)'] + 1.0j * in_data[0].loc[..., 'Im(S12)']
    S21 = in_data[0].loc[..., 'Re(S21)'] + 1.0j * in_data[0].loc[..., 'Im(S21)']
    S22 = in_data[0].loc[..., 'Re(S22)'] + 1.0j * in_data[0].loc[..., 'Im(S22)']

    out_data.loc[..., 1, 1] = S11
    out_data.loc[..., 1, 2] = S12
    out_data.loc[..., 2, 1] = S21
    out_data.loc[..., 2, 2] = S22
    return out_data


s2p_c.add_converter(_s2pri_to_s2pc, s2p_ri)


def _t2pbaab_to_s2pc(out_data, in_data):
    T11 = in_data.loc[..., 1, 1]
    T12 = in_data.loc[..., 1, 2]
    T21 = in_data.loc[..., 2, 1]
    T22 = in_data.loc[..., 2, 2]

    out_data.loc[..., 1, 1] = T12 / T22
    out_data.loc[..., 1, 2] = (T11 * T22 - T12 * T21) / T22
    out_data.loc[..., 2, 1] = 1.0 / T22
    out_data.loc[..., 2, 2] = -T21 / T22
    return out_data


s2p_c.add_converter(_t2pbaab_to_s2pc, t2p_baab)


def _t2pabab_to_s2pc(out_data, in_data):
    T11 = in_data.loc[..., 1, 1]
    T12 = in_data.loc[..., 1, 2]
    T21 = in_data.loc[..., 2, 1]
    T22 = in_data.loc[..., 2, 2]

    out_data.loc[..., 1, 1] = T22 / T12
    out_data.loc[..., 1, 2] = (-T11 * T22 + T12 * T21) / T12
    out_data.loc[..., 2, 1] = 1.0 / T12
    out_data.loc[..., 2, 2] = -T11 / T12
    return out_data


s2p_c.add_converter(_t2pabab_to_s2pc, t2p_abab)


def _w2pri_to_s2pc(out_data, in_data):
    # includes a switch term correction
    # port1 as source
    a1_1 = in_data.loc[..., 'Re(a11)'] + 1.0j * in_data.loc[..., 'Im(a11)']
    b1_1 = in_data.loc[..., 'Re(b11)'] + 1.0j * in_data.loc[..., 'Im(b11)']
    a2_1 = in_data.loc[..., 'Re(a21)'] + 1.0j * in_data.loc[..., 'Im(a21)']
    b2_1 = in_data.loc[..., 'Re(b21)'] + 1.0j * in_data.loc[..., 'Im(b21)']
    # port 2 as source
    a1_2 = in_data.loc[..., 'Re(a12)'] + 1.0j * in_data.loc[..., 'Im(a12)']
    b1_2 = in_data.loc[..., 'Re(b12)'] + 1.0j * in_data.loc[..., 'Im(b12)']
    a2_2 = in_data.loc[..., 'Re(a22)'] + 1.0j * in_data.loc[..., 'Im(a22)']
    b2_2 = in_data.loc[..., 'Re(b22)'] + 1.0j * in_data.loc[..., 'Im(b22)']
    # deltaGammaR = (reshaped[:,2]+1.0j*reshaped[:,3])*scale

    GammaF = a2_1 / b2_1
    GammaR = a1_2 / b1_2

    S11F = b1_1 / a1_1
    S21F = b2_1 / a1_1
    S12R = b1_2 / a2_2
    S22R = b2_2 / a2_2

    S11 = (S11F - S12R * S21F * GammaF) / (1.0 - S12R * S21F * GammaR * GammaF)
    S21 = (S21F - S22R * S21F * GammaF) / (1.0 - S12R * S21F * GammaR * GammaF)
    S12 = (S12R - S11F * S12R * GammaR) / (1.0 - S12R * S21F * GammaR * GammaF)
    S22 = (S22R - S12R * S21F * GammaR) / (1.0 - S12R * S21F * GammaR * GammaF)

    out_data.loc[..., 1, 1] = S11
    out_data.loc[..., 1, 2] = S12
    out_data.loc[..., 2, 1] = S21
    out_data.loc[..., 2, 2] = S22
    return out_data


s2p_c.add_converter(_w2pri_to_s2pc, w2p_ri)
