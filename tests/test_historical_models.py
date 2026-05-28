# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 09:09:52 2026

@author: dcg2
"""

import microcalorimetry.analysis as anl
import matplotlib.pyplot as plt
import h5py
from rmellipse.utils import save_object
from pathlib import Path


LOCAL = Path(__file__).parents[0]
MUTABLE = LOCAL / 'mutable_datafiles'
SAMPLE_HIST = LOCAL / 'sample_historical_data'


def test_historical_data(make_new_reference: bool = False):
    historical_files = [
        SAMPLE_HIST / 'C24N118.yml',
        SAMPLE_HIST / 'C24N132.yml',
    ]

    # generate a data defined model
    print('in test @ ', Path.cwd().resolve())
    data_model, fig, coeffs = anl.make_eta_repeatability_model(
        historical_files, make_plots=True
    )

    if make_new_reference:
        with h5py.File(SAMPLE_HIST / 'model.h5', 'w') as f:
            save_object(f, '24histmodel', data_model)


if __name__ == '__main__':
    plt.close('all')
    test_historical_data(make_new_reference=False)
    plt.show()
