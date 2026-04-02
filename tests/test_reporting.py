import microcalorimetry.analysis as anl
import microcalorimetry.measurements.rfsweep as rfsweep
import microcalorimetry.measurements.dcsweep as dcsweep
import microcalorimetry.configs as configs
import os
import h5py
import io
from contextlib import redirect_stdout
from pathlib import Path
from rmellipse.utils import save_object
from copy import copy, deepcopy


LOCAL = Path(__file__).parents[0]
MUTABLE = LOCAL / 'mutable_datafiles'
SAMPLE_CALRUNS = LOCAL / 'sample_calruns'
THINFILM_MODEL = '8474E-K01'
KSTE_MODEL = 'KSTE'
SRUNS = LOCAL / 'sample_sensitivity_runs'
S1P_FILES = LOCAL / 's1p_files'
ETA_REFS = LOCAL / 'eta_references'

HISTORICAL_MODEL = LOCAL / 'sample_historical_data/model.h5/24histmodel'
HISTORICAL_DATA = configs.EtaHistorical(
    {
        '077': ETA_REFS / 'C24N118_077.eff',
        '078': ETA_REFS / 'C24N118_078.eff',
        '080': ETA_REFS / 'C24N118_080.eff',
        '082': ETA_REFS / 'C24N118_082.eff',
    }
)


def test_eta_reporting():
    anl.review_eta(
        LOCAL / 'test_analysis_scripted_refs/C24N118.h5' / 'eta',
        HISTORICAL_DATA,
        k=2,
    )
    pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    test_eta_reporting()
    plt.show()
