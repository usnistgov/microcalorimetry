import microcalorimetry.configs
import microcalorimetry.export
import numpy as np
from pathlib import Path

LOCAL = Path(__file__).parents[0]
THINFILM_REF_DATA = LOCAL / 'test_analysis_scripted_refs' / 'C24N118.h5'
S1P_FILES = LOCAL / 's1p_files'
MUTABLE = LOCAL / 'mutable_datafiles'


def test_export_doteff():
    s11 = S1P_FILES / 'C24N118.dut'
    eta = THINFILM_REF_DATA / 'eta'
    path = MUTABLE / 'exported_eff.eff'
    eta_decimals = 4

    # test settings
    microcalorimetry.export.as_doteff(
        MUTABLE / 'exported_eff.eff',
        eta,
        s11,
        eta_decimals=eta_decimals
    )

    test_read = microcalorimetry.configs.Eta(path).load().nom
    eta_input = np.round(microcalorimetry.configs.Eta(eta).load().nom, eta_decimals)
    
    diff = np.abs(np.sum(test_read - eta_input))
    print('Test output and read difference sum: ', float(diff))
    # nominals should be within machine precision
    assert diff < 1e-12

if __name__ == '__main__':
    test_export_doteff()