"""
This module is for test running the RF sweep measurement.

Certain aspects of this test can be run on CI, but the part
that actually connects tot he measurements and tries to run
will fail, so it is not included in the test suite and must
be executed manually at a lab station.

"""

import microcalorimetry.measurements.dcsweep as dcsweep
import microcalorimetry.math as ucalmath
import microcalorimetry.configs as configs
from pathlib import Path

LOCAL = Path(__file__).parents[0]
MUTABLE = LOCAL / 'mutable_datafiles' / 'delete-me'
MUTABLE.mkdir(parents=True, exist_ok=True)
REF_CONFIGS = LOCAL / 'reference_configs'


def _collect_test_set(directory: Path):
    """
    Test set is all the config files and runsettings in a  folder.

    runlist must be called 'runlist.{ext}' and everything else is
    assumed to be a config
    """
    measlist = directory / 'measlist.csv'
    measconfigs = directory / 'settings.csv'
    return measlist, measconfigs


def test_dc_configurations_set_1():
    dir = REF_CONFIGS / 'set_3_dcsweep_2nvms'
    measlist, measconfigs = _collect_test_set(dir)
    dcsweep.run(measconfigs, measlist, output_dir=MUTABLE, dry_run=True)


if __name__ == '__main__':
    test_dc_configurations_set_1()

    # #THIS WILL TRY TO RUN A MEASUREMENT
    # dir = REF_CONFIGS / 'set_3_dcsweep_2nvms'
    # measlist, measconfigs = _collect_test_set(dir)
    # dcsweep.run(
    #     measconfigs,
    #     measlist,
    #     output_dir=MUTABLE,
    #     dry_run = False
    # )
