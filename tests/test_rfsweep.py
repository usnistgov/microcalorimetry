"""
This module is for test running the RF sweep measurement.

Certain aspects of this test can be run on CI, but the part
that actually connects tot he measurements and tries to run
will fail, so it is not included in the test suite and must
be executed manually at a lab station.

"""

import microcalorimetry.measurements.rfsweep as rfsweep
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
    non_configs = ['runlist.csv', 'sensor_master_list.json']
    runlist = directory / 'runlist.csv'
    sensor_master_list = directory / 'sensor_master_list.json'
    configs = [f for f in directory.iterdir() if f.name not in non_configs]
    return configs, runlist, sensor_master_list


def test_rf_configurations_set_1():
    dir = REF_CONFIGS / 'set_1_rfsweep_te_commercial'
    configs, runlist, master_list = _collect_test_set(dir)
    rfsweep.run(
        output_dir=MUTABLE,
        repeats=1,
        configs=configs,
        settings=[runlist],
        sensor_master_list=master_list,
        name='dry-run',
        dry_run=True,
    )


if __name__ == '__main__':
    test_rf_configurations_set_1()

    # print(" THIS WILL TRY TO RUN AN EXPERIMENT")
    # dir = REF_CONFIGS / 'set_2_special_with_sidearm'
    # configs, runlist, master_list = _collect_test_set(dir)
    # rfsweep.run(
    #     output_dir = MUTABLE,
    #     repeats = 1,
    #     configs = configs,
    #     settings = [runlist],
    #     sensor_master_list=master_list,
    #     name = 'dry-run',
    #     dry_run = False
    # )
