"""
Tests the templates presented in the docs/config_templates.

"""

# test_spam.py
from pathlib import Path
import subprocess
import sys
import runpy
import pytest
import microcalorimetry.configs as configs
from microcalorimetry.measurements import rfsweep, dcsweep
from fnmatch import fnmatch

templates = Path(__file__).parents[1] / 'docs/config_templates'
templates = [t for t in templates.rglob('*') if t.is_dir()]

# for testing dc sweep templates
dcsweep_allowed_files = ['measlist.csv', 'settings.csv', 'README.md']
dcsweep_template_pattern = 'dcsweep*'

# for testing rfsweep templates
rfsweep_template_pattern = 'rfsweep*'


LOCAL = Path(__file__).parent
IGNORED = LOCAL / 'mutable_datafiles'


@pytest.mark.parametrize('template', templates)
def test_template(template: Path):
    """Dispatches templates to relevent testing functions."""
    if fnmatch(template.stem, dcsweep_template_pattern):
        check_dcsweep_template(template)
    elif fnmatch(template.stem, rfsweep_template_pattern):
        check_rfsweep_template(template)
    else:
        raise ValueError(
            f'{template.stem} template type not recognized or testable. If new type of template, add a check function and pattern to this module.'
        )


def check_dcsweep_template(template: Path):
    """Dry runs a DC sweep template."""
    files = []
    for f in template.glob('*'):
        if f.name not in dcsweep_allowed_files:
            raise ValueError(f'{f.name} not in {dcsweep_allowed_files}')
        files.append(f)
    # now run checks
    try:
        configs.DCSweepConfiguration(
            template / 'settings.csv',
        )
        dcsweep.run(
            template / 'settings.csv',
            template / 'measlist.csv',
            output_dir=IGNORED,
            dry_run=True,
        )
    except Exception as e:
        msg = f'caught on {template} : {e}'
        raise type(e)(msg) from e


def check_rfsweep_template(template: Path):
    meas_configs = [csv for csv in template.glob('*.csv')]
    print(meas_configs)
    runlist = template / 'runlist.csv'
    master_list = template / 'sensor_master_list.json'
    meas_configs.remove(runlist)
    rfsweep.run(
        IGNORED,
        1,
        meas_configs,
        runlist,
        sensor_master_list=master_list,
        dry_run=True,
        no_confirm=True,
    )


if __name__ == '__main__':
    for t in templates:
        test_template(t)
