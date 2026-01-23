from pathlib import Path
import microcalorimetry.configs as configs
import microcalorimetry.measurements.dcsweep as sensitivity_mes
import rmellipse.utils as utils
import h5py


LOCAL = Path(__file__).parents[0]
MUTABLE = LOCAL / 'mutable_datafiles'
SAMPLE_CALRUNS = LOCAL / 'sample_calruns'
THINFILM_MODEL = '8474E-K01'
KSTE_MODEL = 'KSTE'
SRUNS = LOCAL / 'sample_sensitivity_runs'
S1P_FILES = LOCAL / 's1p_files'

LOCAL = Path(__file__).parents[0]
MUTABLE = LOCAL / 'mutable_datafiles'


def test_DataModelPointer():
    DMP = configs.DataModelContainer
    DMP('my-path1')
    DMP('my-path2.h5/my-group2')
    DMP('my-path2.hdf5/my-group2')


def test_loading_config_interoperability():
    dummy_file = MUTABLE / 'sample.h5'

    # parse the voltage steps
    metadata = SRUNS / 'C24S002_k_c000_r000' / 'voltage_stair_case_metadata.csv'
    saved, figures = sensitivity_mes.parse(
        metadata=metadata,
        measlist=metadata.parents[0] / 'meas_list.csv',
        make_plots=False,
    )

    # check that I can read it in and out
    with h5py.File(dummy_file, 'w') as f:
        utils.save_object(f, 'parsed_dcsweep', saved)
        read = utils.load_object(f['parsed_dcsweep'], load_big_objects=True)
        for k, v in read.items():
            vloaded = configs.DCSweep(v).load()
            vsloaded = configs.DCSweep(saved[k]).load()
            assert (vloaded.cov == vsloaded.cov).all()

        pntrs = configs.ParsedDCSweep(
            {'v': dummy_file / 'parsed_dcsweep/v', 'i': dummy_file / 'parsed_dcsweep/i'}
        )
        utils.save_object(f, 'pntrs', pntrs, verbose=True)

    for k, v in pntrs.items():
        vloaded = configs.DCSweep(v).load()
        vsloaded = configs.DCSweep(saved[k]).load()

        assert (vloaded.cov == vsloaded.cov).all()


if __name__ == '__main__':
    test_DataModelPointer()
    test_loading_config_interoperability()
