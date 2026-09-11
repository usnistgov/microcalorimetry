from rmellipse.utils import load_object
import h5py
from rmellipse.arrschema import ArraySchema, AnnotatedArray
from rmellipse.uobjects import RMEMeas
import microcalorimetry.arrays as arrays

if __name__ == '__main__':
    with h5py.File(r'tests\test_analysis_scripted_refs\S24P02.h5', 'r') as f:
        sample_RFSweep: RMEMeas = load_object(f['S24P02/parsed_rf/E_on'])
        sample_RFSweep = arrays.as_annotated(sample_RFSweep, arrays.RFSweep)
