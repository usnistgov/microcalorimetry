"""
Unit test for some of the conversion things, making sure some of them work (which means they all probably do).
This is not meant to be exhaustive: every combination is not tested, it is more to test that the meta code around
the converters works properly

If something here failed, it needs to be fixed :()

"""

import microcalorimetry._gwex as gwex
import numpy as np
import microcalorimetry._gwex._test_collections as gwex_test
from pathlib import Path
# import h5py

# mutable and constant reference data directories
MUTABLE = Path(__file__).parents[0] / 'mutable_datafiles'


def test_converters():
    # generate sample formats formats
    all_formats = gwex.DataFormat.formats
    samples = {
        name: gwex_test.make_sample(dfm, rand=False, extra_xdims=(4, 5))
        for name, dfm in all_formats.items()
    }

    # test functions in each format
    for name, sample in samples.items():
        dfm = all_formats[name]
        for from_names, fun in dfm.converters.items():
            from_names = from_names.split('; ')
            print(from_names, fun)
            from_samples = [samples[n] for n in from_names]
            gwex.convert(dfm, *from_samples)


def test_read_write():
    # generate sample formats formats
    all_formats = gwex.DataFormat.formats
    samples = {
        name: gwex_test.make_sample(dfm, rand=False)
        for name, dfm in all_formats.items()
    }

    # test functions in each format
    for name, sample in samples.items():
        dfm = all_formats[name]
        print(' ')
        if dfm.to_csv and dfm._from_csv:
            print('testing read/write on', name)
            gwex_test.make_sample(dfm)
            path = gwex.to_csv(MUTABLE, name, sample)
            read = gwex.from_csv(str(path), dfm)
            try:
                assert np.isclose(sample, read).all()
            except ValueError as e:
                print(e)
            pass
        elif dfm.to_csv:
            print(dfm.name + ' has write but no read')
        elif dfm._from_csv:
            print(dfm.name + ' has read but no write')
        else:
            print('No read/write on ', dfm.name)


def test_sample_functions():
    all_formats = gwex.DataFormat.formats
    {name: gwex_test.make_sample(dfm, rand=False) for name, dfm in all_formats.items()}


if __name__ == '__main__':
    # test_converters()
    test_read_write()
