import microcalorimetry.configs as configs
from pathlib import Path


LOCAL = Path(__file__).parents[0]
ETA_REFS = LOCAL / 'eta_references'


def test_read_4col_eta():
    # 4 column
    data1 = configs.EtaLike(ETA_REFS / 'C24N118_001.eff').load()
    assert data1.nom.sel(frequency=0.05, eta=0) == 0.9315
    assert data1.nom.sel(frequency=50, eta=0) == 0.8062
    # 7 column
    data2 = configs.EtaLike(ETA_REFS / 'C24N118_077.eff').load()
    assert data2.nom.sel(frequency=0.2, eta=0) == 0.9287
    pass


if __name__ == '__main__':
    test_read_4col_eta()
