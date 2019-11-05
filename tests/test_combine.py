
from collections import namedtuple

from pandas.core.generic import NDFrame
from pathlib import Path, PosixPath
import pytest

from src.data import combine_tools, collect_tools


@pytest.fixture(scope='session')
def resouces(tmpdir_factory):
    r = namedtuple('resources', 'data_root')
    r.data_root = './data'
    r.raw_root = '%s/raw/' % r.data_root
    r.dataset_list = list(collect_tools.find_all_files_by_name('data/raw/', '*', depth=1))
    return r


def test_get_scanners(resouces):
    site = combine_tools.Site(resouces.dataset_list[0])
    assert isinstance(site.scanner_list, list)


def test_get_files(resouces):
    site = combine_tools.Site(Path('data/raw/COBRE'))
    assert isinstance(site.freesurferData_path, PosixPath)
    site = combine_tools.Site(Path('data/raw/IXI'))
    assert isinstance(site.SCANNER01.freesurferData_path, PosixPath)
    assert isinstance(site.SCANNER01.participants_path, PosixPath)
    assert isinstance(site.SCANNER01.iqm_path, PosixPath)
    assert isinstance(site.SCANNER02.pred_path, PosixPath)
    assert isinstance(site.SCANNER03.qoala_path, PosixPath)


def test_load_files():
    site = combine_tools.Site(Path('data/raw/COBRE'))
    assert isinstance(site.freesurferData, NDFrame)
    assert isinstance(site.participants, NDFrame)
    assert isinstance(site.iqm, NDFrame)
    assert isinstance(site.pred, NDFrame)
    assert isinstance(site.qoala, NDFrame)


def test_combine_files():
    site = combine_tools.Site(Path('data/raw/COBRE'))
    assert isinstance(site.data, NDFrame)


def test_combine_all_scanners():
    site = combine_tools.Site(Path('data/raw/IXI'))
    assert isinstance(site.SCANNER01.data, NDFrame)
    assert isinstance(site.SCANNER02.data, NDFrame)
    assert isinstance(site.SCANNER03.data, NDFrame)
    assert isinstance(site.data, NDFrame)


def test_combine_all_sites():
    DATASET = combine_tools.DataSet(Path('data/raw/'))
    assert isinstance(DATASET.IXI.data, NDFrame)
    assert isinstance(DATASET.COBRE.data, NDFrame)
    assert isinstance(DATASET.data, NDFrame)
