"""Test the combination of the raw data files."""
from collections import namedtuple

from pandas.core.generic import NDFrame
from pathlib import Path, PosixPath
import pytest

from neuroharmony.data import combine_tools, collect_tools


@pytest.fixture(scope='session')
def resources(tmpdir_factory):
    """Set up."""
    r = namedtuple('resources', 'data_root')
    r.data_root = './data'
    r.raw_root = '%s/raw/' % r.data_root
    r.dataset_list = list(collect_tools.find_all_files_by_name('data/raw/', '*', depth=1))
    return r


def test_get_scanners(resources):
    """Test we can get the scanners from a single site."""
    site = combine_tools.Site(resources.dataset_list[0])
    assert isinstance(site.scanner_list, list)


def test_get_files(resources):
    """Test we can get the files for each scanner."""
    site = combine_tools.Site(Path('data/raw/COBRE'))
    assert isinstance(site.SCANNER01.freesurferData_path, PosixPath)
    site = combine_tools.Site(Path('data/raw/IXI'))
    assert isinstance(site.SCANNER01.freesurferData_path, PosixPath)
    assert isinstance(site.SCANNER01.participants_path, PosixPath)
    assert isinstance(site.SCANNER01.iqm_path, PosixPath)
    assert isinstance(site.SCANNER02.pred_path, PosixPath)
    assert isinstance(site.SCANNER03.qoala_path, PosixPath)


def test_load_files():
    """Test the files exist and can be open as NDFrames."""
    site = combine_tools.Site(Path('data/raw/COBRE'))
    assert isinstance(site.SCANNER01.freesurferData, NDFrame)
    assert isinstance(site.SCANNER01.participants, NDFrame)
    assert isinstance(site.SCANNER01.iqm, NDFrame)
    assert isinstance(site.SCANNER01.pred, NDFrame)
    assert isinstance(site.SCANNER01.qoala, NDFrame)


def test_combine_files():
    """Test the combination of the data files return a NDFrame."""
    site = combine_tools.Site(Path('data/raw/COBRE'))
    assert isinstance(site.data, NDFrame)


def test_combine_all_scanners():
    """Test the combination of the scanners in a dataset works."""
    site = combine_tools.Site(Path('data/raw/IXI'))
    assert isinstance(site.SCANNER01.data, NDFrame)
    assert isinstance(site.SCANNER02.data, NDFrame)
    assert isinstance(site.SCANNER03.data, NDFrame)
    assert isinstance(site.data, NDFrame)


def test_combine_all_sites():
    """Test we can combine all sites in a dataset."""
    DATASET = combine_tools.DataSet(Path('data/raw/'))
    assert isinstance(DATASET.IXI.data, NDFrame)
    assert isinstance(DATASET.COBRE.data, NDFrame)
    assert isinstance(DATASET.data, NDFrame)


def test_combine_freesurfer():
    """Test the combination of the freesurfer output return a dataframe."""
    mri_path = collect_tools.fetch_mri_data()
    FREESURFER = combine_tools.combine_freesurfer(f'{mri_path}/derivatives/freesurfer/')
    assert isinstance(FREESURFER, NDFrame)
