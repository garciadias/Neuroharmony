"""Tests for harmonization with Combat."""
from collections import namedtuple

from pandas.core.generic import NDFrame
from pathlib import Path, PosixPath
import pytest

from src.data.combine_tools import DataSet
from src.data.harmonization import ComBat
from src.data.rois import rois


@pytest.fixture(scope='session')
def resouces(tmpdir_factory):
    """Set up."""
    r = namedtuple('resources', 'data_path')
    r.data_path = 'data/raw/IXI'
    r.data = DataSet(Path(r.data_path)).data
    r.features = ['Left-Lateral-Ventricle', 'Left-Inf-Lat-Vent', ]
    r.covars = ['Gender', 'scanner', 'Age']
    return r


def test_combat_is_functional(resouces):
    """Test ComBat harmonization returns a NDFrame with no NaN values."""
    combat = ComBat(resouces.features, resouces.covars)
    data_harmonized = combat.transform(resouces.data[resouces.features + resouces.covars])
    print(data_harmonized.head())
    assert isinstance(data_harmonized, NDFrame)
    assert not data_harmonized.isna().any(axis=1).any()
