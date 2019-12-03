"""Tests for harmonization with Combat."""
from collections import namedtuple

from pandas.core.generic import NDFrame
from pandas import Series, concat
from pathlib import Path, PosixPath
import pytest
from scipy.special import comb
from sklearn.base import BaseEstimator
from sklearn.utils.estimator_checks import check_transformer_general

from src.data.combine_tools import DataSet
from src.models.harmonization import ComBat, Neuroharmony
from src.models.metrics import ks_test_grid
from src.data.rois import rois


def compare_dfs(ks_original, ks_harmonized):
    """Compare two dictionaries of dataframes to measure the success of harmonization."""
    vars = ks_original.keys()
    var_improved = Series(index=vars, dtype='bool')
    for var in vars:
        improved = ks_original[var] < ks_harmonized[var]
        var_improved.loc[var] = improved.sum().sum()
    return var_improved


@pytest.fixture(scope='session')
def resources(tmpdir_factory):
    """Set up."""
    r = namedtuple('resources', 'data_path')
    r.data_path = 'data/raw/IXI'
    r.features = rois[:3]
    r.regression_features = ['Age', 'summary_gm_median', 'spacing_x',
                             'summary_gm_p95', 'cnr', 'size_x',
                             'cjv', 'summary_wm_mean', 'icvs_gm', 'wm2max']
    r.covars = ['Gender', 'scanner', 'Age']
    original_data = DataSet(Path(r.data_path)).data
    original_data.Age = original_data.Age.astype(int)
    scanners = original_data.scanner.unique()
    train_bool = original_data.scanner.isin(scanners[1:])
    test_bool = original_data.scanner.isin(scanners[:1])
    r.X_train_split = original_data[train_bool]
    r.X_test_split = original_data[test_bool]
    r.n_scanners = len(original_data.scanner.unique())
    return r


def test_neuroharmony_is_functional(resources):
    """Test Neuroharmony."""
    x_train, x_test = resources.X_train_split, resources.X_test_split
    neuroharmony = Neuroharmony(resources.features,
                                resources.regression_features,
                                resources.covars,
                                n_jobs=27)
    x_harmonized = neuroharmony.fit_transform(x_train)
    assert isinstance(x_harmonized, NDFrame)
    assert isinstance(neuroharmony, BaseEstimator)
