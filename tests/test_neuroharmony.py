"""Tests for harmonization with Combat."""
from collections import namedtuple

from pandas.core.generic import NDFrame
from pandas import Series, concat
from pathlib import Path, PosixPath
import pytest
from scipy.special import comb

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
def resouces(tmpdir_factory):
    """Set up."""
    r = namedtuple('resources', 'data_path')
    r.data_path = 'data/raw/IXI'
    r.features = rois[:3]
    r.covars = ['Gender', 'scanner', 'Age']
    original_data = DataSet(Path(r.data_path)).data
    extra_vars = original_data.columns[~original_data.columns.isin(rois)]
    original_data.Age = original_data.Age.astype(int)
    scanners = original_data.scanner.unique()
    train_bool = original_data.scanner.isin(scanners[1:])
    test_bool = original_data.scanner.isin(scanners[:1])
    r.X_train_split = original_data[train_bool][r.features + r.covars]
    r.X_test_split = original_data[test_bool][r.features + r.covars]
    r.n_scanners = len(original_data.scanner.unique())
    return r


def test_neuroharmony_is_functional(resouces):
    """Test Neuroharmony."""
    x_train, x_test = resouces.X_train_split, resouces.X_test_split
    neuroharmony = Neuroharmony(resouces.features, resouces.covars)
    # neuroharmony.fit(x_train)
    # x_harmonized = neuroharmony.fit_transform(x_train)
    pass


# def test_sklearn_compatibility(resouces):
#     check_transformer_general('ComBat', ComBat(resouces.features, resouces.covars))


# def test_combat_is_functional(resouces):
#     """Test ComBat harmonization returns a NDFrame with no NaN values and conserves scanner column format."""
#     combat = ComBat(resouces.features, resouces.covars)
#     data_harmonized = combat.transform(resouces.data[resouces.features + resouces.covars])
#     assert isinstance(data_harmonized, NDFrame)
#     assert not data_harmonized.isna().any(axis=1).any()
#     assert isinstance(data_harmonized.scanner[0], str)
#
#
# def test_harmonization_works(resouces):
#     """Test harmonization with ComBat using the Kolmogorov-Smirnov test."""
#     combat = ComBat(resouces.features, resouces.covars)
#     data_harmonized = combat.transform(resouces.data[resouces.features + resouces.covars])
#     KS_original = ks_test_grid(resouces.data, resouces.features, 'scanner')
#     KS_harmonized = ks_test_grid(data_harmonized, resouces.features, 'scanner')
#     assert isinstance(KS_harmonized, dict)
#     assert isinstance(KS_harmonized[resouces.features[0]], NDFrame)
#     assert KS_harmonized[resouces.features[0]].shape == (resouces.n_scanners, resouces.n_scanners)
#     assert (compare_dfs(KS_original, KS_harmonized) == comb(resouces.n_scanners, 2)).all()
