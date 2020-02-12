"""Tests for harmonization with Neuroharmony."""
from collections import namedtuple

from pandas.core.generic import NDFrame
from pandas import Series, concat
from pathlib import Path, PosixPath
import pytest
from scipy.special import comb
from sklearn.base import BaseEstimator
from sklearn.utils.estimator_checks import check_transformer_general
from numpy import int64, int32

from neuroharmony.data.combine_tools import DataSet
from neuroharmony.models.harmonization import ComBat, Neuroharmony, label_encode_covars, label_decode_covars
from neuroharmony.models.metrics import ks_test_grid
from neuroharmony.data.rois import rois


@pytest.fixture(scope='session')
def resources(tmpdir_factory):
    """Set up."""
    r = namedtuple('resources', 'data_path')
    r.data_path = 'data/raw/IXI'
    r.features = rois[:3]
    r.regression_features = ['Age', 'summary_gm_median', 'spacing_x', 'summary_gm_p95',
                             'cnr', 'size_x', 'cjv', 'summary_wm_mean', 'icvs_gm', 'wm2max']
    r.covars = ['Gender', 'scanner', 'Age']
    r.eliminate_variance = ['scanner']
    r.original_data = DataSet(Path(r.data_path)).data
    r.original_data.Age = r.original_data.Age.astype(int)
    scanners = r.original_data.scanner.unique()
    train_bool = r.original_data.scanner.isin(scanners[1:])
    test_bool = r.original_data.scanner.isin(scanners[:1])
    r.X_train_split = r.original_data[train_bool]
    r.X_test_split = r.original_data[test_bool]
    r.n_scanners = len(r.original_data.scanner.unique())
    return r


def test_label_encode_decode(resources):
    """Test encoder and decoder."""
    encoders = label_encode_covars(resources.X_train_split, resources.covars)
    assert all([isinstance(value, int) for value in resources.X_train_split.scanner])
    label_decode_covars(resources.X_train_split, resources.covars, encoders)
    assert all([isinstance(value, str) for value in resources.X_train_split.scanner])


def test_neuroharmony_behaviour(resources):
    """Test Neuroharmony."""
    x_train, x_test = resources.X_train_split, resources.X_test_split
    neuroharmony = Neuroharmony(resources.features,
                                resources.regression_features,
                                resources.covars,
                                resources.eliminate_variance,
                                param_distributions=dict(
                                    RandomForestRegressor__n_estimators=[5, 10, 15, 20],
                                    RandomForestRegressor__random_state=[42, 78],
                                    RandomForestRegressor__warm_start=[False, True],
                                ),
                                estimator_args=dict(n_jobs=1, random_state=42),
                                randomized_search_args=dict(cv=5, n_jobs=27))
    x_train_harmonized = neuroharmony.fit_transform(x_train)
    x_test_harmonized = neuroharmony.predict(x_test)
    data_harmonized = concat([x_train_harmonized, x_test_harmonized], sort=False)
    KS_original = ks_test_grid(resources.original_data, resources.features, 'scanner')
    KS_harmonized = ks_test_grid(data_harmonized, resources.features, 'scanner')
    assert KS_original[resources.features[0]].shape == (resources.n_scanners,
                                                        resources.n_scanners)
    assert KS_harmonized[resources.features[0]].shape == (resources.n_scanners,
                                                          resources.n_scanners)
    assert isinstance(x_test, NDFrame)
    assert isinstance(neuroharmony, BaseEstimator)
