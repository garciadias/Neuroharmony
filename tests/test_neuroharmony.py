"""Tests for harmonization with Neuroharmony."""
from collections import namedtuple

from pandas.core.generic import NDFrame
from pandas import concat
import pytest
from sklearn.base import BaseEstimator

from neuroharmony.data.collect_tools import fetch_sample
from neuroharmony.models.harmonization import Neuroharmony, _label_encode_covariates, _label_decode_covariates
from neuroharmony.models.harmonization import exclude_single_subject_groups
from neuroharmony.models.metrics import ks_test_grid
from neuroharmony.data.rois import rois


@pytest.fixture(scope="session")
def resources(tmpdir_factory):
    """Set up."""
    r = namedtuple("resources", "data_path")
    r.features = rois[:3]
    r.regression_features = [
        "Age",
        "summary_gm_median",
        "spacing_x",
        "summary_gm_p95",
        "cnr",
        "size_x",
        "cjv",
        "summary_wm_mean",
        "icvs_gm",
        "wm2max",
    ]
    r.covariates = ["Gender", "scanner", "Age"]
    r.eliminate_variance = ["scanner"]
    r.original_data = fetch_sample()
    exclude_vars = r.original_data.columns[r.original_data.isna().sum() != 0].to_list()
    r.original_data = r.original_data[[var for var in r.original_data.columns if var not in exclude_vars]]
    r.original_data = r.original_data[~r.original_data[r.covariates].isna().any(axis=1)]
    r.original_data.Age = r.original_data.Age.astype(int)
    scanners = r.original_data.scanner.unique()
    train_bool = r.original_data.scanner.isin(scanners[1:])
    test_bool = r.original_data.scanner.isin(scanners[:1])
    r.X_train_split = r.original_data[train_bool]
    r.X_train_split = exclude_single_subject_groups(r.X_train_split, r.covariates)
    r.X_test_split = r.original_data[test_bool]
    r.n_scanners = len(r.original_data.scanner.unique())
    return r


@pytest.fixture(scope="session")
def model(resources):
    """Define mock model.

    Parameters
    ----------
    resources : namedtuple
        Basic information needed to build a neuroharmony model.

    Returns
    -------
    Neuroharmony class
        Neuroharmony model.
    """
    neuroharmony = Neuroharmony(
        resources.features,
        resources.regression_features,
        resources.covariates,
        resources.eliminate_variance,
        param_distributions=dict(
            RandomForestRegressor__n_estimators=[5, 10, 15, 20],
            RandomForestRegressor__random_state=[42, 78],
            RandomForestRegressor__warm_start=[False, True],
        ),
        estimator_args=dict(n_jobs=1, random_state=42),
        randomized_search_args=dict(cv=5, n_jobs=27),
    )
    return neuroharmony


def test_label_encode_decode(resources):
    """Test encoder and decoder."""
    df, encoders = _label_encode_covariates(resources.X_train_split, resources.covariates)
    assert all([isinstance(value, int) for value in df.scanner])
    df = _label_decode_covariates(df, resources.covariates, encoders)
    assert all([isinstance(value, str) for value in df.scanner])


def test_neuroharmony_behaviour(resources):
    """Test Neuroharmony."""
    x_train, x_test = resources.X_train_split, resources.X_test_split
    neuroharmony = Neuroharmony(
        resources.features,
        resources.regression_features,
        resources.covariates,
        resources.eliminate_variance,
        param_distributions=dict(
            RandomForestRegressor__n_estimators=[5, 10, 15, 20],
            RandomForestRegressor__random_state=[42, 78],
            RandomForestRegressor__warm_start=[False, True],
        ),
        estimator_args=dict(n_jobs=1, random_state=42),
        randomized_search_args=dict(cv=5, n_jobs=27),
    )
    x_train_harmonized = neuroharmony.fit_transform(x_train)
    x_test_harmonized = neuroharmony.predict(x_test)
    data_harmonized = concat([x_train_harmonized, x_test_harmonized], sort=False)
    KS_original = ks_test_grid(resources.original_data, resources.features, "scanner")
    KS_harmonized = ks_test_grid(data_harmonized, resources.features, "scanner")
    assert KS_original[resources.features[0]].shape == (resources.n_scanners, resources.n_scanners)
    assert KS_harmonized[resources.features[0]].shape == (resources.n_scanners, resources.n_scanners)
    assert isinstance(x_test, NDFrame)
    assert isinstance(neuroharmony, BaseEstimator)
    assert not neuroharmony.prediction_is_covered_.all(), "No subjects out of the range."


def test_ckeck_training_range(model, resources):
    """Test check model can record the training range of each variables."""
    neuroharmony = model
    neuroharmony._check_training_ranges(resources.X_train_split)
    assert isinstance(neuroharmony.coverage_, NDFrame), "coverage_ is not DataFrame."
    assert not neuroharmony.coverage_.isna().any().any(), "NaN field detected."


def test_ckeck_prediction_range(model, resources):
    """Test we can verify if the prediction sample is covered by the training ranges."""
    neuroharmony = model
    neuroharmony._check_training_ranges(resources.X_train_split)
    neuroharmony._check_prediction_ranges(resources.X_test_split)
    assert isinstance(neuroharmony.prediction_is_covered_, NDFrame), "prediction_is_covered_ is not DataFrame."
    assert not neuroharmony.prediction_is_covered_.isna().any(), "NaN field detected."
    assert not neuroharmony.prediction_is_covered_.all(), "No subjects out of the range."
    assert isinstance(neuroharmony.subjects_out_of_range_, list), "The subjects_out_of_range_ is not a list."
