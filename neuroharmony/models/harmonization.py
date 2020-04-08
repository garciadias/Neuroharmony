"""Tools for harmonization."""

import os
import sys

from neuroCombat import neuroCombat
from numpy import unique
from pandas import Series, DataFrame, concat
from pandas.core.generic import NDFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler
from tqdm import tqdm


def supress_print(func):
    """Define decorator for print supression."""
    def wrapper(*args, **kwargs):
        sys.stdout = open(os.devnull, 'w')
        resuts = func(*args, **kwargs)
        sys.stdout = sys.__stdout__
        return resuts
    return wrapper


def label_encode_covars(df, covars):
    """Encode dataframe covars."""
    encoders = {}
    for covar in covars:
        encoders[covar] = LabelEncoder()
        df[covar] = encoders[covar].fit_transform(df[covar]).astype(int)
    return encoders


def label_decode_covars(df, covars, encoders):
    """Decode dataframe covars."""
    for covar in covars:
        df[covar] = encoders[covar].inverse_transform(df[covar])


@supress_print
def combat(*args, **kwargs):
    """Redefine ComBat to supress printing unhrlpful data."""
    return neuroCombat(*args, **kwargs)


class ComBat(BaseEstimator, TransformerMixin):
    """ComBat class."""

    def __init__(self, features, covars, eliminate_variance):
        """Init class with the original data."""
        self.features = features
        self.covars = covars
        self.eliminate_variance = eliminate_variance

    def _check_data(self, df):
        type_error = "Input data should be a pandas dataframe (NDFrame)."
        assert isinstance(df, NDFrame), TypeError(type_error)
        self._check_vars(df, self.features)
        self._check_vars(df, self.covars)
        self._check_vars(df, self.eliminate_variance)

    def _check_vars(self, df, vars):
        vars = Series(vars)
        is_feature_present = vars.isin(df.columns)
        missing_features_str = "Missing features: %s" % ', '.join(vars[~is_feature_present])
        assert is_feature_present.all(), ValueError(missing_features_str)

    def _reconstruct_original_fieds(self, df, harmonized, extra_vars):
        """Concatenate ComBat data with the original data fields."""
        harmonized = DataFrame(harmonized, index=df.index, columns=self.features)
        return concat([harmonized, df.loc[harmonized.index][extra_vars]], axis=1, sort=True)

    def _exclude_single_subject_groups(self, df):
        """Exclude subjectis that are the only representants of a value in the variable field."""
        for covar in self.covars:
            instances, n = unique(df[covar], return_counts=True)
            category_counts = DataFrame(n, columns=['N'], index=instances)
            single_subj = category_counts[category_counts.N == 1].index.tolist()
            df = concat([df.groupby(covar).get_group(group)
                         for group in df[covar].unique()
                         if group not in single_subj])
        return df

    def _run_combat(self, df):
        """Run ComBat for all covars."""
        extra_vars = df.columns[~df.columns.isin(self.features)]
        harmonized = df.copy()
        for batch_col in self.eliminate_variance:
            harmonized = combat(data=harmonized[self.features],
                                covars=harmonized[self.covars],
                                batch_col=batch_col, )
            harmonized = self._reconstruct_original_fieds(df, harmonized, extra_vars)
        return harmonized

    def _exclude_subjects_with_nans(self, df):
        return df[~df.isna().any(axis=1)]

    def transform(self, df, y=None):
        """Run ComBat normalization.

        Transforms.

        Parameters
        ----------
        df: NDFrame, of shape [n_subjects, n_features]
         Dataframe with data for each subject. The dataframe has to contain the features to be harmonized pandas
         the covars which you want to use to harmonization.

        Returns
        -------
        harmonized: NDFrame, shape(n_subjects, n_features)
         Dataframe with the harmonized data.

        Raises
        ------
        ValueError:
         If there are missing features among the covars or the features.

        Examples
        --------
        >>> ixi = DataSet('data/raw/IXI').data
        >>> features = ['Left-Lateral-Ventricle', 'Left-Inf-Lat-Vent', ]
        >>> covars = ['Gender', 'Dataset', 'Age']
        >>> combat = ComBat(features, covars)
        >>> harmonized = combat(ixi)
        >>> harmonized

                      Left-Lateral-Ventricle  Left-Inf-Lat-Vent  Gender  Dataset  Age
        sub-002-00-2  0.007666                0.000134           0       2        31
        sub-012-00-1  0.003778                0.000064           1       1        14
        sub-013-00-1  0.012818                0.000464           1       1        17
        sub-014-00-1  0.006289                0.000167           0       1        19
        sub-015-00-1  0.003310                0.000154           1       1        26

        """
        self._check_data(df)
        self.encoders = label_encode_covars(df, self.covars)
        df = self._exclude_single_subject_groups(df)
        df = self._exclude_subjects_with_nans(df)
        self.harmonized_ = self._run_combat(df)
        label_decode_covars(self.harmonized_, self.covars, self.encoders)
        return self.harmonized_

    def fit(self, df):
        """Fit the model.

        Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.

        Parameters
        ----------
        df: NDFrame, of shape [n_subjects, n_features]
         Training data. Must fulfill input requirements of first step of the pipeline.

        Returns
        -------
        self : ComBat
         This estimator

        """
        self.harmonized_ = self.transform(df)
        return self

    def fit_transform(self, df):
        """Fit to data, then transform it.

        Fits transformer to df and y with optional parameters fit_params
        and returns a transformed version of df.

        Parameters
        ----------
        df: NDFrame, of shape [n_subjects, n_features]
         Training set.

        Returns
        -------
        harmonized_: NDFrame, of shape [n_samples, n_features_new]
         Transformed array.

        """
        self.harmonized_ = self.transform(df)
        return self.harmonized_


class Neuroharmony(BaseEstimator, TransformerMixin):
    """Create a Neuroharmony model.

    Parameters
    ----------
    features: list
     Target features to be harmonized, for example, ROIs.
    regression_features: list
     Features used to derive harmonization rules, for example, IQMs.
    covars: list
     Variables for which we whant to eliminate the bias, for example, age, sex, and scanner.
    estimator: sklearn estimator, default=RandomForestRegressor()
     Model to make the harmonization regression.
    scaler: sklearn scaler, default=RobustScaler()
     Scaler used as first step of the harmonization regression.
    param_distributions: dict, default=dict(RandomForestRegressor__n_estimators=[100, 200, 500],
                                            RandomForestRegressor__warm_start=[False, True], )
     Distribution of parameters to be testes on the RandomizedSearchCV.
    **estimator_args: dict
     Parameters for the estimator.
    **scaler_args: dict
     Parameters for the scaler.
    **randomized_search_args: dict
     Parameters for the RandomizedSearchCV.
     See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
    **pipeline_args: dict
     Parameters for the sklearn Pipeline.
     See https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

    Attributes
    ----------
    X_harmonized_: NDFrame [n_subjects, n_features]
     Input data harmonized.
    leaveonegroupout_:
     Leave One Group Out cross-validator.
    models_by_feature_:
     Estimators by features.

    """

    def __init__(self,
                 features,
                 regression_features,
                 covars,
                 eliminate_variance,
                 estimator=RandomForestRegressor(),
                 scaler=RobustScaler(),
                 param_distributions=dict(RandomForestRegressor__n_estimators=[100, 200, 500],
                                          RandomForestRegressor__criterion=['mse', 'mae'],
                                          RandomForestRegressor__warm_start=[False, True], ),
                 estimator_args=dict(n_jobs=1, random_state=42, criterion='mae', verbose=False),
                 scaler_args=dict(),
                 randomized_search_args=dict(),
                 pipeline_args=dict(),
                 ):
        """Init."""
        self.features = features
        self.regression_features = regression_features
        self.covars = covars
        self.eliminate_variance = eliminate_variance
        self.estimator = estimator
        self.scaler = scaler
        self.param_distributions = param_distributions
        self.randomized_search_args = randomized_search_args
        self.estimator.set_params(**estimator_args)
        self.scaler.set_params(**scaler_args)
        self.pipeline_args = pipeline_args

    def _check_vars(self, df, vars):
        vars = Series(vars)
        is_feature_present = vars.isin(df.columns)
        missing_features_str = "Missing features: %s" % ', '.join(vars[~is_feature_present])
        assert is_feature_present.all(), ValueError(missing_features_str)

    def _check_data(self, df):
        type_error = "Input data should be a pandas dataframe (NDFrame)."
        assert isinstance(df, NDFrame), TypeError(type_error)
        self._check_vars(df, self.features)
        self._check_vars(df, self.covars)
        self._check_vars(df, self.eliminate_variance)

    def _random_search_with_leave_one_group_out_cv(self, X, y, groups):
        self.leaveonegroupout_ = LeaveOneGroupOut()
        self.cv = list(self.leaveonegroupout_.split(X, y, groups))
        self.pipeline = Pipeline(
            steps=[(self.scaler.__class__.__name__, self.scaler),
                   (self.estimator.__class__.__name__, self.estimator),
                   ])
        self.pipeline.set_params(**self.pipeline_args)
        self.randomized_search_cv = RandomizedSearchCV(self.pipeline,
                                                       param_distributions=self.param_distributions,
                                                       cv=self.cv
                                                       )
        self.randomized_search_cv.set_params(**self.randomized_search_args)
        self.randomized_search_cv.fit(X, y)
        return self.randomized_search_cv

    def _reconstruct_original_fieds(self, df, harmonized, extra_vars):
        """Concatenate ComBat data with the original data fields."""
        harmonized = DataFrame(harmonized, index=df.index, columns=self.features)
        return concat([harmonized, df.loc[harmonized.index][extra_vars]], axis=1, sort=True)

    def _train_neurofind(self, estimator=RandomForestRegressor()):
        return estimator.__class__.__name__

    def _run_combat(self, df):
        self.extra_vars = df.columns[~df.columns.isin(self.features)]
        combat = ComBat(self.features, self.covars, self.eliminate_variance)
        self.X_harmonized_ = combat.transform(df)
        label_decode_covars(self.X_harmonized_, self.covars, self.encoders)
        delta = df[self.features] - self.X_harmonized_[self.features]
        y_train_split = concat([delta, df[self.extra_vars]], axis=1, sort=False).dropna()
        X_train_split = df.loc[y_train_split.index]
        return X_train_split, y_train_split

    def fit(self, df):
        """Fit the model.

        Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.

        Parameters
        ----------
        df: NDFrame, of shape [n_subjects, n_features]
         Training data. Must fulfill input requirements of first step of the pipeline.

        Returns
        -------
        self : Neuroharmony
         This estimator

        """
        self._check_data(df)
        self.encoders = label_encode_covars(df, self.covars)
        X_train_split, y_train_split = self._run_combat(df)
        self.models_by_feature_ = {}
        desc = 'Randomized search of %s, ROIs regression' % self.estimator.__class__.__name__
        for var in tqdm(self.features, desc=desc):
            self.models_by_feature_[var] = self._random_search_with_leave_one_group_out_cv(
                X_train_split[self.regression_features + [var]],
                y_train_split[var],
                y_train_split['scanner'], )
        return self

    def fit_transform(self, df):
        """Fit to data, then transform it.

        Fits transformer to df and y with optional parameters fit_params
        and returns a transformed version of df.

        Parameters
        ----------
        df: NDFrame, of shape [n_subjects, n_features]
         Training set.

        Returns
        -------
        harmonized_: NDFrame, of shape [n_samples, n_features_new]
         Data harmonized with ComBat.

        """
        self._check_data(df)
        self.fit(df)
        return self.X_harmonized_

    def predict(self, df):
        """Predict regression target for df.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.
        Parameters
        ----------
        df : NDFrame [n_samples, n_features]
            Pandas dataframe with features, regression_features and covars.
        Returns
        -------
        y : NDFrame [n_samples, n_features]
            Data harmonized with Neuroharmony.

        """
        # Check data
        self._check_data(df)
        self.models_by_feature_[self.features[0]]._check_is_fitted('predict')
        self.predicted_ = DataFrame([], columns=self.features, index=df.index)
        for var in self.features:
            predicted_y_1 = self.models_by_feature_[var].predict(
                df[self.regression_features + [var]])
            self.predicted_[var] = df[var] - predicted_y_1
        self.predicted_ = self._reconstruct_original_fieds(df, self.predicted_, self.extra_vars)
        return self.predicted_

        def transform(self, df):
            return self.predict(df)
