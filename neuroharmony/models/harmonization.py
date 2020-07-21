from os import devnull
from warnings import warn
import sys

from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from pandas.core.generic import NDFrame
from pandas import Series, DataFrame, concat, merge
from numpy import unique
from neuroCombat import neuroCombat


def exclude_single_subject_groups(df, covariates):
    """Exclude categories with only one subject.

    Parameters
    ==========
    df : NDFrame of shape [n_subjects, n_features]
        Pandas DataFrame in the Neuroharmony input format containing the variables in the `covariates` list.
    covariates : list
        List of covariates for which the Harmonization should eliminate or conserve.

    Returns
    =======
    df : NDFrame of shape [n_subjects, n_features]
        Padas DataFrame excluding the subjects that would result in a single subject split of the covariates grouping.
    """
    for covar in covariates:
        instances, n = unique(df[covar], return_counts=True)
        category_counts = DataFrame(n, columns=["N"], index=instances)
        single_subj = category_counts[category_counts.N == 1].index.tolist()
        df = concat([df.groupby(covar).get_group(group) for group in df[covar].unique() if group not in single_subj])
    return df


def _supress_print(func):
    """Define decorator for print suppression."""

    def wrapper(*args, **kwargs):
        sys.stdout = open(devnull, "w")
        resuts = func(*args, **kwargs)
        sys.stdout = sys.__stdout__
        return resuts

    return wrapper


def _label_encode_covariates(df, covariates):
    """Encode dataframe covariates."""
    encoders = {}
    for covar in covariates:
        encoders[covar] = LabelEncoder()
        df[covar] = encoders[covar].fit_transform(df[covar])
    return df, encoders


def _label_decode_covariates(df, covariates, encoders):
    """Decode dataframe covariates."""
    for covar in covariates:
        df[covar] = encoders[covar].inverse_transform(df[covar])
    return df


@_supress_print
def combat(*args, **kwargs):
    """Redefine ComBat to suppress printing unhelpful data."""
    return neuroCombat(*args, **kwargs)


class ComBat(BaseEstimator, TransformerMixin):
    """ComBat model for harmonization.

    A wrapper to the `NeuroCombat <https://github.com/ncullen93/neuroCombat>`_ harmonization as described by
    `Fortin et al., (2018) <https://doi.org/10.1016/j.neuroimage.2017.11.024>`_.

    Parameters
    ==========
    features : list
        List of features to be harmonized with ComBat.
    covariates :
        List of covariates for which the variance needs to be conserved.
    eliminate_variance :
        List of variables for which the variance needs to be eliminated.

    Attributes
    ==========
    harmonized_ : NDFrame of shape [n_subjects, n_features]
        ComBat harmonized DataFrame.
    """

    def __init__(self, features, covariates, eliminate_variance):
        self.covariates = covariates
        self.eliminate_variance = eliminate_variance
        self.features = features
        self.reindexed = False

    def transform(self, df, y=None):
        """Run ComBat normalization.

        Applies trained model.

        Parameters
        ----------
        df: NDFrame of shape [n_subjects, n_features]
            Dataframe with data for each subject. The dataframe has to contain the features to be harmonized pandas
            the covariates which you want to use to harmonization.

        Returns
        -------
        harmonized: NDFrame of shape [n_subjects, n_features]
            Dataframe with the harmonized data.

        Raises
        ------
        ValueError:
            If there are missing features among the covariates or the features.

        Examples
        --------
        >>> ixi = DataSet('data/raw/IXI').data
        >>> features = ['Left-Lateral-Ventricle', 'Left-Inf-Lat-Vent', ]
        >>> covariates = ['Gender', 'Dataset', 'Age']
        >>> combat = ComBat(features, covariates)
        >>> harmonized = combat(ixi)
        >>> harmonized

        +------------+----------------------+-----------------+------+-------+---+
        |subject_id  |Left-Lateral-Ventricle|Left-Inf-Lat-Vent|Gender|Dataset|Age|
        +============+======================+=================+======+=======+===+
        |sub-002-00-2|  0.007666            |    0.000134     |  0   |   2   |31 |
        +------------+----------------------+-----------------+------+-------+---+
        |sub-012-00-1|  0.003778            |    0.000064     |  1   |   1   |14 |
        +------------+----------------------+-----------------+------+-------+---+
        |sub-013-00-1|  0.012818            |    0.000464     |  1   |   1   |17 |
        +------------+----------------------+-----------------+------+-------+---+
        |sub-014-00-1|  0.006289            |    0.000167     |  0   |   1   |19 |
        +------------+----------------------+-----------------+------+-------+---+
        |sub-015-00-1|  0.003310            |    0.000154     |  1   |   1   |26 |
        +------------+----------------------+-----------------+------+-------+---+
        """
        self._check_data(df.copy())
        self._check_single_subject_groups(df.copy())
        self._check_subjects_with_nans(df.copy())
        df = self._check_index(df.copy())
        df, self.encoders = _label_encode_covariates(df.copy(), unique(self.covariates + self.eliminate_variance))
        self.harmonized_ = self._run_combat(df.copy())
        self.harmonized_ = _label_decode_covariates(
            self.harmonized_, unique(self.covariates + self.eliminate_variance), self.encoders,
        )
        if self.reindexed:
            self.harmonized_ = self._clean_index(self.harmonized_)
        return self.harmonized_

    def fit(self, df):
        """Fit the model.

        Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.

        Parameters
        ----------
        df: NDFrame of shape [n_subjects, n_features]
            Training data. Must fulfil input requirements of first step of the pipeline.

        Returns
        -------
        self : ComBat
            This estimator
        """
        self.harmonized_ = self.transform(df.copy())
        return self

    def fit_transform(self, df):
        """Fit to data, then transform it.

        Fits transformer to df and y with optional parameters fit_params and returns a transformed version of df.

        Parameters
        ----------
        df: NDFrame of shape [n_subjects, n_features]
            Training set.

        Returns
        -------
        harmonized_: NDFrame of shape [n_samples, n_features_new]
            Transformed array.
        """
        self.harmonized_ = self.transform(df.copy())
        return self.harmonized_

    def _check_data(self, df):
        type_error = "Input data should be a pandas dataframe (NDFrame)."
        assert isinstance(df, NDFrame), TypeError(type_error)
        self._check_vars(df, self.features)
        self._check_vars(df, self.covariates)
        self._check_vars(df, self.eliminate_variance)

    def _check_index(self, df):
        if df.index.duplicated().sum():
            df.index = [f'{index}_TMP_{i}' for i, index in enumerate(df.index)]
            self.reindexed = True
        return df

    def _clean_index(self, df):
        df.index = [index.split('_TMP_')[0] for index in df.index]
        return df

    def _check_vars(self, df, vars):
        vars = Series(vars)
        is_feature_present = vars.isin(df.columns)
        missing_features_str = "Missing features: %s" % ", ".join(vars[~is_feature_present])
        assert is_feature_present.all(), ValueError(missing_features_str)

    def _reconstruct_original_fieds(self, df, harmonized, extra_vars):
        """Concatenate ComBat data with the original data fields."""
        if df.index.name is not None:
            index_name = df.index.name
        else:
            index_name = "subject_index"
        if df.index.name in df.columns:
            index_name = df.index.name + "_y"
        harmonized = DataFrame(harmonized, index=df.index, columns=self.features)
        harmonized.index.rename(index_name, inplace=True)
        df = df.loc[harmonized.index][extra_vars].reset_index().copy()
        harmonized = harmonized.reset_index()
        return merge(harmonized, df, how="inner", on=index_name).set_index(index_name)

    def _check_single_subject_groups(self, df):
        """Exclude subjects with only a value in the variable field."""
        for covar in self.covariates:
            instances, n = unique(df[covar], return_counts=True)
            category_counts = DataFrame(n, columns=["N"], index=instances)
            single_subj = category_counts[category_counts.N == 1].index.astype("str").tolist()
            if len(single_subj) > 0:
                raise ValueError(
                    "ComBat harmonization requires more than one subject in each split group."
                    f"The following covar imply groups of a single subject: {covar}"
                )

    def _check_subjects_with_nans(self, df):
        if df.isna().any(axis=1).sum() > 0:
            raise ValueError("NaN values found on subjects data.")

    def _run_combat(self, df):
        """Run ComBat for all covariates."""
        extra_vars = df.columns[~df.columns.isin(self.features)]
        harmonized = df.copy()
        for batch_col in self.eliminate_variance:
            harmonized = combat(
                data=harmonized[self.features].copy(), covars=harmonized[self.covariates].copy(), batch_col=batch_col,
            )
            harmonized = self._reconstruct_original_fieds(df, harmonized, extra_vars)
        return harmonized


class Neuroharmony(TransformerMixin, BaseEstimator):
    """ Harmonization tool to mitigate scanner bias.

    Parameters
    ----------
    features : list
        Target features to be harmonized, for example, ROIs.
    regression_features : list
        Features used to derive harmonization rules, for example, IQMs.
    covariates : list
        Variables for which we want to eliminate the bias, for example, age, sex, and scanner.
    estimator : sklearn estimator, default=RandomForestRegressor()
        Model to make the harmonization regression.
    scaler : sklearn scaler, default=StandardScaler()
        Scaler used as the first step of the harmonization regression.
    param_distributions : dict, default=dict(RandomForestRegressor__n_estimators=[100, 200, 500],
                                             RandomForestRegressor__warm_start=[False, True], )
        Distribution of parameters to be testes on the RandomizedSearchCV.
    **estimator_args : dict
        Parameters for the estimator.
    **scaler_args : dict
        Parameters for the scaler.
    **randomized_search_args : dict
        Parameters for the RandomizedSearchCV.
        See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
    **pipeline_args : dict
        Parameters for the sklearn Pipeline.
        See https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

    Attributes
    ----------
    X_harmonized_ : NDFrame [n_subjects, n_features]
        Input data harmonized.
    leaveonegroupout_ :
        Leave One Group Out cross-validator.
    models_by_feature_ :
        Estimators by features.
    """

    def __init__(
        self,
        features,
        regression_features,
        covariates,
        eliminate_variance,
        estimator=RandomForestRegressor(),
        scaler=StandardScaler(),
        decomposition=PCA(),
        param_distributions=dict(
            RandomForestRegressor__n_estimators=[100, 200, 500],
            RandomForestRegressor__criterion=["mse", "mae"],
            RandomForestRegressor__warm_start=[False, True],
        ),
        estimator_args=dict(n_jobs=1, random_state=42, criterion="mae", verbose=False),
        scaler_args=dict(),
        randomized_search_args=dict(),
        pipeline_args=dict(),
    ):
        self.covariates = covariates
        self.decomposition = decomposition
        self.eliminate_variance = eliminate_variance
        self.estimator = estimator
        self.estimator.set_params(**estimator_args)
        self.features = features
        self.param_distributions = param_distributions
        self.pipeline_args = pipeline_args
        self.randomized_search_args = randomized_search_args
        self.regression_features = regression_features
        self.reindexed = False
        self.scaler = scaler
        self.scaler.set_params(**scaler_args)

    def fit(self, df):
        """Fit the model.

        Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.

        Parameters
        ----------
        df : NDFrame of shape [n_subjects, n_features]
             Training data. Must fulfil input requirements of the first step of the pipeline.

        Returns
        -------
        self : Neuroharmony
               This estimator
        """
        self._check_data(df.copy())
        self._check_training_ranges(df.copy())
        df = self._check_index(df.copy())
        df, self.encoders = _label_encode_covariates(df.copy(), unique(self.covariates + self.eliminate_variance))
        X_train_split, y_train_split = self._run_combat(df.copy())
        self.models_by_feature_ = {}
        desc = "Randomized search of Neuroharmony hyperparameters: "
        for var in tqdm(self.features, desc=desc):
            self.models_by_feature_[var] = self._random_search_with_leave_one_group_out_cv(
                X_train_split[self.regression_features + [var]], y_train_split[var], y_train_split["scanner"],
            )
        return self

    def fit_transform(self, df):
        """Fit to data, then transform it.

        Fits transformer to df and y with optional parameters fit_params
        and returns a transformed version of df.

        Parameters
        ----------
        df: NDFrame of shape [n_subjects, n_features]
            Training set.

        Returns
        -------
        harmonized_: NDFrame of shape [n_samples, n_features_new]
            Data harmonized with ComBat.
        """
        self.fit(df.copy())
        return self.X_harmonized_

    def predict(self, df):
        """Predict regression target for df.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.

        Parameters
        ----------
        df : NDFrame of shape [n_samples, n_features]
            Pandas dataframe with features, regression_features and covariates.

        Returns
        -------
        y : NDFrame of shape [n_samples, n_features]
            Data harmonized with Neuroharmony.
        """
        # Check data
        self._check_data(df.copy())
        self._check_prediction_ranges(df.copy())
        df = self._check_index(df.copy())
        df, self.encoders = _label_encode_covariates(df.copy(), unique(self.covariates + self.eliminate_variance))
        self.models_by_feature_[self.features[0]]._check_is_fitted("predict")
        self.predicted_ = DataFrame([], columns=self.features, index=df.index)
        for var in self.features:
            predicted_y_1 = self.models_by_feature_[var].predict(df[self.regression_features + [var]])
            self.predicted_[var] = df[var] - predicted_y_1
        self.predicted_ = self._reconstruct_original_fieds(df, self.predicted_, self.extra_vars)
        self.predicted_ = _label_decode_covariates(
            self.predicted_, unique(self.covariates + self.eliminate_variance), self.encoders,
        )
        if self.reindexed:
            self.predicted_ = self._clean_index(self.predicted_)
        return self.predicted_

    def transform(self, df):
        """Predict regression target for df.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.

        Parameters
        ----------
        df : NDFrame of shape [n_samples, n_features]
            Pandas dataframe with features, regression_features and covariates.

        Returns
        -------
        y : NDFrame of shape [n_samples, n_features]
            Data harmonized with Neuroharmony.
        """
        return self.predict(df.copy())

    def _check_vars(self, df, vars):
        vars = Series(vars)
        is_feature_present = vars.isin(df.columns)
        missing_features_str = "Missing features: %s" % ", ".join(vars[~is_feature_present])
        assert is_feature_present.all(), ValueError(missing_features_str)

    def _check_training_ranges(self, df):
        self.coverage_ = concat(
            [
                df[self.features + self.regression_features].min(skipna=True),
                df[self.features + self.regression_features].max(skipna=True),
            ],
            axis=1,
            keys=["min", "max"],
        )

    def _check_prediction_ranges(self, df):
        self.prediction_is_covered_ = (
            df[self.features + self.regression_features]
            .apply(
                lambda column: column.between(self.coverage_["min"][column.name], self.coverage_["max"][column.name],)
            )
            .all(axis=1)
        )
        if not self.prediction_is_covered_.all():
            warn(
                "Some of the subject are out of the training range. "
                "See Neuroharmony.subjects_out_of_range_ for a list of the affected subjects."
            )
            self.subjects_out_of_range_ = self.prediction_is_covered_[~self.prediction_is_covered_].index.tolist()

    def _check_data(self, df):
        type_error = "Input data should be a pandas dataframe (NDFrame)."
        assert isinstance(df.copy(), NDFrame), TypeError(type_error)
        self._check_vars(df.copy(), self.features)
        self._check_vars(df.copy(), self.covariates)
        self._check_vars(df.copy(), self.eliminate_variance)

    def _check_index(self, df):
        if df.index.duplicated().sum():
            df.index = [f'{index}_TMP_{i}' for i, index in enumerate(df.index)]
            self.reindexed = True
        return df

    def _clean_index(self, df):
        df.index = [index.split('_TMP_')[0] for index in df.index]
        return df

    def _get_pca_n_componets(self, X):
        X = self.scaler.fit_transform(X)
        self.decomposition.set_params(n_components=X.shape[1])
        self.decomposition.fit(X)
        n = next(i for i, x in enumerate(self.decomposition.explained_variance_ratio_) if x < 0.01)
        return [n + 1]

    def _clean_bad_pca_parameters(self, X):
        n_vars = len(self.regression_features) + 1
        if "PCA__n_components" in self.param_distributions.keys():
            if any([n_components > n_vars for n_components in self.param_distributions["PCA__n_components"]]):
                self.param_distributions["PCA__n_components"] = [
                    n_components
                    for n_components in self.param_distributions["PCA__n_components"]
                    if n_components > n_vars
                ]
                warn("Decomposition n_components > n_features are excluded from the parameters search.")
        else:
            self.param_distributions["PCA__n_components"] = self._get_pca_n_componets(X)

    def _random_search_with_leave_one_group_out_cv(self, X, y, groups):
        if self.decomposition.__class__.__name__ == "PCA":
            self._clean_bad_pca_parameters(X)
        self.leaveonegroupout_ = LeaveOneGroupOut()
        self.cv = list(self.leaveonegroupout_.split(X, y, groups))
        self.pipeline = Pipeline(
            steps=[
                (self.scaler.__class__.__name__, self.scaler),
                (self.decomposition.__class__.__name__, self.decomposition),
                (self.estimator.__class__.__name__, self.estimator),
            ]
        )
        self.pipeline.set_params(**self.pipeline_args)
        self.randomized_search_cv = RandomizedSearchCV(
            self.pipeline, param_distributions=self.param_distributions, cv=self.cv
        )
        self.randomized_search_cv.set_params(**self.randomized_search_args)
        self.randomized_search_cv.fit(X, y)
        return self.randomized_search_cv

    def _reconstruct_original_fieds(self, df, harmonized, extra_vars):
        """Concatenate ComBat data with the original data fields."""
        if df.index.name is not None:
            index_name = df.index.name
        else:
            index_name = "subject_index"
        if df.index.name in df.columns:
            index_name = df.index.name + "_y"
        harmonized = DataFrame(harmonized, index=df.index, columns=self.features)
        harmonized.index.rename(index_name, inplace=True)
        df = df.loc[harmonized.index][extra_vars].reset_index().copy()
        harmonized = harmonized.reset_index()
        return merge(harmonized, df, how="inner", on=index_name).set_index(index_name)

    def _train_neurofind(self, estimator=RandomForestRegressor()):
        return estimator.__class__.__name__

    def _run_combat(self, df):
        self.extra_vars = df.columns[~df.columns.isin(self.features)]
        combat = ComBat(self.features, self.covariates, self.eliminate_variance)
        self.X_harmonized_ = combat.transform(df.copy())
        self.X_harmonized_ = _label_decode_covariates(
            self.X_harmonized_,
            unique(self.covariates + self.eliminate_variance),
            self.encoders,
        )
        self.X_harmonized_.drop_duplicates(inplace=True)
        delta = df[self.features].subtract(self.X_harmonized_[self.features])
        # if delta.shape != df.shape:
        #     raise ValueError(f'DELTA = {delta.shape}, DF = {df.shape}, xh = {self.X_harmonized_.shape}')
        # y_train_split = merge(delta, df[self.extra_vars], how="inner", on=index_name).set_index(index_name)
        y_train_split = concat([delta, df[self.extra_vars]], axis=1, sort=False).dropna()
        X_train_split = df.loc[y_train_split.index]
        return X_train_split, y_train_split
