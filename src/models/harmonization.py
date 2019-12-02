"""Tools for harmonization."""

import os
import sys

from neuroCombat import neuroCombat
from numpy import unique
from pandas.core.generic import NDFrame
from pandas import Series, DataFrame, concat
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder, RobustScaler

from src.data.rois import rois


def supress_print(func):
    """Define decorator for print supression."""
    def wrapper(*args, **kwargs):
        sys.stdout = open(os.devnull, 'w')
        resuts = func(*args, **kwargs)
        sys.stdout = sys.__stdout__
        return resuts
    return wrapper


@supress_print
def combat(*args, **kwargs):
    """Redefine ComBat to supress printing unhrlpful data."""
    return neuroCombat(*args, **kwargs)


class ComBat(BaseEstimator, TransformerMixin):
    """ComBat class."""

    def __init__(self, features, covars):
        """Init class with the original data."""
        self.features = features
        self.covars = covars

    def _check_data(self, df):
        type_error = "Input data should be a pandas dataframe (NDFrame)."
        assert isinstance(df, NDFrame), TypeError(type_error)
        self._check_vars(df, self.features)
        self._check_vars(df, self.covars)

    def _check_vars(self, df, vars):
        vars = Series(vars)
        is_feature_present = vars.isin(df.columns)
        missing_features_str = "Missing features: %s" % ', '.join(vars[~is_feature_present])
        assert is_feature_present.all(), ValueError(missing_features_str)

    def _reconstruct_original_fieds(self, df, harmonized, extra_vars):
        """Concatenate ComBat data with the original data fields."""
        harmonized = DataFrame(harmonized, index=df.index, columns=self.features)
        return concat([harmonized, df.loc[harmonized.index][extra_vars]], axis=1, sort=True)

    def _label_encode_covars(self, df):
        self.encoders = {}
        for covar in self.covars:
            self.encoders[covar] = LabelEncoder()
            df[covar] = self.encoders[covar].fit_transform(df[covar])
        return df

    def _label_dencode_covars(self, df):
        for covar in self.covars:
            df[covar] = self.encoders[covar].inverse_transform(df[covar])
        return df

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
        for batch_col in self.covars:
            harmonized = combat(data=harmonized[self.features],
                                covars=harmonized[self.covars],
                                batch_col=batch_col, )
            harmonized = self._reconstruct_original_fieds(df, harmonized, extra_vars)
        return self._label_dencode_covars(harmonized)

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
        df = self._label_encode_covars(df)
        df = self._exclude_single_subject_groups(df)
        df = self._exclude_subjects_with_nans(df)
        return self._run_combat(df)

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
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.
        .. versionchanged:: 0.20
           The default value of ``n_estimators`` will change from 10 in
           version 0.20 to 100 in version 0.22.
    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.
        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.
    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
        .. versionchanged:: 0.18
           Added float values for fractions.
    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
        .. versionchanged:: 0.18
           Added float values for fractions.
    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        The weighted impurity decrease equation is the following::
            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)
        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.
        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.
        .. versionadded:: 0.19
    min_impurity_split : float, (default=1e-7)
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.
        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19. The default value of
           ``min_impurity_split`` will change from 1e-7 to 0 in 0.23 and it
           will be removed in 0.25. Use ``min_impurity_decrease`` instead.
    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees. If False, the
        whole datset is used to build each tree.
    oob_score : bool, optional (default=False)
        whether to use out-of-bag samples to estimate
        the R^2 on unseen data.
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        `None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    verbose : int, optional (default=0)
        Controls the verbosity when fitting and predicting.
    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.
    with_centering : boolean, True by default
        If True, center the data before scaling.
        This will cause ``transform`` to raise an exception when attempted on
        sparse matrices, because centering them entails building a dense
        matrix which in common use cases is likely to be too large to fit in
        memory.
    with_scaling : boolean, True by default
        If True, scale the data to interquartile range.
    quantile_range : tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0
        Default: (25.0, 75.0) = (1st quantile, 3rd quantile) = IQR
        Quantile range used to calculate ``scale_``.
        .. versionadded:: 0.18
    copy : boolean, optional, default is True
        If False, try to avoid a copy and do inplace scaling instead.
        This is not guaranteed to always work inplace; e.g. if the data is
        not a NumPy array or scipy.sparse CSR matrix, a copy may still be
        returned.
    steps : list
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.
    memory : None, str or object with the joblib.Memory interface, optional
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.
    verbose : boolean, optional
        If True, the time elapsed while fitting each step will be printed as it
        is completed.
    estimator : estimator object.
        A object of that type is instantiated for each grid point.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.
    param_distributions : dict
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.
    n_iter : int, default=10
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution.
    scoring : string, callable, list/tuple, dict or None, default: None
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.
        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.
        NOTE that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.
        See :ref:`multimetric_grid_search` for an example.
        If None, the estimator's score method is used.
    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'
    iid : boolean, default='warn'
        If True, return the average score across folds, weighted by the number
        of samples in each test set. In this case, the data is assumed to be
        identically distributed across the folds, and the loss minimized is
        the total loss per sample, and not the mean loss across the folds. If
        False, return the average score across folds. Default is True, but
        will change to False in version 0.22, to correspond to the standard
        definition of cross-validation.
        .. versionchanged:: 0.20
            Parameter ``iid`` will change from True to False by default in
            version 0.22, and will be removed in 0.24.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.
        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
        .. versionchanged:: 0.20
            ``cv`` default value if None will change from 3-fold to 5-fold
            in v0.22.
    refit : boolean, string, or callable, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.
        For multiple metric evaluation, this needs to be a string denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.
        Where there are considerations other than maximum score in
        choosing a best estimator, ``refit`` can be set to a function which
        returns the selected ``best_index_`` given the ``cv_results``.
        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``RandomizedSearchCV`` instance.
        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_params_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer. When refit is callable, ``best_score_`` is disabled.
        See ``scoring`` parameter to know more about multiple metric
        evaluation.
        .. versionchanged:: 0.20
            Support for callable added.
    verbose : integer
        Controls the verbosity: the higher, the more messages.
    random_state : int, RandomState instance or None, optional, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    error_score : 'raise' or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error. Default is 'raise' but from
        version 0.22 it will change to np.nan.
    return_train_score : boolean, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.
    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).
    n_features_ : int
        The number of features when ``fit`` is performed.
    n_outputs_ : int
        The number of outputs when ``fit`` is performed.
    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
    oob_prediction_ : array of shape = [n_samples]
        Prediction computed with out-of-bag estimate on the training set.
    center_ : array of floats
        The median value for each feature in the training set.
    scale_ : array of floats
        The (scaled) interquartile range for each feature in the training set.
    named_steps : bunch object, a dictionary with attribute access
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.
        For instance the below given table
        +--------------+-------------+-------------------+---+---------------+
        | param_kernel | param_gamma | split0_test_score |...|rank_test_score|
        +==============+=============+===================+===+===============+
        |    'rbf'     |     0.1     |       0.80        |...|       2       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.2     |       0.90        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.3     |       0.70        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+
        will be represented by a ``cv_results_`` dict of::
            {
            'param_kernel' : masked_array(data = ['rbf', 'rbf', 'rbf'],
                                          mask = False),
            'param_gamma'  : masked_array(data = [0.1 0.2 0.3], mask = False),
            'split0_test_score'  : [0.80, 0.90, 0.70],
            'split1_test_score'  : [0.82, 0.50, 0.70],
            'mean_test_score'    : [0.81, 0.70, 0.70],
            'std_test_score'     : [0.01, 0.20, 0.00],
            'rank_test_score'    : [3, 1, 1],
            'split0_train_score' : [0.80, 0.92, 0.70],
            'split1_train_score' : [0.82, 0.55, 0.70],
            'mean_train_score'   : [0.81, 0.74, 0.70],
            'std_train_score'    : [0.01, 0.19, 0.00],
            'mean_fit_time'      : [0.73, 0.63, 0.43],
            'std_fit_time'       : [0.01, 0.02, 0.01],
            'mean_score_time'    : [0.01, 0.06, 0.04],
            'std_score_time'     : [0.00, 0.00, 0.00],
            'params'             : [{'kernel' : 'rbf', 'gamma' : 0.1}, ...],
            }
        NOTE
        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.
        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.
        For multi-metric evaluation, the scores for all the scorers are
        available in the ``cv_results_`` dict at the keys ending with that
        scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
        above. ('split0_test_precision', 'mean_train_precision' etc.)
    best_estimator_ : estimator or dict
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.
        For multi-metric evaluation, this attribute is present only if
        ``refit`` is specified.
        See ``refit`` parameter for more information on allowed values.
    best_score_ : float
        Mean cross-validated score of the best_estimator.
        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.
    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.
        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.
    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.
        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).
        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.
    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.
        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.
    n_splits_ : int
        The number of cross-validation splits (folds/iterations).
    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.
        This is present only if ``refit`` is not False.

    """

    param_distribution = {'randomforestregressor__n_estimators': [100, 200, 500],
                          'randomforestregressor__warm_start': [False, True],
                          }

    def __init__(self,
                 features=rois,
                 covars=['Gender', 'scanner', 'Age'],
                 estimator=RandomForestRegressor(),
                 scaler=RobustScaler(),
                 param_distributions=param_distribution,
                 bootstrap=True,
                 copy=True,
                 criterion="mae",
                 cv='warn',
                 error_score='raise-deprecating',
                 iid='warn',
                 max_depth=None,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 min_samples_leaf=1,
                 min_samples_split=2,
                 min_weight_fraction_leaf=0.,
                 n_estimators='warn',
                 n_iter=10,
                 n_jobs=None,
                 oob_score=False,
                 pre_dispatch='2*n_jobs',
                 quantile_range=(25.0, 75.0),
                 random_state=None,
                 refit=True,
                 return_train_score=False,
                 scoring=None,
                 verbose=0,
                 warm_start=False,
                 with_centering=True,
                 with_scaling=True,
                 ):
        """Init."""
        self.features = features
        self.covars = covars
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.bootstrap = bootstrap
        self.copy = copy
        self.criterion = criterion
        self.cv = cv
        self.error_score = error_score
        self.iid = iid
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.n_estimators = n_estimators
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.oob_score = oob_score
        self.pre_dispatch = pre_dispatch
        self.quantile_range = quantile_range
        self.random_state = random_state
        self.refit = refit
        self.return_train_score = return_train_score
        self.scoring = scoring
        self.verbose = verbose
        self.warm_start = warm_start
        self.with_centering = with_centering
        self.with_scaling = with_scaling

    def _random_search_with_leave_one_group_out_cv(self, X, y, groups):
        self.leaveonegroupout_ = LeaveOneGroupOut()
        self.cv_ = list(self.leaveonegroupout_.split(X, y, groups))
        pipeline = Pipeline(
            steps=[(self.scaler.__class__.__name__, self.scaler),
                   (self.estimator.__class__.__name__, self.estimator),
                   ])
        self.randomized_search_cv_ = RandomizedSearchCV(pipeline,
                                                        param_distributions=self.param_distributions,
                                                        n_iter=self.n_iter,
                                                        scoring=self.scoring,
                                                        n_jobs=self.n_jobs,
                                                        iid=self.iid,
                                                        refit=self.refit,
                                                        cv=self.cv_,
                                                        verbose=self.verbose,
                                                        pre_dispatch=self.pre_dispatch,
                                                        random_state=self.random_state,
                                                        error_score=self.error_score,
                                                        return_train_score=self.return_train_score,
                                                        )
        self.randomized_search_cv_.fit(X, y)
        return self.randomized_search_cv_.fit(X, y)

    def _label_encode_covars(self, df):
        self.encoders = {}
        for covar in self.covars:
            self.encoders[covar] = LabelEncoder()
            df[covar] = self.encoders[covar].fit_transform(df[covar])
        return df

    def _label_dencode_covars(self, df):
        for covar in self.covars:
            df[covar] = self.encoders[covar].inverse_transform(df[covar])
        return df

    def _train_neurofind(self, estimator=RandomForestRegressor()):
        name = estimator.__class__.__name__

    def _run_combat(self, df):
        self.extra_vars = df.columns[~df.columns.isin(self.features)]
        combat = ComBat(self.features, self.covars)
        X_harmonized = combat.transform(df)
        delta = df[self.features] - X_harmonized[self.features]
        delta = concat([delta, df[self.extra_vars]], axis=1, sort=False).dropna()
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
        df = self._label_encode_covars(df)
        X_train_split, y_train_split = self._run_combat(df)
        [print(len(x)) for x in [X_train_split, y_train_split, y_train_split.scanner]]
        self._random_search_with_leave_one_group_out_cv(
            X_train_split, y_train_split, y_train_split.scanner)
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
        pass
