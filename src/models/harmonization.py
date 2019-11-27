"""Tools for harmonization."""

import os
import sys

from neuroCombat import neuroCombat
from pandas.core.generic import NDFrame
from pandas import Series, DataFrame, concat
from sklearn.preprocessing import LabelEncoder
from numpy import unique


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


class ComBat():
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

    def transform(self, X):
        """Run ComBat normalization.

        Transforms.

        Parameters
        ----------
        X: NDFrame, shape(n_subjects, n_features)
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

                      Left-Lateral-Ventricle  Left-Inf-Lat-Vent  Gender  Dataset  Age
        sub-002-00-2  0.007666                0.000134           0       2        31
        sub-012-00-1  0.003778                0.000064           1       1        14
        sub-013-00-1  0.012818                0.000464           1       1        17
        sub-014-00-1  0.006289                0.000167           0       1        19
        sub-015-00-1  0.003310                0.000154           1       1        26

        """
        self._check_data(X)
        X = self._label_encode_covars(X)
        X = self._exclude_single_subject_groups(X)
        X = self._exclude_subjects_with_nans(X)
        return self._run_combat(X)


class Neuroharmony(object):
    """docstring for Neuroharmony."""

    def __init__(self):
        super(Neuroharmony, self).__init__()
