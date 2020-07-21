"""Metrics for evaluation of the normalization."""
from __future__ import (nested_scopes, generators, division, absolute_import,
                        with_statement, print_function, unicode_literals)

from itertools import combinations

import numpy as np
from pandas import DataFrame, Series
from scipy.stats import ks_2samp


def check_vars(df, features):
    """Check if all variables in a list are present in a dataframe."""
    features = Series(features)
    is_feature_present = features.isin(df.columns)
    missing_features_str = "Missing features: %s" % ', '.join(features[~is_feature_present])
    assert is_feature_present.all(), ValueError(missing_features_str)


def ks_test_grid(df, features, sampling_variable='scanner'):
    """Calculate the Kolmogorov-Smirnov score for all pairs of scanners.

    Parameters
    ----------
    df: NDFrame of shape [n_subjects, n_features]
        DataFrame with the subjects data.

    features: list
        List of the features to be considered on the Kolmogorov-Smirnov test.

    sampling_variable: str, default='scanner'
        variable for which you want to group subjects.

    Returns
    -------
    KS_by_variable: dict of NDFrames
        Kolmogorov-Smirnov p-values to all pairs of instances in the sampling_variable column.
        The keys in the dictionary are the variables in 'features'. The values of each entry are square NDFrames of
        shape [n_vars, n_vars].

    Raises
    ------
    ValueError:
        If the list of variables contain any variable that is not present in df.

    Examples
    --------
    >>> ixi = DataSet('data/raw/IXI').data
    >>> features = ['Left-Lateral-Ventricle', 'Left-Inf-Lat-Vent', ]
    >>> KS = ks_test_grid(df, features, 'scanner')
    >>> KS[features[0]]
    +--------------------------+----------------------+------------------------+--------------------+
    |                          | SCANNER01-SCANNER01  | SCANNER02-SCANNER01    | SCANNER03-SCANNER01|
    +==========================++=====================+========================+====================+
    |SCANNER01-SCANNER01       | NaN                  | NaN                    | NaN                |
    +--------------------------+----------------------+------------------------+--------------------+
    |SCANNER02-SCANNER01       | 0.000759473          | NaN                    | NaN                |
    +--------------------------+----------------------+------------------------+--------------------+
    |SCANNER03-SCANNER01       | 0.0539998            | 0.625887               | NaN                |
    +--------------------------+----------------------+------------------------+--------------------+
    """
    check_vars(df, features)
    groups = df.groupby(sampling_variable)
    scanners_list = df[sampling_variable].unique()
    scanners_list = scanners_list[np.argsort(df[sampling_variable].str.lower().unique())]
    KS_by_variable = {}
    for var in features:
        KS = DataFrame([], index=scanners_list, columns=scanners_list)
        for scanner_batch in np.array_split(np.array(list(combinations(scanners_list, 2))), 80):
            for scanner_a, scanner_b in scanner_batch:
                group_a = groups.get_group(scanner_a)[var]
                group_b = groups.get_group(scanner_b)[var]
                KS.loc[scanner_b][scanner_a] = ks_2samp(group_a, group_b).pvalue
        KS_by_variable[var] = KS
    return KS_by_variable
