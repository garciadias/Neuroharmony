"""
=================================
Train your own Neuroharmony model
=================================

In this example, we show how to train an instance of Neuroharmony. The dataset we use here is very limited, and the
hyperparameters are not well explored, so we do not expect good results. This is an example of how to format the data
and run the training.
"""
from matplotlib.colors import LogNorm
from neuroharmony import exclude_single_subject_groups, fetch_sample, ks_test_grid, Neuroharmony
from neuroharmony.data.rois import rois
from seaborn import heatmap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Load the data.
# You can do as you wish, as long as the input to Neuroharmony is a NDFrame (pandas).
X = fetch_sample()
features = rois
covariates = ["Gender", "scanner", "Age"]
exclude_vars = X.columns[X.isna().sum() != 0].to_list() + X.columns[X.dtypes == 'O'].to_list() + ['Dataset', 'Diagn']
regression_features = [var for var in X.columns if var not in covariates + features + exclude_vars]
eliminate_variance = ["scanner"]

X.Age = X.Age.astype(int)
scanners = X.scanner.unique()
n_scanners = len(scanners)
# Split train and test leaving one scanner out.
train_bool = X.scanner.isin(scanners[1:])
test_bool = X.scanner.isin(scanners[:1])
X_train_split = X[train_bool][regression_features + covariates + rois]
X_test_split = X[test_bool][regression_features + covariates + rois]
x_train, x_test = X_train_split, X_test_split
x_train = exclude_single_subject_groups(x_train, covariates)

# Create the neuroharmony model.
# Here you can establish the range of the hyperparameters random search or give specific values.
harmony = Neuroharmony(
    features,
    regression_features,
    covariates,
    eliminate_variance,
    param_distributions=dict(
        RandomForestRegressor__n_estimators=[10, 20, 50],
        RandomForestRegressor__random_state=[42, 78],
        RandomForestRegressor__warm_start=[False, True],
    ),
    estimator_args=dict(n_jobs=1, random_state=42),
    randomized_search_args=dict(cv=5, n_jobs=8),
)
# Fit the model.
x_train_harmonized = harmony.fit_transform(x_train)
# Predict correction to unseen data.
x_test_harmonized = harmony.transform(x_test)
# Compose a NDFrame with all the data.
data_harmonized = pd.concat([x_train_harmonized, x_test_harmonized], sort=False)
# Use Kolmogorov-Smirnov test to stablish if the differences between scanners were indeed eliminated.
KS_ORIGINAL = ks_test_grid(X, features, "scanner")
KS_HARMONIZED = ks_test_grid(data_harmonized, features, "scanner")

KS_HARMONIZED_ARRAY = np.zeros((n_scanners, n_scanners, 101))
KS_ORIGINAL_ARRAY = np.zeros((n_scanners, n_scanners, 101))
for i_var, var in enumerate(rois):
    KS_HARMONIZED_ARRAY[:, :, i_var] = KS_HARMONIZED[var]
    KS_ORIGINAL_ARRAY[:, :, i_var] = KS_HARMONIZED[var]
MIN_KS_ORIGINAL = pd.DataFrame(np.nanmin(KS_ORIGINAL_ARRAY, axis=2), index=scanners, columns=scanners).fillna(0)
MIN_KS_HARMONIZED = pd.DataFrame(np.nanmin(KS_HARMONIZED_ARRAY, axis=2), index=scanners, columns=scanners).fillna(0)
MIN_KS = MIN_KS_ORIGINAL + MIN_KS_HARMONIZED.T

vmin, vmax = 1e-4, 1e0
cbar_ticks = [10**i for i in np.arange(np.log10(vmin), np.log10(vmax) + 1)]
fig = plt.figure(figsize=(2 * 5.2283465, 1.2 * 5.2283465))
ax = fig.add_subplot(111)
ax = heatmap(MIN_KS,
             cmap='BrBG', norm=LogNorm(vmin=vmin, vmax=vmax),
             cbar_kws=dict(ticks=cbar_ticks, pad=0.005), vmin=vmin, vmax=vmax, ax=ax)
plt.title('Kolmogorov-Smirnov test (p-value)', fontsize=20)
plt.tick_params(labelsize=12)
plt.minorticks_off()
plt.subplots_adjust(left=0.175, bottom=0.33, top=0.95, right=1.075)
plt.tick_params(labelsize=11)
plt.gca().set_xticks(np.arange(0.5, len(MIN_KS)))
plt.gca().set_xticklabels(MIN_KS.index)
plt.show()
