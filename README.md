# Install Neuroharmony.

Obs: <span style="color:red"> The tests in this repository depend on data from [IXI dataset](https://brain-development.org/ixi-dataset/). We will soon provide preprocessed/formatted data for this tests.</span>

* Clone the repository:
```
git clone https://github.com/garciadias/Neuroharmony.git
```
* Install:
```
python setup.py install
```

# Example of use:

```python
from neuroharmony import Neuroharmony
from neuroharmony.data.rois import rois
from neuroharmony.data.combine_tools import DataSet
from neuroharmony.models.metrics import ks_test_grid

# Load the data.
# You can do as you wish, as long as the input to Neuroharmony is a NDFrame (pandas).
data_path = 'data/raw/IXI'
features = rois[:3]
regression_features = ['Age', 'summary_gm_median', 'spacing_x', 'summary_gm_p95',
                         'cnr', 'size_x', 'cjv', 'summary_wm_mean', 'icvs_gm', 'wm2max']
covars = ['Gender', 'scanner', 'Age']
original_data = DataSet(Path(data_path)).data
original_data.Age = original_data.Age.astype(int)
scanners = original_data.unique()
# Split train and test leaving one scanner out.
train_bool = original_data.isin(scanners[1:])
test_bool = original_data.isin(scanners[:1])
X_train_split = original_data[train_bool]
X_test_split = original_data[test_bool]
n_scanners = len(original_data.scanner.unique())
x_train, x_test = X_train_split, X_test_split

# Create the neuroharmony model.
# Here you can establish the range of the hyperparameters random search or give specific values.
harmony = Neuroharmony(features,
                            regression_features,
                            covars,
                            param_distributions=dict(
                                RandomForestRegressor__n_estimators=[5, 10, 15, 20],
                                RandomForestRegressor__random_state=[42, 78],
                                RandomForestRegressor__warm_start=[False, True],
                            ),
                            estimator_args=dict(n_jobs=1, random_state=42),
                            randomized_search_args=dict(cv=5, n_jobs=27))
# Fit the model.
x_train_harmonized = harmony.fit_transform(x_train)
# Predict correction to unseen data.
x_test_harmonized = harmony.predict(x_test)
# Compose a NDFrame with all the data.
data_harmonized = concat([x_train_harmonized, x_test_harmonized], sort=False)
# Use Kolmogorov-Smirnov test to stablish if the differences between scanners were indeed eliminated.
KS_original = ks_test_grid(original_data, features, 'scanner')
KS_harmonized = ks_test_grid(data_harmonized, features, 'scanner')
```
