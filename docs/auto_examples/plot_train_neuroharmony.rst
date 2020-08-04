.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_plot_train_neuroharmony.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_plot_train_neuroharmony.py:


=================================
Train your own Neuroharmony model
=================================

In this example, we show how to train an instance of Neuroharmony. The dataset we use here is very limited, and the
hyperparameters are not well explored, so we do not expect good results. This is an example of how to format the data
and run the training.



.. image:: /auto_examples/images/sphx_glr_plot_train_neuroharmony_001.png
    :alt: Kolmogorov-Smirnov test (p-value)
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    0.00iB [00:00, ?iB/s]    967kiB [00:00, 11.1MiB/s]
    Randomized search of Neuroharmony hyperparameters:   0%|          | 0/101 [00:00<?, ?it/s]    Randomized search of Neuroharmony hyperparameters:   1%|          | 1/101 [00:01<03:10,  1.90s/it]    Randomized search of Neuroharmony hyperparameters:   2%|1         | 2/101 [00:02<02:25,  1.47s/it]    Randomized search of Neuroharmony hyperparameters:   3%|2         | 3/101 [00:02<01:57,  1.20s/it]    Randomized search of Neuroharmony hyperparameters:   4%|3         | 4/101 [00:03<01:41,  1.04s/it]    Randomized search of Neuroharmony hyperparameters:   5%|4         | 5/101 [00:04<01:22,  1.16it/s]    Randomized search of Neuroharmony hyperparameters:   6%|5         | 6/101 [00:04<01:10,  1.35it/s]    Randomized search of Neuroharmony hyperparameters:   7%|6         | 7/101 [00:05<01:03,  1.48it/s]    Randomized search of Neuroharmony hyperparameters:   8%|7         | 8/101 [00:05<00:57,  1.61it/s]    Randomized search of Neuroharmony hyperparameters:   9%|8         | 9/101 [00:06<00:53,  1.72it/s]    Randomized search of Neuroharmony hyperparameters:  10%|9         | 10/101 [00:06<00:49,  1.82it/s]    Randomized search of Neuroharmony hyperparameters:  11%|#         | 11/101 [00:07<00:50,  1.77it/s]    Randomized search of Neuroharmony hyperparameters:  12%|#1        | 12/101 [00:07<00:49,  1.80it/s]    Randomized search of Neuroharmony hyperparameters:  13%|#2        | 13/101 [00:08<00:47,  1.86it/s]    Randomized search of Neuroharmony hyperparameters:  14%|#3        | 14/101 [00:08<00:44,  1.94it/s]    Randomized search of Neuroharmony hyperparameters:  15%|#4        | 15/101 [00:09<00:42,  2.03it/s]    Randomized search of Neuroharmony hyperparameters:  16%|#5        | 16/101 [00:09<00:42,  1.99it/s]    Randomized search of Neuroharmony hyperparameters:  17%|#6        | 17/101 [00:10<00:45,  1.85it/s]    Randomized search of Neuroharmony hyperparameters:  18%|#7        | 18/101 [00:10<00:44,  1.85it/s]    Randomized search of Neuroharmony hyperparameters:  19%|#8        | 19/101 [00:11<00:46,  1.78it/s]    Randomized search of Neuroharmony hyperparameters:  20%|#9        | 20/101 [00:12<00:48,  1.66it/s]    Randomized search of Neuroharmony hyperparameters:  21%|##        | 21/101 [00:12<00:45,  1.76it/s]    Randomized search of Neuroharmony hyperparameters:  22%|##1       | 22/101 [00:12<00:42,  1.86it/s]    Randomized search of Neuroharmony hyperparameters:  23%|##2       | 23/101 [00:13<00:42,  1.85it/s]    Randomized search of Neuroharmony hyperparameters:  24%|##3       | 24/101 [00:14<00:40,  1.89it/s]    Randomized search of Neuroharmony hyperparameters:  25%|##4       | 25/101 [00:14<00:39,  1.91it/s]    Randomized search of Neuroharmony hyperparameters:  26%|##5       | 26/101 [00:15<00:39,  1.91it/s]    Randomized search of Neuroharmony hyperparameters:  27%|##6       | 27/101 [00:15<00:37,  1.99it/s]    Randomized search of Neuroharmony hyperparameters:  28%|##7       | 28/101 [00:16<00:38,  1.92it/s]    Randomized search of Neuroharmony hyperparameters:  29%|##8       | 29/101 [00:16<00:36,  1.95it/s]    Randomized search of Neuroharmony hyperparameters:  30%|##9       | 30/101 [00:17<00:34,  2.04it/s]    Randomized search of Neuroharmony hyperparameters:  31%|###       | 31/101 [00:17<00:35,  2.00it/s]    Randomized search of Neuroharmony hyperparameters:  32%|###1      | 32/101 [00:18<00:34,  1.99it/s]    Randomized search of Neuroharmony hyperparameters:  33%|###2      | 33/101 [00:18<00:35,  1.93it/s]    Randomized search of Neuroharmony hyperparameters:  34%|###3      | 34/101 [00:19<00:33,  2.00it/s]    Randomized search of Neuroharmony hyperparameters:  35%|###4      | 35/101 [00:19<00:33,  1.99it/s]    Randomized search of Neuroharmony hyperparameters:  36%|###5      | 36/101 [00:20<00:32,  1.98it/s]    Randomized search of Neuroharmony hyperparameters:  37%|###6      | 37/101 [00:20<00:32,  1.95it/s]    Randomized search of Neuroharmony hyperparameters:  38%|###7      | 38/101 [00:21<00:31,  2.01it/s]    Randomized search of Neuroharmony hyperparameters:  39%|###8      | 39/101 [00:21<00:30,  2.04it/s]    Randomized search of Neuroharmony hyperparameters:  40%|###9      | 40/101 [00:22<00:30,  2.01it/s]    Randomized search of Neuroharmony hyperparameters:  41%|####      | 41/101 [00:22<00:31,  1.93it/s]    Randomized search of Neuroharmony hyperparameters:  42%|####1     | 42/101 [00:23<00:31,  1.89it/s]    Randomized search of Neuroharmony hyperparameters:  43%|####2     | 43/101 [00:23<00:30,  1.88it/s]    Randomized search of Neuroharmony hyperparameters:  44%|####3     | 44/101 [00:24<00:29,  1.96it/s]    Randomized search of Neuroharmony hyperparameters:  45%|####4     | 45/101 [00:24<00:28,  1.99it/s]    Randomized search of Neuroharmony hyperparameters:  46%|####5     | 46/101 [00:25<00:27,  1.97it/s]    Randomized search of Neuroharmony hyperparameters:  47%|####6     | 47/101 [00:25<00:28,  1.88it/s]    Randomized search of Neuroharmony hyperparameters:  48%|####7     | 48/101 [00:26<00:27,  1.93it/s]    Randomized search of Neuroharmony hyperparameters:  49%|####8     | 49/101 [00:26<00:25,  2.02it/s]    Randomized search of Neuroharmony hyperparameters:  50%|####9     | 50/101 [00:27<00:25,  1.99it/s]    Randomized search of Neuroharmony hyperparameters:  50%|#####     | 51/101 [00:27<00:24,  2.01it/s]    Randomized search of Neuroharmony hyperparameters:  51%|#####1    | 52/101 [00:28<00:23,  2.08it/s]    Randomized search of Neuroharmony hyperparameters:  52%|#####2    | 53/101 [00:28<00:23,  2.06it/s]    Randomized search of Neuroharmony hyperparameters:  53%|#####3    | 54/101 [00:29<00:24,  1.95it/s]    Randomized search of Neuroharmony hyperparameters:  54%|#####4    | 55/101 [00:29<00:22,  2.05it/s]    Randomized search of Neuroharmony hyperparameters:  55%|#####5    | 56/101 [00:30<00:22,  1.98it/s]    Randomized search of Neuroharmony hyperparameters:  56%|#####6    | 57/101 [00:30<00:21,  2.01it/s]    Randomized search of Neuroharmony hyperparameters:  57%|#####7    | 58/101 [00:31<00:21,  1.99it/s]    Randomized search of Neuroharmony hyperparameters:  58%|#####8    | 59/101 [00:31<00:21,  1.94it/s]    Randomized search of Neuroharmony hyperparameters:  59%|#####9    | 60/101 [00:32<00:21,  1.88it/s]    Randomized search of Neuroharmony hyperparameters:  60%|######    | 61/101 [00:32<00:21,  1.88it/s]    Randomized search of Neuroharmony hyperparameters:  61%|######1   | 62/101 [00:33<00:20,  1.87it/s]    Randomized search of Neuroharmony hyperparameters:  62%|######2   | 63/101 [00:33<00:20,  1.83it/s]    Randomized search of Neuroharmony hyperparameters:  63%|######3   | 64/101 [00:34<00:19,  1.88it/s]    Randomized search of Neuroharmony hyperparameters:  64%|######4   | 65/101 [00:34<00:19,  1.86it/s]    Randomized search of Neuroharmony hyperparameters:  65%|######5   | 66/101 [00:35<00:18,  1.94it/s]    Randomized search of Neuroharmony hyperparameters:  66%|######6   | 67/101 [00:35<00:17,  1.96it/s]    Randomized search of Neuroharmony hyperparameters:  67%|######7   | 68/101 [00:36<00:16,  2.02it/s]    Randomized search of Neuroharmony hyperparameters:  68%|######8   | 69/101 [00:36<00:15,  2.03it/s]    Randomized search of Neuroharmony hyperparameters:  69%|######9   | 70/101 [00:37<00:15,  2.05it/s]    Randomized search of Neuroharmony hyperparameters:  70%|#######   | 71/101 [00:37<00:15,  1.98it/s]    Randomized search of Neuroharmony hyperparameters:  71%|#######1  | 72/101 [00:38<00:14,  2.03it/s]    Randomized search of Neuroharmony hyperparameters:  72%|#######2  | 73/101 [00:38<00:13,  2.05it/s]    Randomized search of Neuroharmony hyperparameters:  73%|#######3  | 74/101 [00:39<00:13,  2.00it/s]    Randomized search of Neuroharmony hyperparameters:  74%|#######4  | 75/101 [00:39<00:13,  1.89it/s]    Randomized search of Neuroharmony hyperparameters:  75%|#######5  | 76/101 [00:40<00:13,  1.88it/s]    Randomized search of Neuroharmony hyperparameters:  76%|#######6  | 77/101 [00:41<00:13,  1.79it/s]    Randomized search of Neuroharmony hyperparameters:  77%|#######7  | 78/101 [00:41<00:12,  1.91it/s]    Randomized search of Neuroharmony hyperparameters:  78%|#######8  | 79/101 [00:42<00:11,  1.95it/s]    Randomized search of Neuroharmony hyperparameters:  79%|#######9  | 80/101 [00:42<00:10,  1.92it/s]    Randomized search of Neuroharmony hyperparameters:  80%|########  | 81/101 [00:43<00:10,  1.85it/s]    Randomized search of Neuroharmony hyperparameters:  81%|########1 | 82/101 [00:43<00:09,  1.90it/s]    Randomized search of Neuroharmony hyperparameters:  82%|########2 | 83/101 [00:44<00:09,  1.93it/s]    Randomized search of Neuroharmony hyperparameters:  83%|########3 | 84/101 [00:44<00:08,  1.95it/s]    Randomized search of Neuroharmony hyperparameters:  84%|########4 | 85/101 [00:45<00:08,  1.99it/s]    Randomized search of Neuroharmony hyperparameters:  85%|########5 | 86/101 [00:45<00:07,  2.04it/s]    Randomized search of Neuroharmony hyperparameters:  86%|########6 | 87/101 [00:46<00:06,  2.15it/s]    Randomized search of Neuroharmony hyperparameters:  87%|########7 | 88/101 [00:46<00:06,  1.98it/s]    Randomized search of Neuroharmony hyperparameters:  88%|########8 | 89/101 [00:47<00:05,  2.03it/s]    Randomized search of Neuroharmony hyperparameters:  89%|########9 | 90/101 [00:47<00:05,  1.91it/s]    Randomized search of Neuroharmony hyperparameters:  90%|######### | 91/101 [00:48<00:05,  1.86it/s]    Randomized search of Neuroharmony hyperparameters:  91%|#########1| 92/101 [00:48<00:04,  1.93it/s]    Randomized search of Neuroharmony hyperparameters:  92%|#########2| 93/101 [00:49<00:04,  1.86it/s]    Randomized search of Neuroharmony hyperparameters:  93%|#########3| 94/101 [00:49<00:04,  1.74it/s]    Randomized search of Neuroharmony hyperparameters:  94%|#########4| 95/101 [00:50<00:03,  1.80it/s]    Randomized search of Neuroharmony hyperparameters:  95%|#########5| 96/101 [00:51<00:02,  1.81it/s]    Randomized search of Neuroharmony hyperparameters:  96%|#########6| 97/101 [00:51<00:02,  1.92it/s]    Randomized search of Neuroharmony hyperparameters:  97%|#########7| 98/101 [00:51<00:01,  1.95it/s]    Randomized search of Neuroharmony hyperparameters:  98%|#########8| 99/101 [00:52<00:01,  1.87it/s]    Randomized search of Neuroharmony hyperparameters:  99%|#########9| 100/101 [00:53<00:00,  1.93it/s]    Randomized search of Neuroharmony hyperparameters: 100%|##########| 101/101 [00:53<00:00,  1.98it/s]
    /home/rgd/training/lib/python3.7/site-packages/Neuroharmony-0.0.1.0-py3.7.egg/neuroharmony/models/harmonization.py:459: UserWarning: Some of the subject are out of the training range. See Neuroharmony.subjects_out_of_range_ for a list of the affected subjects.
      "Some of the subject are out of the training range. "
    /home/rgd/git/neuroharmony_doc/examples/plot_train_neuroharmony.py:67: RuntimeWarning: All-NaN slice encountered
      MIN_KS_ORIGINAL = pd.DataFrame(np.nanmin(KS_ORIGINAL_ARRAY, axis=2), index=scanners, columns=scanners).fillna(0)
    /home/rgd/git/neuroharmony_doc/examples/plot_train_neuroharmony.py:68: RuntimeWarning: All-NaN slice encountered
      MIN_KS_HARMONIZED = pd.DataFrame(np.nanmin(KS_HARMONIZED_ARRAY, axis=2), index=scanners, columns=scanners).fillna(0)
    /home/rgd/git/neuroharmony_doc/examples/plot_train_neuroharmony.py:85: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      plt.show()






|


.. code-block:: default

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


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 1 minutes  21.689 seconds)


.. _sphx_glr_download_auto_examples_plot_train_neuroharmony.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_train_neuroharmony.py <plot_train_neuroharmony.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_train_neuroharmony.ipynb <plot_train_neuroharmony.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
