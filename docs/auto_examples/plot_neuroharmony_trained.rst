.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_plot_neuroharmony_trained.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_plot_neuroharmony_trained.py:


======================================
Apply a pre-trained Neuroharmony model
======================================
An example plot of how to load and apply pre-trained a Neuroharmony model.



.. image:: /auto_examples/images/sphx_glr_plot_neuroharmony_trained_001.png
    :alt: plot neuroharmony trained
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /usr/lib/python3.7/importlib/_bootstrap.py:219: RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 192 from C header, got 216 from PyObject
      return f(*args, **kwds)
    0.00iB [00:00, ?iB/s]    967kiB [00:00, 10.5MiB/s]
    /home/rgd/training/lib/python3.7/site-packages/sklearn/base.py:318: UserWarning: Trying to unpickle estimator DecisionTreeRegressor from version 0.22 when using version 0.22.2.post1. This might lead to breaking code or invalid results. Use at your own risk.
      UserWarning)
    /home/rgd/training/lib/python3.7/site-packages/sklearn/base.py:318: UserWarning: Trying to unpickle estimator RandomForestRegressor from version 0.22 when using version 0.22.2.post1. This might lead to breaking code or invalid results. Use at your own risk.
      UserWarning)
    /home/rgd/training/lib/python3.7/site-packages/sklearn/base.py:318: UserWarning: Trying to unpickle estimator StandardScaler from version 0.22 when using version 0.22.2.post1. This might lead to breaking code or invalid results. Use at your own risk.
      UserWarning)
    /home/rgd/training/lib/python3.7/site-packages/sklearn/base.py:318: UserWarning: Trying to unpickle estimator PCA from version 0.22 when using version 0.22.2.post1. This might lead to breaking code or invalid results. Use at your own risk.
      UserWarning)
    /home/rgd/training/lib/python3.7/site-packages/sklearn/base.py:318: UserWarning: Trying to unpickle estimator LabelEncoder from version 0.22 when using version 0.22.2.post1. This might lead to breaking code or invalid results. Use at your own risk.
      UserWarning)
    /home/rgd/training/lib/python3.7/site-packages/sklearn/base.py:318: UserWarning: Trying to unpickle estimator Pipeline from version 0.22 when using version 0.22.2.post1. This might lead to breaking code or invalid results. Use at your own risk.
      UserWarning)
    /home/rgd/training/lib/python3.7/site-packages/sklearn/base.py:318: UserWarning: Trying to unpickle estimator RandomizedSearchCV from version 0.22 when using version 0.22.2.post1. This might lead to breaking code or invalid results. Use at your own risk.
      UserWarning)
    /home/rgd/training/lib/python3.7/site-packages/Neuroharmony-0.0.1.0-py3.7.egg/neuroharmony/models/harmonization.py:459: UserWarning: Some of the subject are out of the training range. See Neuroharmony.subjects_out_of_range_ for a list of the affected subjects.
      "Some of the subject are out of the training range. "
    /home/rgd/git/neuroharmony/examples/plot_neuroharmony_trained.py:37: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      plt.show()






|


.. code-block:: default

    import matplotlib.pyplot as plt
    from neuroharmony import fetch_trained_model, fetch_sample
    import seaborn as sns

    X = fetch_sample()
    neuroharmony = fetch_trained_model()
    x_harmonized = neuroharmony.transform(X)

    rois = ['Left-Hippocampus',
            'lh_bankssts_volume',
            'lh_posteriorcingulate_volume',
            'lh_superiorfrontal_volume',
            'rh_frontalpole_volume',
            'rh_parsopercularis_volume',
            'rh_parstriangularis_volume',
            'rh_superiorfrontal_volume',
            'Right-Cerebellum-White-Matter',
            ]
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for roi, ax in zip(rois, axes.flatten()):
        ax.plot(neuroharmony.kde_data_[roi]['x'], neuroharmony.kde_data_[roi]['y'],
                color='#fcb85b', ls='--', label='ComBat harmonized training set')
        sns.kdeplot(X[roi], color='#f47376', ls=':', legend=False, ax=ax, label='Original test set')
        sns.kdeplot(x_harmonized[roi], color='#00bcab', ls='-', legend=False, ax=ax, label='Harmonized test set')
        ax.set_xlabel(roi, fontsize=13)
    axes.flatten()[2].legend(ncol=3, bbox_to_anchor=(0.8, 1.175), fontsize=13)
    axes.flatten()[3].set_ylabel('Density', fontsize=15)
    plt.subplots_adjust(left=0.07, right=0.99,
                        bottom=0.05, top=0.96,
                        hspace=0.20, wspace=0.20)
    plt.show()


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  50.101 seconds)


.. _sphx_glr_download_auto_examples_plot_neuroharmony_trained.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_neuroharmony_trained.py <plot_neuroharmony_trained.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_neuroharmony_trained.ipynb <plot_neuroharmony_trained.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
