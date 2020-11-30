.. project-template documentation master file, created by
   sphinx-quickstart on Mon Jan 18 14:44:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _`BIDS validator`: https://bids-standard.github.io/bids-validator/
.. _`mri_data`: https://www.dropbox.com/s/kcbq0266bcab3bx/ds002936.zip
.. _BIDS: https://bids.neuroimaging.io/
.. _FreeSurfer: https://surfer.nmr.mgh.harvard.edu/
.. _recon-all: https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all
.. _MRIQC: https://mriqc.readthedocs.io/en/latest/
.. _`(Estebam, O. et al., 2017)`: https://doi.org/10.1371/journal.pone.0184661


Install
-------

.. code-block:: console

  $ pip install neuroharmony

Getting started
---------------

Neuroharmony is designed to work with FreeSurfer_ regional volumes measured from structural MRI. To generate this data you
need to run FreeSurfer and extract the volume values. This can be done using the function recon-all_ for all images.
The files need to be organized in the BIDS_ format (you can verify the folder using `BIDS validator`_). On the resulting
folders, you need to extract the volumes of the regions. We created an example of a dataset using the FreeSurfer_ Bert
subject data, that you can download at mri_data_. To run FreeSurfer on this dataset one could use the following
commands:

.. code-block:: console

    $ mkdir derivatives/freesurfer/ -p
    $ export SUBJECTS_DIR=$PWD/derivatives/freesurfer/
    $ for subject_img in $(find ./ -name 'sub*nii.gz')
    > do
    > recon-all -all -i $subject_img $(awk -F / '{ print $2; }' <<< $subject_img)
    > done

By default, FreeSurfer outputs the results to the folder defined as SUBJECTS_DIR. To combine the FreeSurfer outputs from
all images we could use the following commands:

.. code-block:: console

    $ list="$(ls sub* -d)"
    $ cd derivatives/freesurfer/
    $ aparcstats2table --subjects $list --hemi rh --meas volume --skip --tablefile rh_aparc_stats.txt
    $ aparcstats2table --subjects $list --hemi lh --meas volume --skip --tablefile lh_aparc_stats.txt
    $ asegstats2table --subjects $list --meas volume --skip --tablefile aseg_stats.txt

Neuroharmony also needs the results from the MRIQC_ tool, which is a tool for automatic quality checking of the raw
data `(Estebam, O. et al., 2017)`_. To run the docker version of MRIQC_ for all subjects in your BIDS_ folder,
Neuroharmony includes a wrapper. The following command will run MRIQC_ on all subjects in a BIDS_ folder.

.. code-block:: console

    $ mriqc-run ./ derivatives/mriqc/ -n_jobs 8

The command will find all subjects in the BIDS_ format on the current folder and save the MRIQC_ results at the
derivatives/mriqc/ folder using 8 cores. You can see the mriqc-run help with `mriqc-run -h`.

With these steps completed, you can use Neuroharmony to combine and harmonize the data, either by training your own
model or by using our pre-trained model.

Combine FreeSurfer_ and MRIQC_ data on the Neuroharmony format and apply the result on a trained model
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

.. code-block:: python

    import pandas as pd
    
    from neuroharmony import fetch_trained_model, combine_freesurfer, combine_mriqc

    mri_path = 'Path for your BIDS folder'
    freesurfer_data = combine_freesurfer(f'{mri_path}/derivatives/freesurfer/')
    participants_data = pd.read_csv(f'{mri_path}/participants.tsv', header=0, sep='\t', index_col=0)
    MRIQC = combine_mriqc(f'{mri_path}/derivatives/mriqc/')
    X = pd.merge(participants_data, MRIQC, left_on='participant_id', right_on='participant_id')

    neuroharmony = fetch_trained_model()
    x_harmonized = neuroharmony.transform(X)


Apply Neuroharmony pre-trained model
::::::::::::::::::::::::::::::::::::

.. code-block:: python

    from neuroharmony import fetch_trained_model, fetch_sample

    X = fetch_sample()
    neuroharmony = fetch_trained_model()
    x_harmonized = neuroharmony.transform(X)


Train your own model
::::::::::::::::::::

.. code-block:: python

    from neuroharmony import exclude_single_subject_groups, fetch_sample, Neuroharmony
    from neuroharmony.data.rois import rois
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
            RandomForestRegressor__n_estimators=[100, 200, 500],
            RandomForestRegressor__random_state=[42, 78],
            RandomForestRegressor__warm_start=[False, True],
        ),
        estimator_args=dict(n_jobs=1, random_state=42),
        randomized_search_args=dict(cv=5, n_jobs=8),
    )
    # Fit the model.
    x_train_harmonized = harmony.fit_transform(x_train)
