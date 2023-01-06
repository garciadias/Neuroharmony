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
.. _`(Esteban, O. et al., 2017)`: https://doi.org/10.1371/journal.pone.0184661
.. _`10.1016/j.neuroimage.2020.117127` : https://www.sciencedirect.com/science/article/pii/S1053811920306133?via%3Dihub
.. _ComBat : https://github.com/Jfortin1/ComBatHarmonization
.. _`Qoala-T` : https://github.com/Qoala-T/QC


Install
-------

.. code-block:: console

  $ pip install neuroharmony

  Getting started
  ---------------

  Before you can run Neuroharmony on your structural MRI data, you need to use FreeSurfer_ to extract the regional 
  volumes and MRIQC_ to extract image quality metrics (IQMs).


  Run FreeSurfer_ for Neuroharmony
  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  To be able to run FreeSurfer, your MRI data files need to be organized in the Brain Imaging Data Structure 
  (BIDS)_ format. You can verify the folder structure using `BIDS validator`_. You can then run FreeSurfer on the resulting
  folders to extract the volumes of the regions. This can be done using the function recon-all_ for all images.

  We created an example of a dataset using the FreeSurfer_ Bert
  subject data that can be downloaded at mri_data_. To run FreeSurfer on this dataset, you could use the following
  commands:

  .. code-block:: console

      $ mkdir derivatives/freesurfer/ -p
      $ export SUBJECTS_DIR=$PWD/derivatives/freesurfer/
      $ for subject_img in $(find ./ -name 'sub*nii.gz')
      > do
      > recon-all -all -i $subject_img $(awk -F / '{ print $2; }' <<< $subject_img)
      > done

  By default, FreeSurfer outputs the results to the folder defined as SUBJECTS_DIR. To combine the FreeSurfer outputs
  from all images, you could use the following commands:

  .. code-block:: console

      $ list="$(ls sub* -d)"
      $ cd derivatives/freesurfer/
      $ aparcstats2table --subjects $list --hemi rh --meas volume --skip --tablefile rh_aparc_stats.txt
      $ aparcstats2table --subjects $list --hemi lh --meas volume --skip --tablefile lh_aparc_stats.txt
      $ asegstats2table --subjects $list --meas volume --skip --tablefile aseg_stats.txt


  Run MRIQC_ for Neuroharmony
  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  Neuroharmony also needs the results from the MRIQC_ tool, which is a tool for automatic quality checking of the raw
  data `(Esteban, O. et al., 2017)`_. To run the docker version of MRIQC_ for all subjects in your BIDS_ folder,
  Neuroharmony includes a wrapper. The following command will run MRIQC_ on all subjects in a BIDS_ folder.

  .. code-block:: console

      $ mriqc-run ./ derivatives/mriqc/ -n_jobs 8

  The command will find all subjects in the BIDS_ format on the current folder and save the MRIQC_ results at the
  derivatives/mriqc/ folder using 8 cores. You can see the mriqc-run help with `mriqc-run -h`.

  With these steps completed, you can use Neuroharmony to combine and harmonize the data, either by training your own
  model or by using our pre-trained model.


  Combine FreeSurfer_ and MRIQC_ data in the Neuroharmony format and apply the result on a trained model
  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  .. code-block:: python

      import pandas as pd

      from neuroharmony import fetch_trained_model, combine_freesurfer, combine_mriqc

      mri_path = 'Path to your BIDS folder'
      freesurfer_data = combine_freesurfer(f'{mri_path}/derivatives/freesurfer/')
      participants_data = pd.read_csv(f'{mri_path}/participants.tsv', header=0, sep='\t', index_col=0)
      MRIQC = combine_mriqc(f'{mri_path}/derivatives/mriqc/')
      X = pd.merge(participants_data, MRIQC, left_on='participant_id', right_on='participant_id')

      neuroharmony = fetch_trained_model()
      x_harmonized = neuroharmony.transform(X)


  Apply a pre-trained Neuroharmony model
  ::::::::::::::::::::::::::::::::::::

  .. code-block:: python

      from neuroharmony import fetch_trained_model, fetch_sample

      X = fetch_sample()
      neuroharmony = fetch_trained_model()
      x_harmonized = neuroharmony.transform(X)


  Train your own Neuroharmony model
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

      # Create the Neuroharmony model.
      # Here you can establish the range of the hyperparameters via random search or give specific values.
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



  Citation
  ---------------
  If you use Neuroharmony, please cite the following publication:
  Garcia-Dias R, et al. 'Neuroharmony: A new tool for harmonizing volumetric MRI data from unseen scanners.'
  Neuroimage. 2020 Oct 15;220:117127. doi: `10.1016/j.neuroimage.2020.117127`.


  FAQs
  ---------------

  What can I use Neuroharmony for?
  ::::::::::::::::::::::::::::::::::::

  Neuroharmony is a data harmonization tool for structural Magnetic Resonance Imaging (MRI) data. Data in multi-site
  research studies are affected by the use of different MRI scanners and acquisition protocols, which may reduce the
  comparability of data between sites. ﻿In particular, inconsistencies can arise from the MRI machine field strength, 
  head motion, gradient non-linearity, time-of-day, among others.

  Data harmonization consists of performing calibration corrections to data from different sources with the aim of
  making their integration and comparison more meaningful. The aim of the harmonization process is not necessarily to
  approximate a ground truth (i.e. the real volume of brain regions) but to make the integration and comparison of 
  data collected using multiple scanners more reliable. Therefore, harmonization does not eliminate possible 
  systematic bias but it guarantees that the distortion affects all data points in the same way.

  The main advantage of Neuroharmony is that it requires no prior knowledge about the way an MRI scan was acquired,
  so it can be applied to single MRI scans.


  How does Neuroharmony work?
  ::::::::::::::::::::::::::::::::::::

  Neuroharmony is a machine learning model that captures the relationship between image quality metrics (IQMs)
  from MRIQC_ (Esteban et al. 2017, 2019) and the relative volume corrections for each brain region
  from ComBat_ (Fortin et al. 2017, 2018) for structural MRI data.

  ComBat_ is a data harmonization tool that uses Bayesian regression to find systematic differences among multiple data
  collected using different scanners. The tool performs additive and multiplicative corrections to produce distortions
  that eliminate these systematic differences from the data. The main limitation of this approach is the need for a
  sample size that guarantees a statistically representative sample from each scanner included in the study.

  The ComBat_ tool performs the harmonization based on a given covariate while conserving the variance due to other 
  covariates of interest. To account for the individual contribution of the different covariates, 
  Neuroharmony applies several ComBat instances in a stepwise manner: first to remove sex-related effects, 
  then age-related effects, and finally scanner bias.

  Neuroharmony uses FreeSurfer_ regional volumes, MRIQC_ IQMs and basic demographic information (age, sex) to predict
  the ComBat_ corrections for an individual MRI scan.


  What kind of input data does Neuroharmony require?
  ::::::::::::::::::::::::::::::::::::

  Neuroharmony uses 101 regional volume measures from FreeSurfer_ , 68 image quality metrics (IQMs) from MRIQC_ ,
  and basic demographic information (age, sex) as input.

  ﻿The 101 FreeSurfer regions were extracted based on the Desikan-Killiany atlas (Desikan et al., 2006) and on the ASEG
  atlas (Fischl et al., 2002). Within Neuroharmony, the regional volumes are normalised by the total intracranial volume.


  What FreeSurfer regions are used by Neuroharmony?
  ::::::::::::::::::::::::::::::::::::

  The 101 included FreeSurfer regions are the following: brain stem, cerebrospinal fluid, corpus callosum anterior,
  corpus callosum central, corpus callosum mid-anterior, corpus callosum mid-posterior, corpus callosum posterior,
  third ventricle, fourth ventricle, left/right amygdala, left/right banks of the superior temporal sulcus, 
  left/right caudal anterior cingulate cortex, left/right caudal middle frontal gyrus, left/right caudate, 
  left/right cerebellum cortex, left/right cerebellum white matter, left/right cuneus cortex, left/right entorhinal 
  cortex, left/right frontal pole, left/right fusiform gyrus, left/right hippocampus, left/right inferior lateral 
  ventricle, left/right inferior parietal cortex, left/right inferior temporal gyrus, left/right insula, left/right 
  isthmus-cingulate cortex, left/right lateral occipital cortex, left/right lateral orbitofrontal, left/right lateral 
  ventricle, left/right lingual gyrus, left/right medial orbital frontal cortex, left/right middle temporal gyrus, 
  left/right nucleus accumbens, left/right pallidum, left/right paracentral lobule, left/right parahippocampal gyrus, 
  left/right pars opercularis, left/right pars orbitalis, left/right pars triangularis, left/right pericalcarine, 
  left/right postcentral gyrus, left/right posterior cingulate cortex, left/right precentral gyrus, left/right precuneus
  cortex, left/right putamen, left/right rostral anterior cingulate cortex, left/right rostral middle frontal gyrus, 
  left/right superior frontal gyrus, left/right superior parietal cortex, left/right superior temporal gyrus, left/right
  supramarginal gyrus, left/right temporal pole, left/right thalamus proper, left/right transverse temporal cortex, and
  left/right ventral diencephalon.


  What are Image Quality Metrics (IQMs)?
  ::::::::::::::::::::::::::::::::::::

  Image Quality Metrics (IQMs) are intrinsic characteristics of an MRI scan, i.e. they are directly measurable from
  individual scans without requiring a statistically representative sample.

  The 68 IQMs used in Neuroharmony were developed by Esteban and colleagues (Esteban et al. 2017, 2019).
  These IQMs include, but are not limited to, contrast-to-noise ratio, signal-to-noise ratio, and the white-matter to
  maximum intensity ratio.


  What kind of machine learning model does Neuroharmony use?
  ::::::::::::::::::::::::::::::::::::

  The machine learning model in Neuroharmony is random forest regression using the scikit-learn_ python package
  (Buitinck et al., 2013; Pedregosa et al., 2011). Principal component analysis is applied to the data to reduce
  dimensionality before training the model using a leave-one-scanner-out cross-validation strategy for 
  hyperparameter tuning.


  What quality checks were implemented in the Neuroharmony development?
  ::::::::::::::::::::::::::::::::::::

  Two publicly available tools were used for automatic quality checking of included data, MRIQC_ for the raw data
  (Esteban et al. 2017, 2019) and `Qoala-T`_ for the FreeSurfer-preprocessed data (Klapwijk et al. 2019). 

  Additionally, we performed outlier checks within each scanner. A subject was considered an outlier if the relative
  volumes of at least 10 regions of interest (ROIs), corresponding to ~10% of the feature space, were more than 2.5
  standard deviations away from the sample mean. You can find more information in our publication.


  What makes Neuroharmony different from other harmonization approaches?
  ::::::::::::::::::::::::::::::::::::

  The main advantage of Neuroharmony is that it does not require a statistically representative sample from a
  scanner and/or acquisition protocol, so it can be applied to single MRI scans.


  Can I apply Neuroharmony to patient data?
  ::::::::::::::::::::::::::::::::::::

  The current version of Neuroharmony has only been evaluated on healthy subjects.


  When applying the Neuroharmony model, I am getting the error that subjects are out of range. What does this mean?
  ::::::::::::::::::::::::::::::::::::

  The warning message that subjects are out of range means that the IQM values for at least one of the subjects it is 
  applied to were not included in the training range of the Neuroharmony model. The tool will still harmonize the data
  for this subject, but it may be less effective than for those subjects whose IQM values fall within the training range.

  You can run 'Neuroharmony.subjects_out_of_range_', where 'Neuroharmony' is the model name, to see a list of the
  affected subjects.


  How long does it take to run Neuroharmony?
  ::::::::::::::::::::::::::::::::::::

  How long it takes to run Neuroharmony depends on the processing power of your computer and the number of subjects in 
  your dataset. The most time-consuming step is the FreeSurfer preprocessing of the images, which can take several 
  hours per subject. Once this is done, training your own model may also take several hours.


  How do I cite Neuroharmony?
  ::::::::::::::::::::::::::::::::::::

  If you use Neuroharmony, please cite the following publication:
  Garcia-Dias R, et al. 'Neuroharmony: A new tool for harmonizing volumetric MRI data from unseen scanners.'
  Neuroimage. 2020 Oct 15;220:117127. doi: `10.1016/j.neuroimage.2020.117127`_ .


  How can I contact the authors?
  ::::::::::::::::::::::::::::::::::::

  You can contact us at mlmh@kcl.ac.uk


  Acknowledgements
  ---------------

  This work has been supported by an Innovator Award from Wellcome (208519/Z/17/Z) and a research grant from the 
  Medical Research Council (MR/X005445/1) to Prof Andrea Mechelli.  



  References
  ---------------

  The below is a list of references used in this documentation.

  ﻿Buitinck, L., Louppe, G., Blondel, M., Pedregosa, F., Mueller, A., Grisel, O., et al., 2013. 
  API design for machine learning software: experiences from the scikit-learn project

  ﻿Desikan, R.S., Ségonne, F., Fischl, B., Quinn, B.T., Dickerson, B.C., Blacker, D., et al., ﻿2006. An automated
  labeling system for subdividing the human cerebral cortex on ﻿MRI scans into gyral based regions of interest.
  Neuroimage 31 (3), 968–980. https:// doi.org/10.1016/j.neuroimage.2006.01.021.

  Esteban, O., Birman, D., Schaer, M., Koyejo, O.O., Poldrack, R.A., Gorgolewski, K.J.,
  2017. MRIQC: advancing the automatic prediction of image quality in MRI from
  unseen sites. PloS One 12 (9). https://doi.org/10.1371/journal.pone.0184661 e0184661.

  Esteban, O., Blair, R.W., Nielson, D.M., Varada, J.C., Marrett, S., Thomas, A.G., et al.,
  2019. Crowdsourced MRI quality metrics and expert quality annotations for training
  of humans and machines. Sci. Data 6 (1), 30. https://doi.org/10.1038/s41597-019-0035-4.

  ﻿Fischl, B., Salat, D.H., Busa, E., Albert, M., Dieterich, M., Haselgrove, C., et al., 2002. 
  Whole brain segmentation: Automated labeling of neuroanatomical structures in the human brain. Neuron 33 (3), 341–355. 
  https://doi.org/10.1016/S0896-6273(02) 00569-X.

  Fortin, J.P., Cullen, N., Sheline, Y.I., Taylor, W.D., Aselcioglu, I., Cook, P.A., et al.,
  2018. Harmonization of cortical thickness measurements across scanners and sites.
  Neuroimage 167, 104–120. https://doi.org/10.1016/j.neuroimage.2017.11.024.

  Fortin, J.P., Parker, D., Tunç, B., Watanabe, T., Elliott, M.A., Ruparel, K., et al., 2017.
  Harmonization of multi-site diffusion tensor imaging data. Neuroimage 161,
  149–170. https://doi.org/10.1016/j.neuroimage.2017.08.047.

  Klapwijk, E.T., van de Kamp, F., van der Meulen, M., Peters, S., Wierenga, L.M., 2019.
  Qoala-T: a supervised-learning tool for quality control of FreeSurfer segmented MRI
  data. Neuroimage 189, 116–129. https://doi.org/10.1016/
  J.NEUROIMAGE.2019.01.014.

  ﻿Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., et al., 2011. 
  Scikit-learn: machine learning in Python. J. Mach. Learn. Res. 12 (Oct), 2825–2830. 
  Retrieved from. http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html.


