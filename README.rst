Neuroharmony: A tool for harmonizing volumetric MRI data from unseen scanners
=============================================================================

The model presented in `Garcia-Dias, et
al. (2020) <https://www.sciencedirect.com/science/article/pii/S1053811920306133>`__.

Documentation
-------------

`neuroharmony.readthedocs.io <https://neuroharmony.readthedocs.io>`__


Install Neuroharmony.
---------------------

::

   pip install neuroharmony

Introduction
------------

The increasing availability of magnetic resonance imaging (MRI) datasets is boosting the interest in the application
of machine learning in neuroimaging. A key challenge to the development of reliable machine learning models, and
their translational implementation in real-word clinical practice, is the integration of datasets collected using
different scanners. Current approaches for harmonizing multi-scanner data, such as the ComBat method, require a
statistical representative sample, and therefore are not suitable for machine learning models aimed at clinical
translation where the focus is on the assessment of individual scans from previously unseen scanners. To overcome
this challenge, Neuroharmony uses image quality metrics (i.e. intrinsic characteristics which can be extracted
from individual images without requiring a statistical representative sample and any extra information about the
scanners) to harmonize single images from unseen/unknown scanners based on.

 .. image:: docs/_static/article.png
   :width: 700
   :target: https://doi.org/10.1016/j.neuroimage.2020.117127
   :alt: Front pafe of the article "Neuroharmony: A new tool for harmonizing volumetric MRI data from unseen scanners".
