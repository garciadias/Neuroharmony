.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_plot_data_in_the_neuroharmony_format.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_plot_data_in_the_neuroharmony_format.py:


=============================
Prepare data for Neuroharmony
=============================

Prepare dataset in the Neuroharmony format.



.. image:: /auto_examples/images/sphx_glr_plot_data_in_the_neuroharmony_format_001.png
    :alt: plot data in the neuroharmony format
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

      0%|          | 0.00/60.3M [00:00<?, ?iB/s]      1%|1         | 697k/60.3M [00:00<00:08, 6.88MiB/s]      3%|3         | 1.99M/60.3M [00:00<00:07, 8.01MiB/s]      5%|5         | 3.25M/60.3M [00:00<00:06, 8.98MiB/s]      7%|7         | 4.29M/60.3M [00:00<00:05, 9.35MiB/s]      9%|8         | 5.20M/60.3M [00:00<00:05, 9.28MiB/s]     11%|#         | 6.58M/60.3M [00:00<00:05, 10.2MiB/s]     13%|#3        | 8.02M/60.3M [00:00<00:04, 11.2MiB/s]     15%|#5        | 9.33M/60.3M [00:00<00:04, 11.6MiB/s]     18%|#7        | 10.6M/60.3M [00:00<00:04, 11.8MiB/s]     20%|#9        | 11.8M/60.3M [00:01<00:04, 12.0MiB/s]     22%|##1       | 13.0M/60.3M [00:01<00:03, 12.0MiB/s]     24%|##3       | 14.4M/60.3M [00:01<00:03, 12.5MiB/s]     26%|##6       | 15.8M/60.3M [00:01<00:03, 12.9MiB/s]     29%|##8       | 17.2M/60.3M [00:01<00:03, 13.1MiB/s]     31%|###       | 18.5M/60.3M [00:01<00:03, 12.8MiB/s]     33%|###2      | 19.8M/60.3M [00:01<00:03, 11.9MiB/s]     35%|###4      | 21.0M/60.3M [00:01<00:03, 11.8MiB/s]     37%|###6      | 22.2M/60.3M [00:01<00:03, 11.4MiB/s]     39%|###8      | 23.4M/60.3M [00:01<00:03, 11.4MiB/s]     41%|####      | 24.6M/60.3M [00:02<00:03, 10.7MiB/s]     43%|####2     | 25.7M/60.3M [00:02<00:03, 10.4MiB/s]     44%|####4     | 26.7M/60.3M [00:02<00:03, 9.85MiB/s]     46%|####5     | 27.7M/60.3M [00:02<00:03, 9.87MiB/s]     48%|####7     | 28.7M/60.3M [00:02<00:03, 9.52MiB/s]     49%|####9     | 29.7M/60.3M [00:02<00:03, 9.35MiB/s]     51%|#####     | 30.6M/60.3M [00:02<00:03, 8.61MiB/s]     52%|#####2    | 31.5M/60.3M [00:02<00:03, 8.16MiB/s]     54%|#####3    | 32.4M/60.3M [00:02<00:03, 8.28MiB/s]     55%|#####5    | 33.2M/60.3M [00:03<00:03, 8.01MiB/s]     57%|#####6    | 34.2M/60.3M [00:03<00:03, 8.47MiB/s]     59%|#####8    | 35.3M/60.3M [00:03<00:02, 9.16MiB/s]     61%|######    | 36.8M/60.3M [00:03<00:02, 10.3MiB/s]     63%|######2   | 38.0M/60.3M [00:03<00:02, 10.8MiB/s]     65%|######5   | 39.3M/60.3M [00:03<00:01, 11.2MiB/s]     67%|######7   | 40.4M/60.3M [00:03<00:01, 10.5MiB/s]     69%|######9   | 41.6M/60.3M [00:03<00:01, 10.8MiB/s]     71%|#######   | 42.7M/60.3M [00:03<00:01, 9.85MiB/s]     73%|#######2  | 43.7M/60.3M [00:04<00:01, 9.96MiB/s]     74%|#######4  | 44.9M/60.3M [00:04<00:01, 10.3MiB/s]     76%|#######6  | 45.9M/60.3M [00:04<00:01, 10.0MiB/s]     78%|#######7  | 47.0M/60.3M [00:04<00:01, 10.1MiB/s]     80%|#######9  | 48.0M/60.3M [00:04<00:01, 9.15MiB/s]     81%|########1 | 49.0M/60.3M [00:04<00:01, 9.35MiB/s]     83%|########2 | 50.0M/60.3M [00:04<00:01, 9.14MiB/s]     84%|########4 | 50.9M/60.3M [00:04<00:01, 8.88MiB/s]     86%|########5 | 51.8M/60.3M [00:04<00:00, 8.85MiB/s]     88%|########7 | 52.9M/60.3M [00:05<00:00, 9.54MiB/s]     90%|########9 | 54.1M/60.3M [00:05<00:00, 10.0MiB/s]     92%|#########1| 55.3M/60.3M [00:05<00:00, 10.6MiB/s]     94%|#########4| 56.7M/60.3M [00:05<00:00, 11.4MiB/s]     96%|#########6| 58.1M/60.3M [00:05<00:00, 12.0MiB/s]     98%|#########8| 59.3M/60.3M [00:05<00:00, 9.89MiB/s]    100%|##########| 60.3M/60.3M [00:05<00:00, 10.5MiB/s]
    | participant_id   |   Diagn |   Gender |   Age |      cjv |     cnr |
    |:-----------------|--------:|---------:|------:|---------:|--------:|
    | sub-0001         |       1 |        1 |    21 | 0.32076  | 4.04011 |
    | sub-0002         |       1 |        0 |    40 | 0.352103 | 3.66634 |
    | sub-0003         |       1 |        0 |    21 | 0.364292 | 3.50784 |

    <matplotlib.axes._subplots.AxesSubplot object at 0x7f04e7469310>





|


.. code-block:: default

    from neuroharmony import fetch_mri_data, combine_freesurfer, combine_mriqc
    import pandas as pd

    mri_path = fetch_mri_data()
    freesurfer_data = combine_freesurfer(f"{mri_path}/derivatives/freesurfer/")
    participants_data = pd.read_csv(f"{mri_path}/participants.tsv", header=0, sep="\t", index_col=0)
    MRIQC = combine_mriqc(f"{mri_path}/derivatives/mriqc/")
    X = pd.merge(participants_data, MRIQC, left_on="participant_id", right_on="participant_id")
    print(X[X.columns[:5]].to_markdown())

    X.plot.scatter(x="prob_y", y="snr_total")


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  8.641 seconds)


.. _sphx_glr_download_auto_examples_plot_data_in_the_neuroharmony_format.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_data_in_the_neuroharmony_format.py <plot_data_in_the_neuroharmony_format.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_data_in_the_neuroharmony_format.ipynb <plot_data_in_the_neuroharmony_format.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
