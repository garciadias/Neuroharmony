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

      0%|          | 0.00/60.3M [00:00<?, ?iB/s]      1%|          | 598k/60.3M [00:00<00:10, 5.93MiB/s]      3%|3         | 1.89M/60.3M [00:00<00:08, 7.09MiB/s]      5%|5         | 3.22M/60.3M [00:00<00:06, 8.24MiB/s]      7%|6         | 4.19M/60.3M [00:00<00:06, 8.58MiB/s]      9%|8         | 5.35M/60.3M [00:00<00:05, 9.31MiB/s]     11%|#1        | 6.68M/60.3M [00:00<00:05, 10.2MiB/s]     13%|#2        | 7.83M/60.3M [00:00<00:04, 10.6MiB/s]     15%|#4        | 8.91M/60.3M [00:00<00:05, 10.1MiB/s]     17%|#6        | 10.2M/60.3M [00:00<00:04, 10.8MiB/s]     19%|#8        | 11.3M/60.3M [00:01<00:04, 10.8MiB/s]     21%|##        | 12.6M/60.3M [00:01<00:04, 11.4MiB/s]     23%|##2       | 13.8M/60.3M [00:01<00:04, 11.1MiB/s]     25%|##4       | 14.9M/60.3M [00:01<00:04, 10.9MiB/s]     27%|##6       | 16.0M/60.3M [00:01<00:04, 10.6MiB/s]     29%|##8       | 17.3M/60.3M [00:01<00:03, 11.2MiB/s]     31%|###       | 18.5M/60.3M [00:01<00:03, 11.2MiB/s]     33%|###2      | 19.8M/60.3M [00:01<00:03, 11.8MiB/s]     35%|###5      | 21.2M/60.3M [00:01<00:03, 12.3MiB/s]     38%|###7      | 22.7M/60.3M [00:01<00:02, 12.8MiB/s]     40%|###9      | 23.9M/60.3M [00:02<00:03, 12.0MiB/s]     42%|####1     | 25.2M/60.3M [00:02<00:03, 11.0MiB/s]     44%|####3     | 26.3M/60.3M [00:02<00:03, 10.8MiB/s]     46%|####5     | 27.4M/60.3M [00:02<00:02, 11.0MiB/s]     48%|####7     | 28.7M/60.3M [00:02<00:02, 11.5MiB/s]     50%|####9     | 29.9M/60.3M [00:02<00:02, 10.6MiB/s]     52%|#####1    | 31.1M/60.3M [00:02<00:02, 10.9MiB/s]     54%|#####3    | 32.3M/60.3M [00:02<00:02, 11.3MiB/s]     56%|#####5    | 33.7M/60.3M [00:02<00:02, 11.8MiB/s]     58%|#####8    | 35.0M/60.3M [00:03<00:02, 12.2MiB/s]     60%|######    | 36.5M/60.3M [00:03<00:01, 12.8MiB/s]     63%|######2   | 37.8M/60.3M [00:03<00:01, 12.2MiB/s]     65%|######4   | 39.0M/60.3M [00:03<00:01, 10.7MiB/s]     67%|######6   | 40.1M/60.3M [00:03<00:01, 10.5MiB/s]     69%|######8   | 41.3M/60.3M [00:03<00:01, 10.9MiB/s]     71%|#######   | 42.5M/60.3M [00:03<00:01, 11.2MiB/s]     72%|#######2  | 43.7M/60.3M [00:03<00:01, 10.7MiB/s]     74%|#######4  | 44.8M/60.3M [00:03<00:01, 10.4MiB/s]     77%|#######6  | 46.2M/60.3M [00:04<00:01, 11.3MiB/s]     79%|#######8  | 47.4M/60.3M [00:04<00:01, 11.0MiB/s]     81%|########  | 48.6M/60.3M [00:04<00:01, 11.2MiB/s]     83%|########2 | 49.8M/60.3M [00:04<00:00, 11.5MiB/s]     85%|########4 | 50.9M/60.3M [00:04<00:00, 11.0MiB/s]     86%|########6 | 52.1M/60.3M [00:04<00:00, 11.0MiB/s]     88%|########8 | 53.2M/60.3M [00:04<00:00, 10.6MiB/s]     90%|######### | 54.3M/60.3M [00:04<00:00, 10.1MiB/s]     92%|#########2| 55.6M/60.3M [00:04<00:00, 10.8MiB/s]     95%|#########4| 57.0M/60.3M [00:05<00:00, 11.5MiB/s]     96%|#########6| 58.2M/60.3M [00:05<00:00, 11.5MiB/s]     98%|#########8| 59.3M/60.3M [00:05<00:00, 10.8MiB/s]    100%|##########| 60.3M/60.3M [00:05<00:00, 11.2MiB/s]
    | participant_id   |   Diagn |   Gender |   Age |      cjv |     cnr |
    |:-----------------|--------:|---------:|------:|---------:|--------:|
    | sub-0001         |       1 |        1 |    21 | 0.32076  | 4.04011 |
    | sub-0002         |       1 |        0 |    40 | 0.352103 | 3.66634 |
    | sub-0003         |       1 |        0 |    21 | 0.364292 | 3.50784 |

    <matplotlib.axes._subplots.AxesSubplot object at 0x7f3a66418510>





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

   **Total running time of the script:** ( 0 minutes  8.266 seconds)


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
