"""
=============================
Prepare data for Neuroharmony
=============================

Prepare dataset in the Neuroharmony format.
"""
from neuroharmony import fetch_mri_data, combine_freesurfer, combine_mriqc
import pandas as pd

mri_path = fetch_mri_data()
freesurfer_data = combine_freesurfer(f"{mri_path}/derivatives/freesurfer/")
participants_data = pd.read_csv(f"{mri_path}/participants.tsv", header=0, sep="\t", index_col=0)
MRIQC = combine_mriqc(f"{mri_path}/derivatives/mriqc/")
X = pd.merge(participants_data, MRIQC, left_on="participant_id", right_on="participant_id")
print(X[X.columns[:5]].to_markdown())

X.plot.scatter(x="prob_y", y="snr_total")
