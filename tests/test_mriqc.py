"""Test tools for data finding and collection."""
from pathlib import Path

from pandas.core.generic import NDFrame

from neuroharmony.models.mriqc import run_mriqc
from neuroharmony.data.collect_tools import fetch_mri_data


def test_mriqc_runs_without_errors():
    """Test mriqc run without erros and output file is created."""
    mri_path = fetch_mri_data()
    subj_dir = mri_path
    out_dir = f"{mri_path}/mriqc/"
    if not Path(f'{out_dir}/mri/group_T1w.tsv').exists():
        IQMs = run_mriqc(subj_dir, out_dir, n_jobs=5)
        assert isinstance(IQMs, NDFrame)
    assert Path(f'{out_dir}/mri/group_T1w.tsv').exists()
