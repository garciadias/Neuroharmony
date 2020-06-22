"""Test tools for data finding and collection."""
from pathlib import Path

from neuroharmony.models.mriqc import run_mriqc


def test_mriqc_runs_without_errors():
    """Test mriqc run without erros and output file is created."""
    subj_dir = "data/test_bids/sample/"
    out_dir = "data/test_bids/mriqc/"
    if not Path('data/test_bids/mriqc/sample/group_T1w.tsv').exists():
        run_mriqc(subj_dir, out_dir, n_jobs=5)
    assert Path('data/test_bids/mriqc/sample/group_T1w.tsv').exists()
