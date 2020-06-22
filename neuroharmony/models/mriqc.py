"""Run MQIQC."""

from os import system, popen
from pathlib import Path
import json

description_template = dict(
    BIDSVersion='1.0.0',
    License='',
    Name='MRI dataset',
    ReferencesAndLinks=['https://github.com/garciadias/Neuroharmony'],
)


def _run_MRIQC_on_dir(subj_dir, sub_out, n_jobs=5):
    """Run MQIQC."""
    system('docker pull poldracklab/mriqc:0.15.2rc1')
    docker_command = f'docker run -it --rm -v {subj_dir}/:/data:ro -v {sub_out}/:/out ' \
                     'poldracklab/mriqc:0.15.2rc1 /data /out participant --participant_label'
    subject_list = _get_subject_list(subj_dir, sub_out)
    for i in range(len(subject_list) // n_jobs + 1):
        run_str = docker_command
        for subj in subject_list[(i * n_jobs):((i + 1) * n_jobs)]:
            run_str += ' ' + subj
        system(run_str + (f' --n_procs {n_jobs} --no-sub'))


def _get_subject_list(subj_dir, sub_out):
    """Get subjects."""
    subj_list = [subj.name.split('-')[1] for subj in subj_dir.glob('sub-*')]
    subj_done = [subj.name.split('-')[1] for subj in sub_out.glob('sub-*') if subj.is_dir()]
    return [subj for subj in subj_list if subj not in subj_done]


def _verify_docker_is_installed():
    if len(popen('command -v docker').read()) == 0:
        raise EnvironmentError('Docker is not installed. Get docker at https://docs.docker.com/get-docker/')


def run_mriqc(subj_dir, out_dir, n_jobs=5):
    """Run MRIQC.

    Parameters
    ----------
    subj_dir: str
     Path to BIDS directory.
    out_dir: str
     Path to the output of MRIQC analysis.
    n_jobs: int
     Number of cpu used to run MRIQC.
    """
    _verify_docker_is_installed()
    subj_dir = Path(subj_dir)
    Path(f'{out_dir}/{subj_dir.name}').mkdir(exist_ok=True, parents=True)
    sub_out = Path(f'{out_dir}/{subj_dir.name}/').absolute()
    if not Path(f'{subj_dir}/dataset_description.json').exists():
        with open(f'{subj_dir}/dataset_description.json', 'w') as description_file:
            json.dump(description_template, description_file)
    _run_MRIQC_on_dir(subj_dir.absolute(), sub_out, n_jobs=n_jobs)
    get_group_cmd = f'docker run -it --rm -v {sub_out}/:/data:ro -v {sub_out}/:/out poldracklab/mriqc:0.15.2rc1' \
                    ' /data /out group'
    get_pred_cmd = f'docker run -v {sub_out}:/scratch -w /scratch --entrypoint=mriqc_clf poldracklab/mriqc:0.15.2rc1' \
                   '--load-classifier -X group_T1w.tsv'
    system(get_group_cmd)
    system(get_pred_cmd)
