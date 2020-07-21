"""Run MQIQC."""
import argparse
from os import system, popen
from pathlib import Path
import json

from pandas import read_csv

description_template = dict(
    BIDSVersion="1.0.0",
    License="",
    Name="MRI dataset",
    ReferencesAndLinks=["https://github.com/garciadias/Neuroharmony"],
)


def _run_MRIQC_on_dir(subj_dir, sub_out, n_jobs=5):
    """Run MQIQC."""
    system("docker pull poldracklab/mriqc:0.15.2rc1")
    docker_command = (
        f"docker run -it --rm -v {subj_dir}/:/data:ro -v {sub_out}/:/out "
        "poldracklab/mriqc:0.15.2rc1 /data /out participant --participant_label"
    )
    subject_list = _get_subject_list(subj_dir, sub_out)
    for i in range(len(subject_list) // n_jobs + 1):
        run_str = docker_command
        for subj in subject_list[(i * n_jobs): ((i + 1) * n_jobs)]:
            run_str += " " + subj
        system(run_str + (f" --n_procs {n_jobs} --no-sub"))


def _get_subject_list(subj_dir, sub_out):
    """Get subjects."""
    subj_list = [subj.name.split("-")[1] for subj in subj_dir.glob("sub-*")]
    subj_done = [subj.name.split("-")[1] for subj in sub_out.glob("sub-*") if subj.is_dir()]
    return [subj for subj in subj_list if subj not in subj_done]


def _verify_docker_is_installed():
    if len(popen("command -v docker").read()) == 0:
        raise EnvironmentError("Docker is not installed. Get docker at https://docs.docker.com/get-docker/")


def run_mriqc(subj_dir, out_dir, n_jobs=1):
    """Run MRIQC.

    .. _MRIQC: https://github.com/poldracklab/mriqc
    .. _`MRIQC documentation`: https://github.com/poldracklab/mriqc
    .. _`docker`: https://docs.docker.com/get-docker/
    This function is a wrapper to run the MRIQC_ tool on a docker_ container. It requires docker to be installed. MRIQC_
    extracts no-reference IQMs (image quality metrics) from structural (T1w and T2w) and functional MRI (magnetic
    resonance imaging) data. MRIQC is an open-source project. More information can be found at `MRIQC documentation`_.

    Parameters
    ----------
    subj_dir: str
     Path to BIDS directory.
    out_dir: str
     Path to the output of MRIQC analysis.
    n_jobs: int
     Number of cpu used to run MRIQC.

    Returns
    -------
    IQMs: NDFrame of shape [n_subjects, 68]
     Dataframe with the IQMs for each subject.
    """
    _verify_docker_is_installed()
    subj_dir = Path(subj_dir)
    Path(f"{out_dir}/{subj_dir.name}").mkdir(exist_ok=True, parents=True)
    sub_out = Path(f"{out_dir}/{subj_dir.name}/").absolute()
    if not Path(f"{subj_dir}/dataset_description.json").exists():
        with open(f"{subj_dir}/dataset_description.json", "w") as description_file:
            json.dump(description_template, description_file)
    _run_MRIQC_on_dir(subj_dir.absolute(), sub_out, n_jobs=n_jobs)
    get_group_cmd = (
        f"docker run -it --rm -v {sub_out}/:/data:ro -v {sub_out}/:/out poldracklab/mriqc:0.15.2rc1" " /data /out group"
    )
    get_pred_cmd = (
        f"docker run -v {sub_out}:/scratch -w /scratch --entrypoint=mriqc_clf poldracklab/mriqc:0.15.2rc1"
        " --load-classifier -X group_T1w.tsv"
    )
    system(get_group_cmd)
    system(get_pred_cmd)
    return read_csv(f"{out_dir}/group_T1w.tsv", delimiter="\t", index_col=0)


def init_argparse():
    description = (
        "A Neuroharmony wrapper to run the docker version of the MRI quality control tool by"
        "Esteban et al., (2017). MRIQC: Advancing the automatic prediction of image quality in MRI from unseen sites."
        " https://github.com/poldracklab/mriqc"
    )

    parser = argparse.ArgumentParser(usage="mriqc-run [subj_dir] [out_dir] [OPTIONS]", description=description)

    parser.add_argument("-v", "--version", action="version", version="mriqc-run version 1.0.0")

    parser.add_argument("subj_dir", action="store", type=str, help="Path to BIDS directory.")
    parser.add_argument("out_dir", action="store", type=str, help="Path to the output of MRIQC analysis.")
    parser.add_argument(
        "-n_jobs",
        action="store",
        default=1,
        type=int,
        help="Number of cpu used in processing the images. Default is 1.",
    )

    return parser


def main():
    parser = init_argparse()
    args = parser.parse_args()
    if popen("command -v docker").read():
        run_mriqc(args.subj_dir, args.out_dir, args.n_jobs)
    else:
        raise OSError("Docker is not installed. Get docker at https://docs.docker.com/get-docker/")
