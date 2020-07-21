"""Collect tools."""
from pathlib import Path
from requests import get
from shutil import copyfile
from zipfile import ZipFile
import joblib
import os

from pandas import read_csv
from tqdm import tqdm


def _download(url, filepath):
    dirpath = Path(filepath).parent
    Path(dirpath).mkdir(exist_ok=True)
    headers = {"user-agent": "Wget/1.16 (linux-gnu)"}
    r = get(url, stream=True, headers=headers)
    total_size = int(r.headers.get("content-length", 0))
    block_size = 1024
    t = tqdm(total=total_size, unit="iB", unit_scale=True)
    with open(filepath, "wb") as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()


def fetch_mri_data():
    """Fetch example of MRI dataset.

    The dataset is a replication of the Bert subject released with the FreeSurfer software for testing.

    Returns
    =======
    mri_path: str
        The path for the MRI data.
    """
    script_path = os.path.dirname(os.path.abspath(__file__))
    filepath = f"{script_path}/../../data/mri.zip"
    unzip_folder = str(Path(filepath).parent) + "/mri/"
    if not Path(filepath).exists():
        _download("https://www.dropbox.com/s/kcbq0266bcab3bx/ds002936.zip", filepath)
        Path(unzip_folder).mkdir(exist_ok=True)
        zip_file = ZipFile(filepath, "r")
        zip_file.extractall(unzip_folder)
        zip_file.close()
        os.remove(filepath)
    return str(Path(unzip_folder).absolute())


def fetch_sample():
    """Fetch a sample of FreeSurfer derived volumes in the Neuroharmony format.

    Fetch the FreeSurfer derived volumes of some subjects in the
    `ADHD200 <http://fcon_1000.projects.nitrc.org/indi/adhd200/index.html>`_ and in
    the `PPMI <http://www.ppmi-info.org/>`_ datasets.

    Returns
    =======
    dataset: NDFrame of shape [n_subjects, n_features]
        DataFrame with data from ADHD200 and the PPMI subjects in the  Neuroharmony format.
    """
    script_path = os.path.dirname(os.path.abspath(__file__))
    filepath = f"{script_path}/../../data/test_sample.csv"
    _download("https://www.dropbox.com/s/mxcaqx2y29n09rp/test_sample.csv", filepath)
    return read_csv(filepath, index_col=0)


def fetch_trained_model():
    """Fetch Neuroharmony pre-trained model.

    Returns
    =======
    neuroharmony: Neuroharmony class
        Pre-trained Neuroharmony model.
    """
    script_path = os.path.dirname(os.path.abspath(__file__))
    filepath = f"{script_path}/../../data/neuroharmony.pkl.gz"
    if not Path(filepath).exists():
        _download("https://www.dropbox.com/s/s3521oqd3fpi0ll/neuroharmony.pkl.gz", filepath)
    try:
        return joblib.load(filepath)
    except KeyError:
        Path(filepath).unlinke()
        _download("https://www.dropbox.com/s/s3521oqd3fpi0ll/neuroharmony.pkl.gz", filepath)
        return joblib.load(filepath)


def find_all_files_by_name(directory_path, file_pattern, depth=2):
    """Find all files in a folder.

    Parameters
    ==========
    directory_path: string
        The path for a giving folder.

    file_pattern: string
        File extension (.csv, .pdf, .txt, ...).

    depth: int
        Depth of the file search.

    Returns
    =======
    filelist: list
        List of paths to the found files in the format of pathlib.PosixPath.
    """
    filelist = []
    for level in range(depth):
        filelist.extend(list(Path(directory_path).glob("/".join(level * ["*"] + [file_pattern]))))
    return filelist


def collect_datafile(filepath, root_path, local_path):
    """Collect a datafile.

    Parameters
    ==========
    filepath: string or pathlib.PosixPath
        The path for the file to be copied.

    root_path: string
        The path root to the origin of the data.

    local_path: string
        The path to the local folder you want to save the copied data.

    Returns
    =======
    file_exists: boolean
        Returns True if the files were copied correctly and False otherwise.
    """
    filepath = str(filepath)
    local_final_path = filepath.replace(root_path, local_path)
    Path(local_final_path).parent.mkdir(parents=True, exist_ok=True)
    copyfile(filepath, local_final_path)
    return Path(local_final_path).exists()


def collect_multiple_datafile(filepath_list, root_path, local_path):
    """Collect a list of datafiles.

    Parameters
    ==========
    filepath_list: list of strings or pathlib.PosixPath
        List of paths for the file to be copied.

    root_path: string
        The path root to the origin of the data.

    local_path: string
        The path to the local folder you want to save the copied data.

    Returns
    =======
    file_exists: boolean
        Returns True if the files were copied correctly and False otherwise.
    """
    for filepath in filepath_list:
        collect_datafile(filepath, root_path, local_path)


if __name__ == "__main__":
    SERVER_ROOT = "/media/kcl_2/HDD/SynologyDrive"
    PARTICIPANT_ROOT = "%s/BIDS_data/" % SERVER_ROOT
    FREESURFER_ROOT = "%s/FreeSurfer_preprocessed/" % SERVER_ROOT
    QOALA_ROOT = "%s/Qoala/" % SERVER_ROOT
    MRIQC_ROOT = "%s/MRIQC/" % SERVER_ROOT
    Path("./data/processed").mkdir(exist_ok=True, parents=True)
    PARTICIPANTS_FILES = find_all_files_by_name(PARTICIPANT_ROOT, "participants.tsv", depth=3)
    for file_path in PARTICIPANTS_FILES:
        df = read_csv(file_path, header=0, sep="\t")
        df["image_id"] = df[["participant_id", "session_id", "acq_id", "run_id"]].agg("_".join, axis=1) + "_T1w"
        df.to_csv(file_path, index=False, sep="\t")
    FSURFER_FILES = find_all_files_by_name(FREESURFER_ROOT, "freesurferData.csv", depth=3)
    QOALA_FILES = find_all_files_by_name(QOALA_ROOT, "Qoala*.csv", depth=3)
    MRIQC_GROUP_FILES = find_all_files_by_name(MRIQC_ROOT, "group_T1w.tsv", depth=3)
    MRIQC_PRED_FILES = find_all_files_by_name(MRIQC_ROOT, "*pred.csv", depth=3)
    collect_multiple_datafile(PARTICIPANTS_FILES, PARTICIPANT_ROOT, "./data/raw/")
    collect_multiple_datafile(FSURFER_FILES, FREESURFER_ROOT, "./data/raw/")
    collect_multiple_datafile(QOALA_FILES, QOALA_ROOT, "./data/raw/")
    collect_multiple_datafile(MRIQC_GROUP_FILES, MRIQC_ROOT, "./data/raw/")
    collect_multiple_datafile(MRIQC_PRED_FILES, MRIQC_ROOT, "./data/raw/")
