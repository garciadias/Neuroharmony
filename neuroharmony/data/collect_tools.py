"""Collect tools."""
from pathlib import Path
from shutil import copyfile


def find_all_files_by_name(directory_path, file_pattern, depth=2):
    """
    Find all files in a folder.

    Parameters
    ----------
    directory_path: string
        Path for a giving folder.

    file_pattern: string
        File extension (.csv, .pdf, .txt, ...).

    depth: int
        Depth of the file search.

    Returns
    -------
    filelist: list
        List of paths to the found files in the format of pathlib.PosixPath.

    """
    filelist = []
    for level in range(depth):
        filelist.extend(list(Path(directory_path).glob('/'.join(level * ['*'] + [file_pattern]))))
    return filelist


def collect_datafile(filepath, root_path, local_path):
    """
    Collect a datafile.

    Parameters
    ----------
    filepath: string or pathlib.PosixPath
        Path for the file to be copied.

    root_path: string
        Path root to the origin of the data.

    local_path: string
        Path to the local folder you want to save the copied data.

    Returns
    -------
    file_exists: boolean
        Returns True if the files were copied correctly and False otherwise.

    """
    filepath = str(filepath)
    local_final_path = filepath.replace(root_path, local_path)
    Path(local_final_path).parent.mkdir(parents=True, exist_ok=True)
    copyfile(filepath, local_final_path)
    return Path(local_final_path).exists()


def collect_multiple_datafile(filepath_list, root_path, local_path):
    """
    Collect a list of datafiles.

    Parameters
    ----------
    filepath_list: list of strings or pathlib.PosixPath
        List of paths for the file to be copied.

    root_path: string
        Path root to the origin of the data.

    local_path: string
        Path to the local folder you want to save the copied data.

    Returns
    -------
    file_exists: boolean
        Returns True if the files were copied correctly and False otherwise.

    """
    for filepath in filepath_list:
        collect_datafile(filepath, root_path, local_path)


if __name__ == '__main__':
    from pandas import read_csv

    SERVER_ROOT = '/media/kcl_2/HDD/SynologyDrive'
    PARTICIPANT_ROOT = '%s/BIDS_data/' % SERVER_ROOT
    FREESURFER_ROOT = '%s/FreeSurfer_preprocessed/' % SERVER_ROOT
    QOALA_ROOT = '%s/Qoala/' % SERVER_ROOT
    MRIQC_ROOT = '%s/MRIQC/' % SERVER_ROOT
    Path('./data/processed').mkdir(exist_ok=True, parents=True)
    PARTICIPANTS_FILES = find_all_files_by_name(PARTICIPANT_ROOT, 'participants.tsv', depth=3)
    for file_path in PARTICIPANTS_FILES:
        df = read_csv(file_path, header=0, sep='\t')
        df['image_id'] = df[['participant_id', 'session_id', 'acq_id', 'run_id']].agg('_'.join, axis=1) + '_T1w'
        df.to_csv(file_path, index=False, sep='\t')
    FSURFER_FILES = find_all_files_by_name(FREESURFER_ROOT, 'freesurferData.csv', depth=3)
    QOALA_FILES = find_all_files_by_name(QOALA_ROOT, 'Qoala*.csv', depth=3)
    MRIQC_GROUP_FILES = find_all_files_by_name(MRIQC_ROOT, 'group_T1w.tsv', depth=3)
    MRIQC_PRED_FILES = find_all_files_by_name(MRIQC_ROOT, '*pred.csv', depth=3)
    collect_multiple_datafile(PARTICIPANTS_FILES, PARTICIPANT_ROOT, './data/raw/')
    collect_multiple_datafile(FSURFER_FILES, FREESURFER_ROOT, './data/raw/')
    collect_multiple_datafile(QOALA_FILES, QOALA_ROOT, './data/raw/')
    collect_multiple_datafile(MRIQC_GROUP_FILES, MRIQC_ROOT, './data/raw/')
    collect_multiple_datafile(MRIQC_PRED_FILES, MRIQC_ROOT, './data/raw/')
