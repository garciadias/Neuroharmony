
from pathlib import Path
from shutil import copyfile


def find_all_files_by_name(directory_path, file_pattern, depth=2):
    filelist = []
    for level in range(depth):
        filelist.extend(list(Path(directory_path).glob('/'.join(level * ['*'] + [file_pattern]))))
    return filelist


def collect_datafile(filepath, root_path, local_path):
    filepath = str(filepath)
    local_final_path = filepath.replace(root_path, local_path)
    Path(local_final_path).parent.mkdir(parents=True, exist_ok=True)
    copyfile(filepath, local_final_path)
    return Path(local_final_path).exists()


def collect_multiple_datafile(filepath_list, root_path, local_path):
    for filepath in filepath_list:
        collect_datafile(filepath, root_path, local_path)


if __name__ == '__main__':
    SERVER_ROOT = '/run/user/1000/gvfs/smb-share:server=kc-deeplab.local,share=deeplearning'
    PARTICIPANT_ROOT = '%s/BIDS_data/' % SERVER_ROOT
    FREESURFER_ROOT = '%s/FreeSurfer_preprocessed/' % SERVER_ROOT
    QOALA_ROOT = '%s/Qoala/' % SERVER_ROOT
    MRIQC_ROOT = '%s/MRIQC/' % SERVER_ROOT
    Path('./data/processed').mkdir(exist_ok=True, parents=True)
    PARTICIPANTS_FILES = find_all_files_by_name(PARTICIPANT_ROOT, 'participants.tsv', depth=3)
    FSURFER_FILES = find_all_files_by_name(FREESURFER_ROOT, 'freesurferData.csv', depth=3)
    QOALA_FILES = find_all_files_by_name(QOALA_ROOT, 'Qoala*.csv', depth=3)
    MRIQC_GROUP_FILES = find_all_files_by_name(MRIQC_ROOT, 'group_T1w.tsv', depth=3)
    MRIQC_PRED_FILES = find_all_files_by_name(MRIQC_ROOT, '*pred.csv', depth=3)
    collect_multiple_datafile(PARTICIPANTS_FILES, PARTICIPANT_ROOT, './data/raw/')
    collect_multiple_datafile(FSURFER_FILES, FREESURFER_ROOT, './data/raw/')
    collect_multiple_datafile(QOALA_FILES, QOALA_ROOT, './data/raw/')
    collect_multiple_datafile(MRIQC_GROUP_FILES, MRIQC_ROOT, './data/raw/')
    collect_multiple_datafile(MRIQC_PRED_FILES, MRIQC_ROOT, './data/raw/')
