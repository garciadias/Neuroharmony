
from pathlib import Path
import warnings

import pandas as pd

from src.data.collect_tools import find_all_files_by_name
from src.data.rois import rois


class Site(object):
    """Site class."""

    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.name = dir_path.name
        self._get_scanners()

    def _get_scanners(self):
        subdirs = [subpath.name for subpath in self.dir_path.glob('*') if subpath.is_dir()]
        if len(subdirs) == 0:
            self.scanner_list = []
            self._get_files()
            self._load_files()
            self._combine_files()
            self.combined['scanner'] = 'SCANNER01'
        else:
            self.scanner_list = subdirs
            for scanner_name in subdirs:
                scanner = Scanner(self.dir_path / Path(scanner_name))
                setattr(self, scanner_name, scanner)
                scanner._get_files()

    def _files_exists(self, directory_path, file_pattern):
        file_search = find_all_files_by_name(directory_path, file_pattern)
        if len(file_search) == 0:
            return False
        elif len(file_search) == 1:
            return file_search[0]
        else:
            return file_search
            warnings.warn('There are more than one %s in this site.' % file_pattern)

    def _get_files(self):
        self.freesurferData_path = self._files_exists(self.dir_path, 'freesurferData.csv')
        self.participants_path = self._files_exists(self.dir_path, 'participants.tsv')
        self.iqm_path = self._files_exists(self.dir_path, 'group_T1w.tsv')
        self.pred_path = self._files_exists(self.dir_path, 'mclf*csv')
        self.qoala_path = self._files_exists(self.dir_path, 'Qoala*.csv')

    def _load_files(self):
        self.freesurferData = pd.read_csv(self.freesurferData_path, header=0)
        self.participants = pd.read_csv(self.participants_path, header=0, sep='\t')
        self.iqm = pd.read_csv(self.iqm_path, header=0, sep='\t')
        self.pred = pd.read_csv(self.pred_path, header=0)
        self.qoala = pd.read_csv(self.qoala_path, header=0)

    def _combine_files(self):
        self.freesurferData['Participant_ID'] = self.freesurferData['Image_ID'].str.split("_").str[0]
        df = pd.merge(self.participants, self.freesurferData, on='Participant_ID', how='inner')
        df = df.dropna(how="any")
        df['subject_id'] = df['Image_ID'].str.replace('_T1w/', '').str.replace('sub-', '')
        self.pred['subject_id'] = self.pred['subject_id'].astype('str')
        df = pd.merge(df, self.pred, on='subject_id', how='inner')
        df['bids_name'] = df['Image_ID'].str.replace('/', '')
        df = pd.merge(df, self.iqm, on='bids_name', how='inner')
        x = df[rois].astype('float32').divide(df['EstimatedTotalIntraCranialVol'], axis=0)
        x['tiv'] = df['EstimatedTotalIntraCranialVol'].values.astype('float32')
        extra_data = list(df.columns)
        redundant_names = ['Image_ID', 'bids_name', 'EstimatedTotalIntraCranialVol', 'Handedness']
        for var in rois + redundant_names:
            try:
                extra_data.remove(var)
            except ValueError:
                pass
        for var in extra_data:
            x[var] = df[var]
        self.combined = x

    def _combine_all_scanners(self):
        for scanner_name in self.scanner_list:
            getattr(self, scanner_name)._load_files()
            getattr(self, scanner_name)._combine_files()
            getattr(self, scanner_name).combined['scanner'] = scanner_name
        self.combined = pd.concat([getattr(self, scanner_name).combined for scanner_name in self.scanner_list])


class Scanner(Site):
    """Scanners class."""

    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.name = dir_path.name


if __name__ == '__main__':
    # SERVER_ROOT = '/run/user/1000/gvfs/smb-share:server=kc-deeplab.local,share=deeplearning'
    # PARTICIPANT_ROOT = '%s/BIDS_data/' % SERVER_ROOT
    # FREESURFER_ROOT = '%s/FreeSurfer_preprocessed/' % SERVER_ROOT
    # QOALA_ROOT = '%s/Qoala/' % SERVER_ROOT
    # MRIQC_ROOT = '%s/MRIQC/' % SERVER_ROOT
    pass
