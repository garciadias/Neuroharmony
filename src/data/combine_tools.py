
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
        subdirs = [subpath.name for subpath in self.dir_path.glob('*')
                   if subpath.is_dir()]
        if len(subdirs) == 0:
            self.scanner_list = []
            self._get_files()
            if self._is_complete():
                self._load_files()
                self._combine_files()
                self.data['scanner'] = '%s-SCANNER01' % self.name
                self.data.index = self.data.Participant_ID + '-00'
                self.complete_scanners = ['%s-SCANNER01' % self.name]
            else:
                self.data = pd.DataFrame()
        else:
            self.scanner_list = subdirs
            for scanner_name in subdirs:
                scanner = Scanner(self.dir_path / Path(scanner_name))
                setattr(self, scanner_name, scanner)
                scanner._get_files()
            self._combine_all_scanners()
        self.data['site'] = self.name

    def _files_exists(self, directory_path, file_pattern):
        file_search = find_all_files_by_name(directory_path, file_pattern)
        if len(file_search) == 0:
            return False
        elif len(file_search) == 1:
            return file_search[0]
        else:
            return file_search
            warnings.warn('There are more than one %s file in this site.' % file_pattern)

    def _get_files(self):
        self.freesurferData_path = self._files_exists(self.dir_path, 'freesurferData.csv')
        self.participants_path = self._files_exists(self.dir_path, 'participants.tsv')
        self.iqm_path = self._files_exists(self.dir_path, 'group_T1w.tsv')
        self.pred_path = self._files_exists(self.dir_path, 'mclf*csv')
        self.qoala_path = self._files_exists(self.dir_path, 'Qoala*.csv')

    def _is_complete(self):
        return all([self.freesurferData_path,
                    self.participants_path,
                    self.iqm_path,
                    self.pred_path,
                    self.qoala_path, ])

    def _load_files(self):
        self.freesurferData = pd.read_csv(self.freesurferData_path, header=0)
        self.participants = pd.read_csv(self.participants_path, header=0, sep='\t')
        self.iqm = pd.read_csv(self.iqm_path, header=0, sep='\t')
        self.pred = pd.read_csv(self.pred_path, header=0)
        self.qoala = pd.read_csv(self.qoala_path, header=0)

    def _combine_files(self):
        participant_id = self.freesurferData['Image_ID'].str.split("_").str[0]
        self.freesurferData['Participant_ID'] = participant_id
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
        redundant_names = ['Image_ID',
                           'bids_name',
                           'EstimatedTotalIntraCranialVol',
                           'Handedness']
        for var in rois + redundant_names:
            try:
                extra_data.remove(var)
            except ValueError:
                pass
        for var in extra_data:
            x[var] = df[var]
        self.data = x

    def _combine_all_scanners(self):
        n_scanners = len(self.scanner_list)
        participant_id_format = '-%%0%dd' % len(str(n_scanners))
        for scanner_id, scanner_name in enumerate(self.scanner_list):
            if getattr(self, scanner_name)._is_complete():
                getattr(self, scanner_name)._load_files()
                getattr(self, scanner_name)._combine_files()
                scanner_data = getattr(self, scanner_name).data
                scanner_data['scanner'] = '%s-%s' % (self.name, scanner_name)
                id_appendix = participant_id_format % scanner_id
                scanner_data.index = scanner_data.Participant_ID + id_appendix
        self.complete_scanners = [scanner_name
                                  for scanner_name in self.scanner_list
                                  if getattr(self, scanner_name)._is_complete()]
        if len(self.complete_scanners) > 0:
            self.data = pd.concat([getattr(self, scanner).data
                                   for scanner in self.complete_scanners], sort=True)
        else:
            self.data = pd.DataFrame()


class Scanner(Site):
    """Scanners class."""

    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.name = dir_path.name


class DataSet(Site):
    """Dataset class."""

    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.name = 'All sites'
        self._get_sites()
        self._combine_all_sites()

    def _get_sites(self):
        self.site_paths = [subpath for subpath in self.dir_path.glob('*')
                           if subpath.is_dir()]
        self.sites = [site_path.name for site_path in self.site_paths]
        self.not_empty_sites = []
        for site_path in self.site_paths:
            site = Site(site_path)
            setattr(self, site.name, site)
            self.not_empty_sites.append(site.name)

    def _combine_all_sites(self):
        n_sites = len(self.not_empty_sites)
        participant_id_format = '-%%0%dd' % len(str(n_sites))
        self.data = []
        if n_sites > 0:
            for site_id, site_name in enumerate(self.not_empty_sites):
                id_appendix = participant_id_format % site_id
                site = getattr(self, site_name)
                site.data.index = site.data.index + id_appendix
                self.data.append(site.data)
            self.data = pd.concat(self.data, sort=True)
        else:
            self.data = False


if __name__ == '__main__':
    DATASET = DataSet(Path('data/raw/'))
    DATASET.data.to_csv('data/processed/dataset.csv')
