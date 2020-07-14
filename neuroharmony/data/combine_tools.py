"""Combining tools."""
from pathlib import Path
import warnings

import pandas as pd

from neuroharmony.data.collect_tools import find_all_files_by_name
from neuroharmony.data.rois import rois


def _files_exists(directory_path, file_pattern):
    """Verify if a file exists and is the unique file of that kind in the folder."""
    file_search = find_all_files_by_name(directory_path, file_pattern)
    if len(file_search) == 0:
        return False
    elif len(file_search) == 1:
        return file_search[0]
    else:
        return file_search
        warnings.warn("There are more than one %s file in this site." % file_pattern)


def combine_freesurfer(freesurfer_path):
    """Combine aparc and aseg data files in a single csv files.

    It uses the list in columns_name.list file to select the relevant features in the freesurfer output.

    Parameters
    ----------
    freesurfer_path : string
        The path to the subject directories.

    Returns
    -------
    combined : NDFrame of shape [n_subjects, n_features]
        A DataFrame with the FreeSurfer volumetric data for each subject.
    """
    aseg_stats = pd.read_csv(f"{freesurfer_path}/aseg_stats.txt", delimiter="\t")
    lh_aparc_stats = pd.read_csv(f"{freesurfer_path}/lh_aparc_stats.txt", delimiter="\t")
    rh_aparc_stats = pd.read_csv(f"{freesurfer_path}/rh_aparc_stats.txt", delimiter="\t")

    combined = pd.merge(aseg_stats, lh_aparc_stats, left_on="Measure:volume", right_on="lh.aparc.volume")
    combined = pd.merge(combined, rh_aparc_stats, left_on="Measure:volume", right_on="rh.aparc.volume")
    combined.rename(columns={"Measure:volume": "participant_id"}, inplace=True)
    combined.participant_id = combined.participant_id.str.replace("/", "")
    combined = combined.set_index("participant_id")[rois]
    return combined


def combine_mriqc(mri_path):
    """Combine group_T1w and mclf files from the MRIQC run.

    It uses the list in columns_name.list file to select the relevant features in the freesurfer output.

    Parameters
    ----------
    mri_path: string
        The path to a BIDS directory.

    Returns
    -------
    combined : NDFrame of shape [n_subjects, n_features]
        A DataFrame with the MRIQC information for each subject.
    """
    iqm_path = _files_exists(mri_path, "group_T1w.tsv")
    pred_path = _files_exists(mri_path, "mclf*")
    if not iqm_path or not pred_path:
        raise FileNotFoundError(f"MRIQC files not found in {mri_path}")
    iqm = pd.read_csv(iqm_path, header=0, sep="\t")
    pred = pd.read_csv(pred_path, header=0)
    pred["bids_name"] = "sub-" + pred.subject_id + "_T1w"
    pred.drop(columns="subject_id", inplace=True)
    combined = pd.merge(iqm, pred, left_on="bids_name", right_on="bids_name")
    combined.rename(columns={"bids_name": "participant_id"}, inplace=True)
    combined.participant_id = [subject_line[0] for subject_line in combined.participant_id.str.split("_")]
    combined = combined.set_index("participant_id")
    return combined


class Site(object):
    """A class for site definition.

    Tools for collecting and combining data from a site given a path to the site data.

    Parameters
    ----------
    dir_path: pathlib.PosixPath
        Path to the data of the site. The following files are required: freesurferData.csv, participants.tsv,
        group_T1w.tsv, mclf*.csv, and Qoala*.csv. These files should be inside scanner folders.

    Attributes
    ----------
    data: NDFrame
        Dataframe containing the combined data of all scanners in the site.

    Examples
    --------
    >>> ixi = Site(Path(data/raw/IXI))
    >>> ixi.data

    +---------------+-------------+--------------+-------------+----------+---+----+
    |               |3rd-Ventricle|4th-Ventricle | Age         |Brain-Stem|...|site|
    +---------------+-------------+--------------+-------------+----------+---+----+
    |participant_id |                                                              |
    +---------------+-------------+--------------+-------------+----------+---+----+
    |sub-035-0      |0.000528     | 0.000607     | 37.144422   |0.015725  |...|IXI |
    +---------------+-------------+--------------+-------------+----------+---+----+
    |sub-230-0      |0.000751     | 0.001479     | 21.152635   |0.013917  |...|IXI |
    +---------------+-------------+--------------+-------------+----------+---+----+
    |sub-231-0      |0.000861     | 0.001068     | 58.992471   |0.012547  |...|IXI |
    +---------------+-------------+--------------+-------------+----------+---+----+
    |sub-232-0      |0.000478     | 0.000502     | 28.810404   |0.013322  |...|IXI |
    +---------------+-------------+--------------+-------------+----------+---+----+
    |sub-233-0      |0.000420     | 0.000725     | 26.754278   |0.014778  |...|IXI |
    +---------------+-------------+--------------+-------------+----------+---+----+
    >>> ixi.SCANNER01.data

    +---------------+----------------------+-----------------+---+-------------+
    |               |Left-Lateral-Ventricle|Left-Inf-Lat-Vent|...|scanner      |
    +---------------+----------------------+-----------------+---+-------------+
    |participant_id |                                                          |
    +---------------+----------------------+-----------------+---+-------------+
    |sub-002-2	    |      0.003274	       |     0.000123	 |...|IXI-SCANNER01|
    +---------------+----------------------+-----------------+---+-------------+
    |sub-016-2	    |      0.015889	       |     0.000380	 |...|IXI-SCANNER01|
    +---------------+----------------------+-----------------+---+-------------+
    |sub-017-2	    |      0.007326	       |     0.000148	 |...|IXI-SCANNER01|
    +---------------+----------------------+-----------------+---+-------------+
    |...	        |         ...	       |       ...       |...|     ...	   |
    +---------------+----------------------+-----------------+---+-------------+
    |sub-582-2	    |      0.002624	       |     0.000146	 |...|IXI-SCANNER01|
    +---------------+----------------------+-----------------+---+-------------+
    |sub-584-2	    |      0.004854	       |     0.000188	 |...|IXI-SCANNER01|
    +---------------+----------------------+-----------------+---+-------------+
    |sub-585-2	    |      0.004373	       |     0.000218	 |...|IXI-SCANNER01|
    +---------------+----------------------+-----------------+---+-------------+
    """

    def __init__(self, dir_path, merge_on="image_id", renamed_id="participant_id"):
        self.dir_path = dir_path
        self.merge_on = merge_on
        self.renamed_id = renamed_id
        self.name = dir_path.name
        self._get_scanners()

    def _get_scanners(self):
        """
        Collect data for the site.

        This method verifies if the site has multiples scanners and combine the data from freesurfer, MRIQC and Qoala.
        If the data is not available it returns an empty dataframe.

        """
        self.dir_name = self.dir_path.name
        subdirs = [subpath.name for subpath in self.dir_path.glob("*") if subpath.is_dir()]
        if len(subdirs) == 0:
            self.scanner_list = []
            self._get_files()
            if self._is_complete():
                self._load_files()
                self._combine_files()
                self.data["scanner"] = self.name
                self.data.index = self.data[self.renamed_id] + "-00"
                self.complete_scanners = [self.name]
            else:
                self.data = pd.DataFrame()
        else:
            self.scanner_list = [subdir.replace(f"{self.dir_name}-", "") for subdir in subdirs]
            for scanner_name in subdirs:
                scanner = Scanner(
                    self.dir_path / Path(scanner_name), merge_on=self.merge_on, renamed_id=self.renamed_id
                )
                scanner_name = scanner_name.replace(f"{self.dir_name}-", "")
                setattr(self, scanner_name, scanner)
                scanner._get_files()
            self._combine_all_scanners()
        self.data["site"] = self.name

    def _files_exists(self, directory_path, file_pattern):
        """Verify if a file exists and is the unique file of that kind in the folder."""
        file_search = find_all_files_by_name(directory_path, file_pattern)
        if len(file_search) == 0:
            return False
        elif len(file_search) == 1:
            return file_search[0]
        else:
            return file_search
            warnings.warn("There are more than one %s file in this site." % file_pattern)

    def _get_files(self):
        """Verify if each of the files exist and is unique."""
        self.freesurferData_path = self._files_exists(self.dir_path, "freesurferData.csv")
        self.participants_path = self._files_exists(self.dir_path, "participants.tsv")
        self.iqm_path = self._files_exists(self.dir_path, "group_T1w.tsv")
        self.pred_path = self._files_exists(self.dir_path, "mclf*csv")
        self.qoala_path = self._files_exists(self.dir_path, "Qoala*.csv")

    def _is_complete(self):
        """Verify if all necessary files are there."""
        return all([self.freesurferData_path, self.participants_path, self.iqm_path, self.pred_path, self.qoala_path])

    def _load_files(self):
        self.freesurferData = pd.read_csv(self.freesurferData_path, header=0)
        self.participants = pd.read_csv(self.participants_path, header=0, sep="\t")
        self.iqm = pd.read_csv(self.iqm_path, header=0, sep="\t")
        self.pred = pd.read_csv(self.pred_path, header=0)
        self.qoala = pd.read_csv(self.qoala_path, header=0)

    def _combine_files(self):
        if all(
            self.merge_on in dataframe
            for dataframe in [self.freesurferData, self.participants, self.iqm, self.pred, self.qoala]
        ):
            self.data = pd.merge(self.participants, self.freesurferData, on=self.merge_on, how="inner")
            self.data = pd.merge(self.data, self.iqm, on=self.merge_on, how="inner")
            self.data = pd.merge(self.data, self.pred, on=self.merge_on, how="inner")
            self.data = pd.merge(self.data, self.qoala, on=self.merge_on, how="inner")
            self.data[self.renamed_id] = self.data[self.merge_on]
        else:
            warnings.warn("The field %s is not found in %s." % (self.merge_on, self.dir_path.name))
            self.data = pd.DataFrame([""], index=[""], columns=[self.renamed_id])

    def _combine_all_scanners(self):
        """If there are multiple scanners on a site they will be combined in the attribute 'data'."""
        n_scanners = len(self.scanner_list)
        participant_id_format = "-%%0%dd" % len(str(n_scanners))
        for scanner_id, scanner_name in enumerate(self.scanner_list):
            if getattr(self, scanner_name)._is_complete():
                getattr(self, scanner_name)._load_files()
                getattr(self, scanner_name)._combine_files()
                scanner_data = getattr(self, scanner_name).data
                if len(scanner_data) > 0:
                    scanner_data["scanner"] = "%s-%s" % (self.name, scanner_name)
                    try:
                        id_appendix = participant_id_format % scanner_id
                        scanner_data.index = scanner_data[self.renamed_id] + str(id_appendix)
                    except TypeError:
                        print(self.dir_path.name)
        self.complete_scanners = [
            scanner_name for scanner_name in self.scanner_list if getattr(self, scanner_name)._is_complete()
        ]
        if len(self.complete_scanners) > 0:
            self.data = pd.concat([getattr(self, scanner).data for scanner in self.complete_scanners], sort=True)
        else:
            self.data = pd.DataFrame()


class Scanner(Site):
    """
    A class for scanner definition.

    Tools for collecting and combining data from a single scanner given a path to the scanner data.

    Parameters
    ----------
    dir_path: pathlib.PosixPath
        Path to the data of the scanner. The following files are required: freesurferData.csv, participants.tsv,
        group_T1w.tsv, mclf*.csv, and Qoala*.csv.

    Attributes
    ----------
    data: NDFrame
        Dataframe containing the combined data of all scanners in the site.

    Examples
    --------
    >>> scanner_01 = Site(Path(data/raw/IXI/SCANNER01))
    >>> scanner_01.data
                Left-Lateral-Ventricle	 Left-Inf-Lat-Vent ... scanner
    participant_id
    sub-002-2	0.003274	             0.000123	... IXI-SCANNER01
    sub-016-2	0.015889	             0.000380	... IXI-SCANNER01
    sub-017-2	0.007326	             0.000148	... IXI-SCANNER01
    ...	        ...	                     ...	    ...	...
    sub-582-2	0.002624	             0.000146	... IXI-SCANNER01
    sub-584-2	0.004854	             0.000188	... IXI-SCANNER01
    sub-585-2	0.004373	             0.000218	... IXI-SCANNER01

    [313 rows × 179 columns]

    """

    def __init__(self, dir_path, merge_on="image_id", renamed_id="participant_id"):
        self.dir_path = dir_path
        self.name = dir_path.name
        self.merge_on = merge_on
        self.renamed_id = renamed_id


class DataSet(Site):
    """
    A class for dataset definition.

    Tools for collecting and combining data from all sites given a path to a BIDS dataset.

    Parameters
    ----------
    dir_path: pathlib.PosixPath
        Path to the data of the site. The following files are required: freesurferData.csv, participants.tsv,
        group_T1w.tsv, mclf*.csv, and Qoala*.csv. These files should be inside scanner folders which should be inside
        site folders.

    Attributes
    ----------
    data: NDFrame
        Dataframe containing the combined data of all scanners in the site.

    Examples
    --------
    >>> DATASET = DataSet(Path(data/raw/))
    >>> Data.data
                         3rd-Ventricle  4th-Ventricle  Age          Brain-Stem ... site
    participant_id
    sub-035-00-00        0.000528       0.000607       37.144422    0.015725   ... IXI
    sub-230-00-00        0.000751       0.001479       21.152635    0.013917   ... IXI
    sub-231-00-00        0.000861       0.001068       58.992471    0.012547   ... IXI
    ...                  ...            ...            ...          ...        ... ...
    sub-45-00-33	     0.000570	    0.001581	   21.0	        0.011838   ... MoralJudgment
    sub-46-00-33	     0.000552	    0.001528	   36.0	        0.013947   ... MoralJudgment
    sub-47-00-33	     0.000609	    0.000893	   20.0	        0.013515   ... MoralJudgment

    [9303 rows × 180 columns]

    >>> DATASET.IXI.data
                         3rd-Ventricle  4th-Ventricle  Age          Brain-Stem ... site
    participant_id
    sub-035-0            0.000528       0.000607       37.144422    0.015725   ... IXI
    sub-230-0            0.000751       0.001479       21.152635    0.013917   ... IXI
    sub-231-0            0.000861       0.001068       58.992471    0.012547   ... IXI
    sub-232-0            0.000478       0.000502       28.810404    0.013322   ... IXI
    sub-233-0            0.000420       0.000725       26.754278    0.014778   ... IXI

    [535 rows x 180 columns]

    >>> DATASET.IXI.SCANNER01.data
                Left-Lateral-Ventricle	 Left-Inf-Lat-Vent ... scanner
    participant_id
    sub-002-2	0.003274	             0.000123	... IXI-SCANNER01
    sub-016-2	0.015889	             0.000380	... IXI-SCANNER01
    sub-017-2	0.007326	             0.000148	... IXI-SCANNER01
    ...	        ...	                     ...	    ...	...
    sub-582-2	0.002624	             0.000146	... IXI-SCANNER01
    sub-584-2	0.004854	             0.000188	... IXI-SCANNER01
    sub-585-2	0.004373	             0.000218	... IXI-SCANNER01

    [313 rows × 179 columns]

    """

    def __init__(self, dir_path, merge_on="image_id", renamed_id="participant_id"):
        self.dir_path = dir_path
        self.name = "All sites"
        self.merge_on = merge_on
        self.renamed_id = renamed_id
        self._get_sites()
        self._combine_all_sites()

    def _get_sites(self):
        self.site_paths = [subpath for subpath in self.dir_path.glob("*") if subpath.is_dir()]
        self.sites = [site_path.name for site_path in self.site_paths]
        self.not_empty_sites = []
        for site_path in self.site_paths:
            site = Site(site_path, merge_on=self.merge_on, renamed_id="participant_id")
            setattr(self, site.name, site)
            self.not_empty_sites.append(site.name)

    def _combine_all_sites(self):
        n_sites = len(self.not_empty_sites)
        participant_id_format = "-%%0%dd" % len(str(n_sites))
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


if __name__ == "__main__":
    DATASET = DataSet(Path("data/raw/"))
    DATASET.data.to_csv("data/processed/dataset.csv")
