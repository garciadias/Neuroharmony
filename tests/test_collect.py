"""Test tools for data finding and collection."""
from collections import namedtuple
from pathlib import Path
from os import popen

import pytest

from neuroharmony.data import collect_tools


@pytest.fixture(scope='session')
def resources(tmpdir_factory):
    """Set up."""
    r = namedtuple('resources', 'server_root')
    r.server_root = '/media/d/My Passport/SynologyDrive/'
    r.freesurfer_root = '%s/FreeSurfer_preprocessed/' % r.server_root
    r.bids_root = '%sBIDS_data/' % r.server_root
    r.fsurfer_root = '%sFreeSurfer_preprocessed/' % r.server_root
    tmpdir = tmpdir_factory.mktemp('tmp').mkdir('output_tmp')
    r.tmpdir = tmpdir
    r.participants_list = list(collect_tools.find_all_files_by_name(r.bids_root,
                                                                    'participants.tsv',
                                                                    depth=3))
    return r


def test_find_all_files(resources):
    """Test we can find the raw data files."""
    participants_list = resources.participants_list
    n_files = int(popen('find %s -maxdepth 3 -name \'participants.tsv\' | wc -l' %
                        resources.bids_root.replace(' ', '\\ ')).read())
    rise_server_disconnected = "The path for the bids data is not found. Verify if the server is connected"
    assert Path(resources.bids_root).exists(), rise_server_disconnected
    assert len(participants_list) == n_files
    return participants_list


def test_collect_datafile(resources):
    """Test we can collect the raw data files."""
    participants_list = resources.participants_list
    collect_tools.collect_datafile(participants_list[0],
                                   resources.bids_root, str(resources.tmpdir))
    assert Path(str(participants_list[0]).replace(resources.bids_root,
                                                  str(resources.tmpdir))).exists()
