
from collections import namedtuple
from pathlib import Path
from os import popen

import pytest

from src.data import collect_tools


@pytest.fixture(scope='session')
def resouces(tmpdir_factory):
    r = namedtuple('resources', 'server_root')
    r.server_root = '/run/user/1000/gvfs/smb-share:server=kc-deeplab.local,share=deeplearning/'
    r.freesurfer_root = '%s/FreeSurfer_preprocessed/' % r.server_root
    r.bids_root = '%sBIDS_data/' % r.server_root
    r.fsurfer_root = '%sFreeSurfer_preprocessed/' % r.server_root
    tmpdir = tmpdir_factory.mktemp('tmp').mkdir('output_tmp')
    r.tmpdir = tmpdir
    r.participants_list = list(collect_tools.find_all_files_by_name(r.bids_root, 'participants.tsv', depth=2))
    return r


def test_find_all_files_and_collect_datafile(resouces):
    participants_list = resouces.participants_list
    n_files = int(popen('find %s -maxdepth 2 -name \'participants.tsv\' | wc -l' % resouces.bids_root).read())
    assert len(participants_list) == n_files
    return participants_list


def test_collect_datafile(resouces):
    participants_list = resouces.participants_list
    collect_tools.collect_datafile(str(participants_list[0]), resouces.bids_root, str(resouces.tmpdir))
    assert Path(str(participants_list[0]).replace(resouces.bids_root, str(resouces.tmpdir))).exists()
