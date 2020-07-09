"""Test tools for data finding and collection."""
from collections import namedtuple
from pathlib import Path
from os import popen

import pytest

from neuroharmony.data import collect_tools
from pandas.core.generic import NDFrame


def test_fetch_sample():
    data = collect_tools.fetch_sample()
    assert isinstance(data, NDFrame)
