import pytest

from .core import Universe


@pytest.fixture
def world(tmp_path, monkeypatch):
    with Universe(tmp_path, monkeypatch, seed=22026) as universe:
        yield universe
