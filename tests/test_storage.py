import os
import tempfile
from pathlib import Path
import pytest
from brainlayer.storage import BrainStorage


@pytest.fixture
def storage(tmp_path):
    return BrainStorage(base_dir=tmp_path)


def test_get_project_dir_creates_structure(storage, tmp_path):
    project_dir = storage.get_project_dir("myproject")
    assert project_dir.exists()
    assert (project_dir / "docs.local").exists()
    assert (project_dir / "plans").exists()


def test_store_file(storage):
    storage.store("myproject", "docs.local/research/thing.md", "# Research\nSome content")
    stored = storage.read("myproject", "docs.local/research/thing.md")
    assert stored == "# Research\nSome content"


def test_list_projects(storage):
    storage.get_project_dir("myproject")
    storage.get_project_dir("other-project")
    projects = storage.list_projects()
    assert set(projects) == {"myproject", "other-project"}


def test_list_files(storage):
    storage.store("myproject", "docs.local/a.md", "a")
    storage.store("myproject", "docs.local/b.md", "b")
    storage.store("myproject", "plans/plan.md", "plan")
    files = storage.list_files("myproject")
    assert len(files) == 3
    assert any("a.md" in str(f) for f in files)


def test_default_base_dir():
    s = BrainStorage()
    assert str(s.base_dir).endswith("brainlayer/storage")


def test_store_creates_subdirectories(storage):
    storage.store("myproject", "docs.local/research/deep/nested/file.md", "content")
    assert storage.read("myproject", "docs.local/research/deep/nested/file.md") == "content"


def test_path_traversal_blocked_project(storage):
    with pytest.raises(ValueError, match="escapes storage root"):
        storage.store("..", "evil.txt", "pwned")


def test_path_traversal_blocked_relative_path(storage):
    with pytest.raises(ValueError, match="escapes storage root"):
        storage.read("myproject", "../../../etc/passwd")


def test_path_traversal_blocked_exists(storage):
    with pytest.raises(ValueError, match="escapes storage root"):
        storage.exists("..", "etc/passwd")
