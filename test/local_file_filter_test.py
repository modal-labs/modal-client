# Copyright Modal Labs 2024
import os
import pytest

from modal._utils.local_file_filter import LocalFileFilter


@pytest.fixture
def tmp_path_with_content(tmp_path):
    (tmp_path / "venv").mkdir()
    (tmp_path / "venv" / "lib").mkdir()
    (tmp_path / "venv" / "lib" / "python3.12").mkdir()
    (tmp_path / "venv" / "lib" / "python3.12" / "site-packages").mkdir()
    fp = tmp_path / "venv" / "lib" / "python3.12" / "site-packages" / "foo.py"
    fp.write_text("foo")
    print(f"fp: {fp}")

    (tmp_path / "data.txt").write_text("hello")

    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "sub.txt").write_text("world")

    (tmp_path / "module").mkdir()
    (tmp_path / "module" / "__init__.py").write_text("foo")
    (tmp_path / "module" / "sub.py").write_text("bar")
    (tmp_path / "module" / "sub").mkdir()
    (tmp_path / "module" / "sub" / "__init__.py").write_text("baz")
    (tmp_path / "module" / "sub" / "foo.pyc").write_text("baz")
    (tmp_path / "module" / "sub" / "sub.py").write_text("qux")

    # walk tmp_path and add all file paths to a set
    file_paths = set()
    for root, _, files in os.walk(tmp_path):
        for file in files:
            file_paths.add(os.path.join(root, file))

    return tmp_path, file_paths


def test_against_paths(tmp_path_with_content):
    tmp_path, file_paths = tmp_path_with_content

    # match everything that's not ignored
    lff = LocalFileFilter("**/*", "!**/venv")

    for file_path in file_paths:
        if "venv" in file_path:
            assert not lff(file_path)
        else:
            assert lff(file_path)

    lff = LocalFileFilter("**/*.py")

    for file_path in file_paths:
        if file_path.endswith(".py"):
            assert lff(file_path)
        else:
            assert not lff(file_path)


def test_empty_patterns(tmp_path_with_content):
    tmp_path, file_paths = tmp_path_with_content
    lff = LocalFileFilter()

    for file_path in file_paths:
        assert not lff(file_path)


def test_invert_patterns(tmp_path_with_content):
    tmp_path, file_paths = tmp_path_with_content

    # match everything that's not ignored
    lff = ~LocalFileFilter("**/*", "!**/venv")

    for file_path in file_paths:
        if "venv" in file_path:
            assert lff(file_path)
        else:
            assert not lff(file_path)

    # empty patterns should match nothing
    # inverted empty patterns should match everything
    lff = ~LocalFileFilter()
    for file_path in file_paths:
        assert lff(file_path)

    # single negative pattern should match nothing
    lff = LocalFileFilter("!**/*.txt")
    for file_path in file_paths:
        assert not lff(file_path)
