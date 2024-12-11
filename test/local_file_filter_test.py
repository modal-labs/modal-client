import pytest

from modal._utils.local_file_filter import LocalFileFilter


@pytest.fixture
def tmp_path_with_content(tmp_path):
    (tmp_path / "venv").mkdir()
    (tmp_path / "venv" / "lib").mkdir()
    (tmp_path / "venv" / "lib" / "python3.12").mkdir()
    (tmp_path / "venv" / "lib" / "python3.12" / "site-packages").mkdir()
    (tmp_path / "venv" / "lib" / "python3.12" / "site-packages" / "foo.py").write_text("foo")

    (tmp_path / "data.txt").write_text("hello")

    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "sub.txt").write_text("world")

    (tmp_path / "module").mkdir()
    (tmp_path / "module" / "__init__.py").write_text("foo")
    (tmp_path / "module" / "sub.py").write_text("bar")
    (tmp_path / "module" / "sub").mkdir()
    (tmp_path / "module" / "sub" / "__init__.py").write_text("baz")
    (tmp_path / "module" / "sub" / "sub.py").write_text("qux")
    return tmp_path


def test_against_paths(tmp_path_with_content):
    # include all files, exclude .venv files
    lff = LocalFileFilter("**/*", "!**/venv")
    assert not lff(tmp_path_with_content / "venv")
    assert not lff(tmp_path_with_content / "venv" / "lib")
    assert not lff(tmp_path_with_content / "venv" / "lib" / "python3.12")
    assert not lff(tmp_path_with_content / "venv" / "lib" / "python3.12" / "site-packages")
    assert not lff(tmp_path_with_content / "venv" / "lib" / "python3.12" / "site-packages" / "foo.py")
