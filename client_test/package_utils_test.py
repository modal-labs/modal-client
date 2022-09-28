import os
import pytest
from pathlib import Path

from modal_utils.package_utils import get_module_mount_info, import_stub_by_ref


def test_get_module_mount_info():
    res = get_module_mount_info("modal")
    assert len(res) == 1
    assert res[0][0] == True

    res = get_module_mount_info("asyncio")
    assert len(res) == 1
    assert res[0][0] == True

    res = get_module_mount_info("six")
    assert len(res) == 1
    assert res[0][0] == False


# Some helper vars for import_stub_by_ref tests:

python_module_src = """
stub = "FOO"
other_stub = "BAR"
assert not __package__
"""

python_package_src = """
stub = "FOO"
other_stub = "BAR"
assert __package__ == "pack"
"""

python_subpackage_src = """
stub = "FOO"
other_stub = "BAR"
assert __package__ == "pack.sub"
"""

empty_dir_with_python_file = {"mod.py": python_module_src}


dir_containing_python_package = {
    "dir": {"sub": {"mod.py": python_module_src}},
    "pack": {
        "mod.py": python_package_src,
        "__init__.py": "",
        "sub": {"mod.py": python_subpackage_src, "__init__.py": ""},
    },
}


@pytest.mark.parametrize(
    ["dir_structure", "ref", "expected_stub_value"],
    [
        # # file syntax
        (empty_dir_with_python_file, "mod.py", "FOO"),
        (empty_dir_with_python_file, "mod.py::stub", "FOO"),
        (empty_dir_with_python_file, "mod.py:stub", "FOO"),
        (empty_dir_with_python_file, "mod.py::other_stub", "BAR"),
        (dir_containing_python_package, "pack/mod.py", "FOO"),
        (dir_containing_python_package, "pack/sub/mod.py", "FOO"),
        (dir_containing_python_package, "dir/sub/mod.py", "FOO"),
        # python module syntax
        (empty_dir_with_python_file, "mod", "FOO"),
        (empty_dir_with_python_file, "mod::stub", "FOO"),
        (empty_dir_with_python_file, "mod:stub", "FOO"),
        (empty_dir_with_python_file, "mod::other_stub", "BAR"),
        (dir_containing_python_package, "pack.mod", "FOO"),
        (dir_containing_python_package, "pack.mod::other_stub", "BAR"),
    ],
)
def test_import_stub_by_ref(dir_structure, ref, expected_stub_value, mock_dir):
    with mock_dir(dir_structure):
        imported_stub = import_stub_by_ref(ref)
        assert imported_stub == expected_stub_value


def test_import_package_properly():
    # https://modalbetatesters.slack.com/archives/C031Z7H15DG/p1664063245553579
    # if importing pkg/mod.py, it should be imported as pkg.mod,
    # so that __package__ is set properly

    p = Path(__file__).parent.parent / "modal_test_support/assert_package.py"
    abs_p = str(p.absolute())
    rel_p = str(p.relative_to(os.getcwd()))
    print(f"abs_p={abs_p} rel_p={rel_p}")

    assert import_stub_by_ref(rel_p) == "xyz"
    assert import_stub_by_ref(abs_p) == "xyz"
