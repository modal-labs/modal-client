import pytest
from pathlib import Path

from modal_utils.package_utils import get_module_mount_info, import_stub_by_ref


def test_get_module_mount_info():
    res = get_module_mount_info("modal")
    assert len(res) == 1
    assert res[0][0] == "modal"

    res = get_module_mount_info("asyncio")
    assert len(res) == 1
    assert res[0][0] == "asyncio"


# Some helper vars for import_stub_by_ref tests:

python_module_src = """
stub = "FOO"
other_stub = "BAR"
"""

empty_dir_with_python_file = {"mod.py": python_module_src}

dir_containing_python_package = {"pack": {"mod.py": python_module_src, "__init__.py": ""}}


@pytest.mark.parametrize(
    ["dir_structure", "ref", "expected_stub_value"],
    [
        # # file syntax
        (empty_dir_with_python_file, "mod.py", "FOO"),
        (empty_dir_with_python_file, "mod.py::stub", "FOO"),
        (empty_dir_with_python_file, "mod.py:stub", "FOO"),
        (empty_dir_with_python_file, "mod.py::other_stub", "BAR"),
        # python module syntax
        (empty_dir_with_python_file, "mod", "FOO"),
        (empty_dir_with_python_file, "mod::stub", "FOO"),
        (empty_dir_with_python_file, "mod:stub", "FOO"),
        (empty_dir_with_python_file, "mod::other_stub", "BAR"),
        (dir_containing_python_package, "pack/mod.py", "FOO"),
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

    p = "client/modal_test_support/assert_package.py"
    assert import_stub_by_ref(p) == "xyz"

    # TODO(erikbern): import_stub_by_ref doesn't take absolute paths
    # we should fix that and do this instead:
    p = str((Path(__file__) / "../../client/modal_test_support/assert_package.py").resolve())
    # assert import_stub_by_ref(p) == "xyz"  # DOES NOT WORK
