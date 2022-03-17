import os
import pytest

from modal_utils.package_utils import get_module_mount_info, import_app_by_ref


def test_get_module_mount_info():
    res = get_module_mount_info("modal")
    assert len(res) == 1
    assert res[0][0] == "modal"

    res = get_module_mount_info("asyncio")
    assert len(res) == 1
    assert res[0][0] == "asyncio"


# Some helper vars for import_app_by_ref tests:

python_module_src = """
app = "FOO"
other_app = "BAR"
"""

empty_dir_with_python_file = {"mod.py": python_module_src}

dir_containing_python_package = {"pack": {"mod.py": python_module_src, "__init__.py": ""}}


@pytest.mark.parametrize(
    ["dir_structure", "ref", "expected_app_value"],
    [
        # # file syntax
        (empty_dir_with_python_file, "mod.py", "FOO"),
        (empty_dir_with_python_file, "mod.py::app", "FOO"),
        (empty_dir_with_python_file, "mod.py:app", "FOO"),
        (empty_dir_with_python_file, "mod.py::other_app", "BAR"),
        # python module syntax
        (empty_dir_with_python_file, "mod", "FOO"),
        (empty_dir_with_python_file, "mod::app", "FOO"),
        (empty_dir_with_python_file, "mod:app", "FOO"),
        (empty_dir_with_python_file, "mod::other_app", "BAR"),
        (dir_containing_python_package, "pack/mod.py", "FOO"),
        (dir_containing_python_package, "pack.mod", "FOO"),
        (dir_containing_python_package, "pack.mod::other_app", "BAR"),
    ],
)
def test_import_app_by_ref(dir_structure, ref, expected_app_value, mock_dir):
    with mock_dir(dir_structure) as root_dir:
        os.chdir(root_dir)
        imported_app = import_app_by_ref(ref)
        assert imported_app == expected_app_value
