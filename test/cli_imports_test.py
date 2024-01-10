# Copyright Modal Labs 2023
import os
import pytest
from pathlib import Path

from modal.cli.import_refs import (
    DEFAULT_STUB_NAME,
    get_by_object_path,
    import_file_or_module,
    parse_import_ref,
)
from modal.functions import _Function
from modal.stub import _LocalEntrypoint, _Stub
from modal_utils.async_utils import synchronizer

# Some helper vars for import_stub tests:
local_entrypoint_src = """
import modal

stub = modal.Stub()
@stub.local_entrypoint()
def main():
    pass
"""
python_module_src = """
import modal
stub = modal.Stub("FOO")
other_stub = modal.Stub("BAR")
@other_stub.function()
def func():
    pass
@stub.cls()
class Parent:
    @modal.method()
    def meth(self):
        pass

assert not __package__
"""

python_package_src = """
import modal
stub = modal.Stub("FOO")
other_stub = modal.Stub("BAR")
@other_stub.function()
def func():
    pass
assert __package__ == "pack"
"""

python_subpackage_src = """
import modal
stub = modal.Stub("FOO")
other_stub = modal.Stub("BAR")
@other_stub.function()
def func():
    pass
assert __package__ == "pack.sub"
"""

empty_dir_with_python_file = {"mod.py": python_module_src}


dir_containing_python_package = {
    "dir": {"sub": {"mod.py": python_module_src}},
    "pack": {
        "mod.py": python_package_src,
        "local.py": local_entrypoint_src,
        "__init__.py": "",
        "sub": {"mod.py": python_subpackage_src, "__init__.py": ""},
    },
}


@pytest.mark.parametrize(
    ["dir_structure", "ref", "expected_object_type"],
    [
        # # file syntax
        (empty_dir_with_python_file, "mod.py", _Stub),
        (empty_dir_with_python_file, "mod.py::stub", _Stub),
        (empty_dir_with_python_file, "mod.py::stub.Parent.meth", _Function),
        (empty_dir_with_python_file, "mod.py::other_stub", _Stub),
        (empty_dir_with_python_file, "mod.py::other_stub.func", _Function),
        (dir_containing_python_package, "pack/mod.py", _Stub),
        (dir_containing_python_package, "pack/sub/mod.py", _Stub),
        (dir_containing_python_package, "dir/sub/mod.py", _Stub),
        # # python module syntax
        (empty_dir_with_python_file, "mod", _Stub),
        (empty_dir_with_python_file, "mod::stub", _Stub),
        (empty_dir_with_python_file, "mod::other_stub", _Stub),
        (dir_containing_python_package, "pack.mod", _Stub),
        (dir_containing_python_package, "pack.mod::other_stub", _Stub),
        (dir_containing_python_package, "pack/local.py::stub.main", _LocalEntrypoint),
    ],
)
def test_import_object(dir_structure, ref, expected_object_type, mock_dir):
    with mock_dir(dir_structure):
        import_ref = parse_import_ref(ref)
        module = import_file_or_module(import_ref.file_or_module)
        imported_object = get_by_object_path(module, import_ref.object_path or DEFAULT_STUB_NAME)
        _translated_obj = synchronizer._translate_in(imported_object)
        assert isinstance(_translated_obj, expected_object_type)


def test_import_package_properly():
    # https://modalbetatesters.slack.com/archives/C031Z7H15DG/p1664063245553579
    # if importing pkg/mod.py, it should be imported as pkg.mod,
    # so that __package__ is set properly

    p = Path(__file__).parent.parent / "modal_test_support/assert_package.py"
    abs_p = str(p.absolute())
    rel_p = str(p.relative_to(os.getcwd()))
    print(f"abs_p={abs_p} rel_p={rel_p}")

    assert import_file_or_module(rel_p).name == "xyz"
    assert import_file_or_module(abs_p).name == "xyz"


def test_get_by_object_path():
    class NS(dict):
        def __getattr__(self, n):
            return dict.__getitem__(self, n)

    # simple
    assert get_by_object_path(NS(foo="bar"), "foo") == "bar"

    # nested simple
    assert get_by_object_path(NS(foo=NS(bar="baz")), "foo.bar") == "baz"

    # try to find item keys with periods in them (ugh).
    # this helps resolving lifecycled functions
    assert get_by_object_path(NS({"foo.bar": "baz"}), "foo.bar") == "baz"
