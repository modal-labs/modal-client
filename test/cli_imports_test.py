# Copyright Modal Labs 2023
import pytest

from modal._utils.async_utils import synchronizer
from modal.app import _App, _LocalEntrypoint
from modal.cli.import_refs import (
    DEFAULT_APP_NAME,
    get_by_object_path,
    import_file_or_module,
    parse_import_ref,
)

# Some helper vars for import_stub tests:
local_entrypoint_src = """
import modal

app = modal.App()
@app.local_entrypoint()
def main():
    pass
"""
python_module_src = """
import modal
app = modal.App("FOO")
other_app = modal.App("BAR")
@other_app.function()
def func():
    pass
@app.cls()
class Parent:
    @modal.method()
    def meth(self):
        pass

assert not __package__
"""

python_package_src = """
import modal
app = modal.App("FOO")
other_app = modal.App("BAR")
@other_app.function()
def func():
    pass
assert __package__ == "pack"
"""

python_subpackage_src = """
import modal
app = modal.App("FOO")
other_app = modal.App("BAR")
@other_app.function()
def func():
    pass
assert __package__ == "pack.sub"
"""

python_file_src = """
import modal
app = modal.App("FOO")
other_app = modal.App("BAR")
@other_app.function()
def func():
    pass

assert __package__ == ""
"""

empty_dir_with_python_file = {"mod.py": python_module_src}


dir_containing_python_package = {
    "dir": {"sub": {"mod.py": python_module_src, "subfile.py": python_file_src}},
    "pack": {
        "file.py": python_file_src,
        "mod.py": python_package_src,
        "local.py": local_entrypoint_src,
        "__init__.py": "",
        "sub": {"mod.py": python_subpackage_src, "__init__.py": "", "subfile.py": python_file_src},
    },
}


@pytest.mark.parametrize(
    ["dir_structure", "ref", "expected_object_type"],
    [
        # # file syntax
        (empty_dir_with_python_file, "mod.py", _App),
        (empty_dir_with_python_file, "mod.py::app", _App),
        (empty_dir_with_python_file, "mod.py::other_app", _App),
        (dir_containing_python_package, "pack/file.py", _App),
        (dir_containing_python_package, "pack/sub/subfile.py", _App),
        (dir_containing_python_package, "dir/sub/subfile.py", _App),
        # # python module syntax
        (empty_dir_with_python_file, "mod", _App),
        (empty_dir_with_python_file, "mod::app", _App),
        (empty_dir_with_python_file, "mod::other_app", _App),
        (dir_containing_python_package, "pack.mod", _App),
        (dir_containing_python_package, "pack.mod::other_app", _App),
        (dir_containing_python_package, "pack/local.py::app.main", _LocalEntrypoint),
    ],
)
def test_import_object(dir_structure, ref, expected_object_type, mock_dir):
    with mock_dir(dir_structure):
        import_ref = parse_import_ref(ref)
        module = import_file_or_module(import_ref.file_or_module)
        imported_object = get_by_object_path(module, import_ref.object_path or DEFAULT_APP_NAME)
        _translated_obj = synchronizer._translate_in(imported_object)
        assert isinstance(_translated_obj, expected_object_type)


def test_import_package_and_module_names(monkeypatch, supports_dir):
    # We try to reproduce the package/module naming standard that the `python` command line tool uses,
    # i.e. when loading using a module path (-m flag w/ python) you get a fully qualified package/module name
    # but when loading using a filename, some/mod.py it will not have a __package__

    # The biggest difference is that __name__ of the imported "entrypoint" script
    # is __main__ when using `python` but in the Modal runtime it's the name of the
    # file minus the ".py", since Modal has its own __main__
    monkeypatch.chdir(supports_dir)
    mod1 = import_file_or_module("assert_package")
    assert mod1.__package__ == ""
    assert mod1.__name__ == "assert_package"

    monkeypatch.chdir(supports_dir.parent)
    mod2 = import_file_or_module("test.supports.assert_package")
    assert mod2.__package__ == "test.supports"
    assert mod2.__name__ == "test.supports.assert_package"

    mod3 = import_file_or_module("supports/assert_package.py")
    assert mod3.__package__ == ""
    assert mod3.__name__ == "assert_package"


def test_get_by_object_path():
    class NS(dict):
        def __getattr__(self, n):
            return dict.__getitem__(self, n)

    # simple
    assert get_by_object_path(NS(foo="bar"), "foo") == "bar"
    assert get_by_object_path(NS(foo="bar"), "bar") is None

    # nested simple
    assert get_by_object_path(NS(foo=NS(bar="baz")), "foo.bar") == "baz"

    # try to find item keys with periods in them (ugh).
    # this helps resolving lifecycled functions
    assert get_by_object_path(NS({"foo.bar": "baz"}), "foo.bar") == "baz"
