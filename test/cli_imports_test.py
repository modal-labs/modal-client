# Copyright Modal Labs 2023
import pytest
import types
from typing import cast

from modal.app import App, LocalEntrypoint
from modal.cli.import_refs import (
    CLICommand,
    MethodReference,
    NonConclusiveImportRef,
    import_and_filter,
    import_file_or_module,
    list_cli_commands,
)
from modal.exception import InvalidError
from modal.functions import Function
from modal.partial_function import method, web_server

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
    ["dir_structure", "ref", "returned_runnable_type", "num_error_choices"],
    [
        # # file syntax
        (empty_dir_with_python_file, "mod.py", None, 2),
        (empty_dir_with_python_file, "mod.py::app", MethodReference, None),
        (empty_dir_with_python_file, "mod.py::other_app", Function, None),
        (dir_containing_python_package, "pack/file.py", Function, None),
        (dir_containing_python_package, "pack/sub/subfile.py", Function, None),
        (dir_containing_python_package, "dir/sub/subfile.py", Function, None),
        # # python module syntax
        (empty_dir_with_python_file, "mod", Function, 2),
        (empty_dir_with_python_file, "mod::app", MethodReference, None),
        (empty_dir_with_python_file, "mod::other_app", Function, 2),
        (dir_containing_python_package, "pack.mod", Function, 2),
        (dir_containing_python_package, "pack.mod::other_app", Function, 2),
        (dir_containing_python_package, "pack/local.py::app.main", LocalEntrypoint, None),
    ],
)
def test_import_and_filter(dir_structure, ref, mock_dir, returned_runnable_type, num_error_choices):
    with mock_dir(dir_structure):
        try:
            runnable = import_and_filter(ref, "dummy", accept_local_entrypoint=True, accept_webhook=False)
            assert isinstance(runnable, returned_runnable_type)
        except NonConclusiveImportRef as exc:
            assert len(exc.all_usable_commands) == num_error_choices


def test_import_and_filter_2(monkeypatch, supports_on_path):
    def import_runnable(name_prefix, accept_local_entrypoint=False, accept_webhook=False):
        return import_and_filter(
            f"import_and_filter_source::{name_prefix}", "dummy command", accept_local_entrypoint, accept_webhook
        )

    with pytest.raises(NonConclusiveImportRef, match="non-web") as exc:
        import_runnable("app_with_one_web_function", accept_webhook=False, accept_local_entrypoint=True)
    assert len(exc.value.all_usable_commands) == 4

    import_runnable("app_with_one_web_function", accept_webhook=True)
    import_runnable("app_with_one_function_one_web_endpoint", accept_webhook=False)

    with pytest.raises(NonConclusiveImportRef, match=r"(?s)You need to specify.*(\b)f1(\b).*(\b)web2(\b)") as exc:
        import_runnable("app_with_one_function_one_web_endpoint", accept_webhook=True)
    assert len(exc.value.all_usable_commands) == 7

    with pytest.raises(NonConclusiveImportRef, match="non-web"):
        import_runnable("app_with_one_web_method", accept_webhook=False)

    import_runnable("app_with_one_web_method", accept_webhook=True)

    assert isinstance(
        import_runnable("app_with_local_entrypoint_and_function", accept_local_entrypoint=True), LocalEntrypoint
    )
    assert isinstance(
        import_runnable("app_with_local_entrypoint_and_function", accept_local_entrypoint=False), Function
    )


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


def test_invalid_source_file_exception():
    with pytest.raises(InvalidError, match="Invalid Modal source filename: 'foo.bar.py'"):
        import_file_or_module("path/to/foo.bar.py")


def test_list_cli_commands():
    class FakeModule:
        app = App()
        other_app = App()

    @FakeModule.app.function(serialized=True, name="foo")
    def foo():
        pass

    @FakeModule.app.cls(serialized=True)
    class Cls:
        @method()
        def method_1(self):
            pass

        @web_server(8000)
        def web_method(self):
            pass

    def non_modal_func():
        pass

    FakeModule.non_modal_func = non_modal_func  # type: ignore[attr-defined]
    FakeModule.foo = foo  # type: ignore[attr-defined]
    FakeModule.Cls = Cls  # type: ignore[attr-defined]

    res = list_cli_commands(cast(types.ModuleType, FakeModule))

    assert res == [
        CLICommand(["foo", "app.foo"], foo, False),  # type: ignore
        CLICommand(["Cls.method_1", "app.Cls.method_1"], MethodReference(Cls, "method_1"), False),  # type: ignore
        CLICommand(["Cls.web_method", "app.Cls.web_method"], MethodReference(Cls, "web_method"), True),  # type: ignore
    ]
