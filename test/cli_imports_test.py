# Copyright Modal Labs 2023
import pytest

from modal.app import App, LocalEntrypoint
from modal.cli.import_refs import (
    AutoRunPriority,
    CLICommand,
    ImportRef,
    MethodReference,
    import_and_filter,
    import_file_or_module,
    list_cli_commands,
    parse_import_ref,
)
from modal.exception import InvalidError, PendingDeprecationError
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
app = modal.App("FOO", include_source=True)  # TODO: remove include_source=True)
other_app = modal.App("BAR", include_source=True)  # TODO: remove include_source=True)
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
app = modal.App("FOO", include_source=True)  # TODO: remove include_source=True)
other_app = modal.App("BAR", include_source=True)  # TODO: remove include_source=True)
@other_app.function()
def func():
    pass
assert __package__ == "pack005"
"""

python_subpackage_src = """
import modal
app = modal.App("FOO", include_source=True)  # TODO: remove include_source=True)
other_app = modal.App("BAR", include_source=True)  # TODO: remove include_source=True)
@other_app.function()
def func():
    pass
assert __package__ == "pack007.sub009"
"""

python_file_src = """
import modal
app = modal.App("FOO", include_source=True)  # TODO: remove include_source=True)
other_app = modal.App("BAR", include_source=True)  # TODO: remove include_source=True)
@other_app.function()
def func():
    pass

assert __package__ == ""
"""

empty_dir_with_python_file = {"mod000.py": python_module_src}


dir_containing_python_package = {
    "dir001": {"sub002": {"mod003.py": python_module_src, "subfile004.py": python_file_src}},
    "pack005": {
        "file006.py": python_file_src,
        "mod007.py": python_package_src,
        "local008.py": local_entrypoint_src,
        "__init__.py": "",
        "sub009": {"mod010.py": python_subpackage_src, "__init__.py": "", "subfile011.py": python_file_src},
    },
}


@pytest.mark.parametrize(
    ["dir_structure", "ref", "returned_runnable_type", "num_error_choices"],
    [
        # # file syntax
        (empty_dir_with_python_file, "mod000.py", type(None), 2),
        (empty_dir_with_python_file, "mod000.py::app", MethodReference, 2),
        (empty_dir_with_python_file, "mod000.py::other_app", Function, 2),
        (dir_containing_python_package, "pack005/file006.py", Function, 1),
        (dir_containing_python_package, "pack005/sub009/subfile011.py", Function, 1),
        (dir_containing_python_package, "dir001/sub002/subfile004.py", Function, 1),
        # # python module syntax
        (empty_dir_with_python_file, "mod000::func", Function, 2),
        (empty_dir_with_python_file, "mod000::other_app.func", Function, 2),
        (empty_dir_with_python_file, "mod000::app.func", type(None), 2),
        (empty_dir_with_python_file, "mod000::Parent.meth", MethodReference, 2),
        (empty_dir_with_python_file, "mod000::other_app", Function, 2),
        (dir_containing_python_package, "pack005.mod007", Function, 1),
        (dir_containing_python_package, "pack005.mod007::other_app", Function, 1),
        (dir_containing_python_package, "pack005/local008.py::app.main", LocalEntrypoint, 1),
    ],
)
def test_import_and_filter(dir_structure, ref, mock_dir, returned_runnable_type, num_error_choices):
    with mock_dir(dir_structure):
        import_ref = parse_import_ref(ref)
        runnable, all_usable_commands = import_and_filter(
            import_ref, base_cmd="dummy", accept_local_entrypoint=True, accept_webhook=False
        )
        print(all_usable_commands)
        assert isinstance(runnable, returned_runnable_type)
        assert len(all_usable_commands) == num_error_choices


def test_import_and_filter_2(monkeypatch, supports_on_path):
    def import_runnable(object_path, accept_local_entrypoint=False, accept_webhook=False):
        return import_and_filter(
            ImportRef("import_and_filter_source", use_module_mode=True, object_path=object_path),
            base_cmd="",
            accept_local_entrypoint=accept_local_entrypoint,
            accept_webhook=accept_webhook,
        )

    runnable, all_usable_commands = import_runnable(
        "app_with_one_web_function", accept_webhook=False, accept_local_entrypoint=True
    )
    assert runnable is None
    assert len(all_usable_commands) == 4

    assert import_runnable("app_with_one_web_function", accept_webhook=True)[0]
    assert import_runnable("app_with_one_function_one_web_endpoint", accept_webhook=False)[0]

    runnable, all_usable_commands = import_runnable("app_with_one_function_one_web_endpoint", accept_webhook=True)
    assert runnable is None
    assert len(all_usable_commands) == 7

    runnable, all_usable_commands = import_runnable("app_with_one_web_method", accept_webhook=False)
    assert runnable is None
    assert len(all_usable_commands) == 3

    assert import_runnable("app_with_one_web_method", accept_webhook=True)[0]

    assert isinstance(
        import_runnable("app_with_local_entrypoint_and_function", accept_local_entrypoint=True)[0], LocalEntrypoint
    )
    assert isinstance(
        import_runnable("app_with_local_entrypoint_and_function", accept_local_entrypoint=False)[0], Function
    )


def test_import_package_and_module_names(monkeypatch, supports_dir):
    # We try to reproduce the package/module naming standard that the `python` command line tool uses,
    # i.e. when loading using a module path (-m flag w/ python) you get a fully qualified package/module name
    # but when loading using a filename, some/mod.py it will not have a __package__

    # The biggest difference is that __name__ of the imported "entrypoint" script
    # is __main__ when using `python` but in the Modal runtime it's the name of the
    # file minus the ".py", since Modal has its own __main__
    monkeypatch.chdir(supports_dir)
    mod1 = import_file_or_module(ImportRef("assert_package", use_module_mode=True))
    assert mod1.__package__ == ""
    assert mod1.__name__ == "assert_package"

    monkeypatch.chdir(supports_dir.parent)
    with pytest.warns(PendingDeprecationError, match=r"\s-m\s"):
        # TODO: this should use use_module_mode=True once we remove the deprecation warning
        mod2 = import_file_or_module(ImportRef("test.supports.assert_package", use_module_mode=False))

    assert mod2.__package__ == "test.supports"
    assert mod2.__name__ == "test.supports.assert_package"

    mod3 = import_file_or_module(ImportRef("supports/assert_package.py", use_module_mode=False))
    assert mod3.__package__ == ""
    assert mod3.__name__ == "assert_package"


def test_invalid_source_file_exception():
    with pytest.raises(InvalidError, match="Invalid Modal source filename: 'foo.bar.py'"):
        import_file_or_module(ImportRef("path/to/foo.bar.py", use_module_mode=False))


def test_list_cli_commands():
    app = App()
    other_app = App()

    @app.function(serialized=True, name="foo")
    def foo():
        pass

    @app.cls(serialized=True)
    class Cls:
        @method()
        def method_1(self):
            pass

        @web_server(8000)
        def web_method(self):
            pass

    def non_modal_func():
        pass

    fake_module = {"app": app, "other_app": other_app, "non_modal_func": non_modal_func, "foo": foo, "Cls": Cls}

    res = list_cli_commands(fake_module)

    assert res == [
        CLICommand(["foo", "app.foo"], foo, False, priority=AutoRunPriority.MODULE_FUNCTION),  # type: ignore
        CLICommand(
            ["Cls.method_1", "app.Cls.method_1"],
            MethodReference(Cls, "method_1"),  # type: ignore
            False,
            priority=AutoRunPriority.MODULE_FUNCTION,
        ),
        CLICommand(
            ["Cls.web_method", "app.Cls.web_method"],
            MethodReference(Cls, "web_method"),  # type: ignore
            True,
            priority=AutoRunPriority.MODULE_FUNCTION,
        ),
    ]
