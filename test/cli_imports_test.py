# Copyright Modal Labs 2023
import pytest
import sys

import click

from modal import web_endpoint
from modal.app import App, LocalEntrypoint
from modal.cli.import_refs import (
    MethodReference,
    _import_object,
    _infer_runnable,
    get_by_object_path,
    import_file_or_module,
)
from modal.exception import InvalidError
from modal.partial_function import asgi_app, method

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
        (empty_dir_with_python_file, "mod.py", App),
        (empty_dir_with_python_file, "mod.py::app", App),
        (empty_dir_with_python_file, "mod.py::other_app", App),
        (dir_containing_python_package, "pack/file.py", App),
        (dir_containing_python_package, "pack/sub/subfile.py", App),
        (dir_containing_python_package, "dir/sub/subfile.py", App),
        # # python module syntax
        (empty_dir_with_python_file, "mod", App),
        (empty_dir_with_python_file, "mod::app", App),
        (empty_dir_with_python_file, "mod::other_app", App),
        (dir_containing_python_package, "pack.mod", App),
        (dir_containing_python_package, "pack.mod::other_app", App),
        (dir_containing_python_package, "pack/local.py::app.main", LocalEntrypoint),
    ],
)
def test_import_object(dir_structure, ref, expected_object_type, mock_dir):
    with mock_dir(dir_structure):
        obj, _ = _import_object(ref, base_cmd="modal some_command")
        assert isinstance(obj, expected_object_type)


app_with_one_web_function = App()


@app_with_one_web_function.function()
@web_endpoint()
def web1():
    pass


app_with_one_function_one_web_endpoint = App()


@app_with_one_function_one_web_endpoint.function()
def f1():
    pass


@app_with_one_function_one_web_endpoint.function()
@web_endpoint()
def web2():
    pass


app_with_one_web_method = App()


@app_with_one_web_method.cls()
class C1:
    @asgi_app()
    def web_3(self):
        pass


app_with_one_web_method_one_method = App()


@app_with_one_web_method_one_method.cls()
class C2:
    @asgi_app()
    def web_4(self):
        pass

    @method()
    def f2(self):
        pass


app_with_local_entrypoint_and_function = App()


@app_with_local_entrypoint_and_function.local_entrypoint()
def le_1():
    pass


@app_with_local_entrypoint_and_function.function()
def f3():
    pass


def test_infer_object():
    this_module = sys.modules[__name__]
    with pytest.raises(click.ClickException, match="web endpoint"):
        _infer_runnable(app_with_one_web_function, this_module, accept_webhook=False)

    _, runnable = _infer_runnable(app_with_one_web_function, this_module, accept_webhook=True)
    assert runnable == web1

    _, runnable = _infer_runnable(app_with_one_function_one_web_endpoint, this_module, accept_webhook=False)
    assert runnable == f1

    with pytest.raises(click.UsageError, match="(?s)You need to specify.*\nf1\nweb2\n"):
        _, runnable = _infer_runnable(app_with_one_function_one_web_endpoint, this_module, accept_webhook=True)
    assert runnable == f1

    with pytest.raises(click.UsageError, match="web endpoint"):
        _, runnable = _infer_runnable(app_with_one_web_method, this_module, accept_webhook=False)

    _, runnable = _infer_runnable(app_with_one_web_method, this_module, accept_webhook=True)
    assert runnable == MethodReference(C1, "web_3")  # type: ignore

    _, runnable = _infer_runnable(app_with_local_entrypoint_and_function, this_module, accept_local_entrypoint=True)
    assert runnable == le_1

    _, runnable = _infer_runnable(app_with_local_entrypoint_and_function, this_module, accept_local_entrypoint=False)
    assert runnable == f3


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


def test_invalid_source_file_exception():
    with pytest.raises(InvalidError, match="Invalid Modal source filename: 'foo.bar.py'"):
        import_file_or_module("path/to/foo.bar.py")
