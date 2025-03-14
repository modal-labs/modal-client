# Copyright Modal Labs 2022
import ast
import pytest
import traceback
from typing import Any

from pytest_markdown_docs._runners import DefaultRunner, register_runner
from pytest_markdown_docs.definitions import FenceTestDefinition

import modal
from modal import enable_output
from modal.app import LocalEntrypoint
from modal.cli.import_refs import infer_runnable, list_cli_commands
from modal.functions import Function


def pytest_markdown_docs_globals():
    import math

    return {
        "modal": modal,
        "app": modal.App("pytest-markdown-docs-app"),
        "math": math,
        "__name__": "runtest",
        "fastapi_endpoint": modal.fastapi_endpoint,
        "asgi_app": modal.asgi_app,
        "wsgi_app": modal.wsgi_app,
        "__file__": "xyz.py",
    }


@pytest.fixture(autouse=True)
def disable_auto_mount(monkeypatch):
    monkeypatch.setenv("MODAL_AUTOMOUNT", "0")
    yield


@register_runner()
class ModalRunner(DefaultRunner):
    def runtest(self, test, args):
        try:
            module_name = "markdown_code_fence.py"
            try:
                tree = ast.parse(test.source, filename=module_name)
            except SyntaxError:
                raise

            try:
                # if we don't compile the code, it seems we get name lookup errors
                # for functions etc. when doing cross-calls across inline functions
                compiled = compile(tree, filename=module_name, mode="exec", dont_inherit=True)
            except SyntaxError:
                raise

            exec_globals = args.copy()
            exec_locals: dict[str, Any] = {}  # this will contain anything defined in the fence, like @app.function
            exec(compiled, exec_globals, exec_locals)

            # TODO (elias): add support for runnable "arguments" in pytest-markdown-docs so the code fence
            #  can specify which function to run and what to pass as arguments to it

            # for now, use "modal run" logic:
            cli_commands = list_cli_commands(exec_locals)

            # all commands that satisfy local entrypoint/accept webhook limitations AND object path prefix
            runnable = infer_runnable(cli_commands, "", True, False)

            with enable_output():
                if isinstance(runnable, LocalEntrypoint):
                    with runnable.app.run():
                        runnable()
                elif isinstance(runnable, Function):
                    with runnable.app.run():
                        runnable.remote()
                else:
                    with runnable.cls.app.run():
                        getattr(runnable.cls(), runnable.method_name).remote()
        except:
            traceback.print_exc()
            raise

    def repr_failure(self, test: FenceTestDefinition, excinfo: pytest.ExceptionInfo[BaseException], style=None):
        return "Error during app run, see logs"
