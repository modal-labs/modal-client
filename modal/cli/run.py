import asyncio
import functools
import inspect
import sys
import traceback

import typer
from synchronicity import Interface

from modal.cli.app import DEFAULT_STUB_NAME, _show_stub_ref_failure_help
from modal_utils.async_utils import synchronizer
from modal_utils.package_utils import NoSuchStub, import_stub, parse_stub_ref

run_cli = typer.Typer(name="run")


def _get_run_wrapper_function_handle(_stub, function_tag: str, detach: bool):
    stub = synchronizer._translate_out(_stub, Interface.BLOCKING)

    @functools.wraps(_stub._blueprint[function_tag]._info.raw_f)
    def f(*args, **kwargs):
        with stub.run(detach=detach) as app:
            function_handle = getattr(app, function_tag)
            function_handle.call(*args, **kwargs)

    return f


def _get_run_wrapper_local_entrypoint(_stub, entrypoint_name: str, detach: bool):
    stub = synchronizer._translate_out(_stub, Interface.BLOCKING)
    func = _stub._local_entrypoints[entrypoint_name]

    isasync = inspect.iscoroutinefunction(func)

    @functools.wraps(func)
    def f(*args, **kwargs):
        with stub.run(detach=detach):
            if isasync:
                asyncio.run(func(*args, **kwargs))
            else:
                func(*args, **kwargs)

    return f


def run(
    ctx: typer.Context,
    stub_ref: str = typer.Argument(
        ..., help="Path to a Python file or module, optionally identifying the name of your stub: `./main.py:mystub`."
    ),
    function_name: str = typer.Argument(None, help="Name of the Modal function to run"),
    detach: bool = typer.Option(default=False, help="Allows app to continue running if local terminal disconnects."),
):
    try:
        import_path, stub_name = parse_stub_ref(stub_ref, DEFAULT_STUB_NAME)
        stub = import_stub(import_path, stub_name)
    except NoSuchStub:
        _show_stub_ref_failure_help(import_path, stub_name)
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    _stub = synchronizer._translate_in(stub)
    function_choices = list(set(_stub.registered_functions) | set(_stub.registered_entrypoints.keys()))
    registered_functions_str = "\n".join(function_choices)
    if not function_name:
        if len(function_choices) == 1:
            function_name = function_choices[0]
        elif len(_stub.registered_entrypoints) == 1:
            function_name = list(_stub.registered_entrypoints.keys())[0]
        else:
            print(
                f"""You need to specify an entrypoint Modal function to run, e.g. `modal run app.py my_function [...args]`.
Registered functions and entrypoints on the selected stub are:
{registered_functions_str} {_stub.registered_entrypoints}
"""
            )
            exit(1)
    elif function_name not in function_choices:
        print(
            f"No function `{function_name}` could be found in the specified stub. Registered functions are: {registered_functions_str}"
        )
        exit(1)

    func_typer = typer.Typer()
    if function_name in _stub.registered_functions:
        func_typer.command(name=function_name)(_get_run_wrapper_function_handle(_stub, function_name, detach))
    else:
        func_typer.command(name=function_name)(_get_run_wrapper_local_entrypoint(_stub, function_name, detach))

    # TODO: propagate help to sub-invocation if enough arguments are available
    func_typer(args=ctx.args)
