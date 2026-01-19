# Copyright Modal Labs 2022
import typing
from datetime import datetime
from typing import Optional

from modal import App, method

app = App()


@app.local_entrypoint()
def dt_arg(dt: datetime):
    print(f"the day is {dt.day}")


@app.local_entrypoint()
def int_arg(i: int):
    print(repr(i), type(i))


@app.local_entrypoint()
def default_arg(i: int = 10):
    print(repr(i), type(i))


@app.local_entrypoint()
def unannotated_arg(i):
    print(repr(i), type(i))


@app.local_entrypoint()
def unannotated_default_arg(i=10):
    print(repr(i), type(i))


@app.function()
def int_arg_fn(i: int):
    print(repr(i), type(i))


@app.cls()
class ALifecycle:
    @method()
    def some_method(self, i):
        print(repr(i), type(i))

    @method()
    def some_method_int(self, i: int):
        print(repr(i), type(i))


@app.local_entrypoint()
def optional_arg(i: Optional[int] = None):
    print(repr(i), type(i))


@app.local_entrypoint()
def optional_arg_pep604(i: "int | None" = None):
    print(repr(i), type(i))


@app.local_entrypoint()
def optional_arg_postponed(i: "Optional[int]" = None):
    print(repr(i), type(i))


@app.function()
def optional_arg_fn(i: Optional[int] = None):
    print(repr(i), type(i))


@app.local_entrypoint()
def unparseable_annot(i: typing.Union[int, str]):
    pass


@app.local_entrypoint()
def unevaluatable_annot(i: "no go"):  # type: ignore  # noqa
    pass


@app.local_entrypoint()
def literal_str_arg(mode: typing.Literal["read", "write", "append"]):
    print(repr(mode), type(mode))


@app.local_entrypoint()
def literal_int_arg(level: typing.Literal[1, 2, 3]):
    print(repr(level), type(level))


@app.local_entrypoint()
def literal_bool_arg(val: typing.Literal[True, False]):
    """This should fail - booleans are not supported."""
    print(repr(val), type(val))


@app.local_entrypoint()
def literal_with_default(mode: typing.Literal["dev", "prod"] = "dev"):
    print(repr(mode), type(mode))


@app.local_entrypoint()
def literal_int_with_default(level: typing.Literal[1, 2, 3] = 2):
    print(repr(level), type(level))


@app.function()
def literal_arg_fn(level: typing.Literal[1, 2, 3]):
    print(repr(level), type(level))


@app.local_entrypoint()
def literal_ambiguous_arg(val: typing.Literal["2", 2]):
    """This should fail - mixed types (str and int) not supported."""
    print(repr(val), type(val))
