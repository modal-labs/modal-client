# Copyright Modal Labs 2022
from datetime import datetime
from typing import Optional

from modal import Stub, method

stub = Stub()


@stub.local_entrypoint()
def dt_arg(dt: datetime):
    print(f"the day is {dt.day}")


@stub.local_entrypoint()
def int_arg(i: int):
    print(repr(i), type(i))


@stub.local_entrypoint()
def default_arg(i: int = 10):
    print(repr(i), type(i))


@stub.local_entrypoint()
def unannotated_arg(i):
    print(repr(i), type(i))


@stub.local_entrypoint()
def unannotated_default_arg(i=10):
    print(repr(i), type(i))


@stub.function()
def int_arg_fn(i: int):
    print(repr(i), type(i))


@stub.cls()
class ALifecycle:
    @method()
    def some_method(self, i):
        print(repr(i), type(i))

    @method()
    def some_method_int(self, i: int):
        print(repr(i), type(i))


@stub.local_entrypoint()
def optional_arg(i: Optional[int] = None):
    print(repr(i), type(i))


@stub.local_entrypoint()
def optional_arg_pep604(i: "int | None" = None):
    print(repr(i), type(i))
