# Copyright Modal Labs 2022
from datetime import datetime

import modal

stub = modal.Stub()


@stub.local_entrypoint
def dt_arg(dt: datetime):
    print(f"the day is {dt.day}")


@stub.local_entrypoint
def int_arg(i: int):
    print(repr(i))


@stub.local_entrypoint
def default_arg(i: int = 10):
    print(repr(i))


@stub.local_entrypoint
def unannotated_arg(i):
    print(repr(i))
