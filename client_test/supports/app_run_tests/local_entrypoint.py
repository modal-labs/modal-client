# Copyright Modal Labs 2022
import datetime

import modal

stub = modal.Stub()


@stub.function
def foo():
    pass


@stub.local_entrypoint
def main():
    print("called locally")
    foo.call()
    foo.call()


@stub.local_entrypoint
def other(dt: datetime.datetime):
    print(f"the day is {dt.day}")
