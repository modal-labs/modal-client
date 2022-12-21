from datetime import datetime

import modal

stub = modal.Stub()


@stub.local_entrypoint
def func_with_args(dt: datetime):
    print(f"the day is {dt.day}")
