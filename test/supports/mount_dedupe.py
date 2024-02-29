# Copyright Modal Labs 2023
import os

import modal
from modal import Mount

stub = modal.Stub()
import pkg_a  # noqa


if int(os.environ["USE_EXPLICIT"]):
    explicit_mounts1 = [Mount.from_local_python_packages("pkg_a")]  # this should be reused
    # same as above, but different instance - should be stub-deduplicated:
    explicit_mounts2 = [
        Mount.from_local_python_packages("pkg_a"),  # identical to first explicit mount and auto mounts
        Mount.from_local_python_packages("pkg_a", condition=lambda fn: "__pycache__" not in fn),  # custom condition, include normally_not_included.pyc
    ]
else:
    explicit_mounts1 = explicit_mounts2 = []  # only use automounting


@stub.function(mounts=explicit_mounts1)
def foo():
    pass


@stub.function(mounts=explicit_mounts2)
def bar():
    pass
