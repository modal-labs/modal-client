# Copyright Modal Labs 2022
from modal import Stub, create_package_mounts

stub = Stub()

mounts = create_package_mounts(["module_1"])


@stub.function()
def num_mounts(_x):
    return len(mounts)
