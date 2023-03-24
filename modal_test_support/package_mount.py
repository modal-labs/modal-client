# Copyright Modal Labs 2022
from modal import Mount, Stub

stub = Stub()

mounts = Mount.create_package_mounts(["module_1"])


@stub.function()
def num_mounts(_x):
    return len(mounts)
