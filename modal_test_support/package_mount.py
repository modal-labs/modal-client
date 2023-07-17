# Copyright Modal Labs 2022
from modal import Mount, Stub

stub = Stub()

mount = Mount.from_local_python_packages("module_1")


@stub.function()
def num_mounts(_x):
    return len(mount.entries)
