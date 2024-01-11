# Copyright Modal Labs 2022
from modal import Mount, Stub

stub = Stub()


@stub.function()
def num_mounts(_x):
    mount = Mount.from_local_python_packages("module_1")
    return len(mount.entries)
