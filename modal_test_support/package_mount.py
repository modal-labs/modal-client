# Copyright Modal Labs 2022
from modal import Mount, Stub

stub = Stub()

mount = Mount.from_local_python_packages("module_1")


@stub.function(mounts=[mount])
def num_mounts(_x):
    pass  # return len(mount.entries)
