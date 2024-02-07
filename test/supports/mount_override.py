# Copyright Modal Labs 2023

import modal

stub = modal.Stub()
import pkg_a  # noqa


@stub.function()  # mounts=[Mount.from_local_dir(Path(__file__).parent / "pkg_a", remote_path="/root/pkg_a")])
def foo():
    pass
