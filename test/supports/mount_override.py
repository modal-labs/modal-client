# Copyright Modal Labs 2023
from pathlib import Path

import modal
from modal import Mount

stub = modal.Stub()
import pkg_a  # noqa


@stub.function()
def no_override():
    pass


# pkg is auto-mounted, but if user adds their own version of it, it should have precedence
@stub.function(mounts=[Mount.from_local_dir(Path(__file__).parent / "pkg_a", remote_path="/root/pkg_a")])
def with_override():
    pass
