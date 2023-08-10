# Copyright Modal Labs 2022

import hashlib
import platform
import pytest
import time
from pathlib import Path

from modal import Image, Mount, Stub

stub = Stub()


skip_non_linux = pytest.mark.skipif(platform.system() != "Linux", reason="sandbox mock uses subprocess")


@skip_non_linux
def test_spawn_sandbox(client, servicer):
    with stub.run(client=client) as app:
        sb = app.spawn_sandbox("bash", "-c", "echo bye >&2 && sleep 1 && echo hi && exit 42", timeout=600)

        t0 = time.time()
        sb.wait()
        # Test that we actually waited for the sandbox to finish.
        assert time.time() - t0 > 0.3

        assert sb.stdout.read() == "hi\n"
        assert sb.stderr.read() == "bye\n"

        assert sb.returncode == 42


@skip_non_linux
def test_sandbox_mount(client, servicer, tmpdir):
    tmpdir.join("a.py").write(b"foo")

    with stub.run(client=client) as app:
        sb = app.spawn_sandbox(
            "echo",
            "hi",
            mounts=[Mount.from_local_dir(Path(tmpdir), remote_path="/m")],
        )
        sb.wait()

    sha = hashlib.sha256(b"foo").hexdigest()
    assert servicer.files_sha2data[sha]["data"] == b"foo"


@skip_non_linux
def test_sandbox_image(client, servicer, tmpdir):
    tmpdir.join("a.py").write(b"foo")

    with stub.run(client=client) as app:
        sb = app.spawn_sandbox("echo", "hi", image=Image.debian_slim().pip_install("foo", "bar", "potato"))
        sb.wait()

    idx = max(servicer.images.keys())
    last_image = servicer.images[idx]

    assert all(c in last_image.dockerfile_commands[-1] for c in ["foo", "bar", "potato"])
