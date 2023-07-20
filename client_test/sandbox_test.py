# Copyright Modal Labs 2022

import time

from modal import Image, Stub

stub = Stub()


def test_spawn_sandbox(client, servicer):
    with stub.run(client=client) as app:
        sb = app.spawn_sandbox("bash", "-c", "sleep 1 && echo hi", image=Image.debian_slim().pip_install("pandas"))

        t0 = time.time()
        sb.wait()
        # We actually waited for the sandbox to finish.
        assert time.time() - t0 > 0.3

        assert sb.stdout.read() == "hi\n"
