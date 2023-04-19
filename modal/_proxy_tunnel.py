# Copyright Modal Labs 2022
import contextlib
import os
import subprocess
import tempfile

from modal_proto import api_pb2


@contextlib.contextmanager
def proxy_tunnel(info: api_pb2.ProxyInfo):
    if not info.elastic_ip:
        yield
        return

    with tempfile.NamedTemporaryFile(suffix=".pem") as t:
        f = open(t.name, "w")
        f.write(info.proxy_key)
        f.close()

        if info.remote_addr:
            # Forward port from address.
            args = [
                "-L",
                f"{info.remote_port}:{info.remote_addr}:{info.remote_port}",
            ]
        else:
            # Set up SOCKS proxy.
            # TODO: add a local_port column and proxy_type (this is all being rewritten anyway)
            args = ["-D", f"{info.remote_port}"]
            os.environ["HTTP_PROXY"] = f"socks5://localhost:{info.remote_port}"

            # https://github.com/getsentry/sentry-python/issues/1049
            os.environ["NO_PROXY"] = "sentry.io"

        cmd = [
            "autossh",
            "-M 0",  # don't use a monitoring port for autossh
            "-i",
            t.name,  # use pem file
            "-T",  # ignore the tty
            "-n",  # no input
            "-N",  # don't execute a command
            *args,
            f"ubuntu@{info.elastic_ip}",
            "-o",
            "StrictHostKeyChecking=no",  # avoid prompt for host
            "-o",
            "ServerAliveInterval=5",  # seconds before client sends keepalive to server
            "-o",
            "ServerAliveCountMax=1",  # number of failures before terminating ssh (autossh will restart)
        ]
        p = subprocess.Popen(cmd)

        import time

        time.sleep(3)
        try:
            yield
        finally:
            p.kill()
