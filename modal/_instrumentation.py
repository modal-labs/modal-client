import importlib
import os  # noqa: I001
import sys
import time

from modal.config import logger


class ImportInterceptor(importlib.abc.Loader):
    def __init__(self, tracing_socket):
        self.loading = set()
        self.tracing_socket = tracing_socket

    def find_module(self, fullname, path=None):
        if fullname in self.loading:
            return None
        return self

    def load_module(self, fullname):
        t0 = time.monotonic()
        self.emit(f"loading module {fullname}\n")
        self.loading.add(fullname)
        try:
            module = importlib.import_module(fullname)
            return module
        finally:
            self.loading.remove(fullname)
            latency = time.monotonic() - t0
            self.emit(f"loaded module {fullname}: {latency=}\n")

    def emit(self, message):
        # TODO(dano): send a separate thread to avoid blocking
        # TODO(dano): send structured json
        if self.tracing_socket:
            try:
                self.tracing_socket.send(message.encode("utf-8"))
            except BaseException as e:
                logger.debug(f"failed to send import trace: {e}")


def instrument_imports():
    if hasattr(sys, "frozen"):
        raise Exception("unable to patch meta_path: sys is frozen")
    socket_filename = os.environ.get("MODAL_IMPORT_TRACING_SOCKET")
    tracing_socket = None
    if socket_filename:
        import socket

        tracing_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        tracing_socket.connect(socket_filename)
    sys.meta_path = [ImportInterceptor(tracing_socket)] + sys.meta_path


def auto_instrument_imports():
    if os.environ.get("MODAL_IMPORT_TRACING_SOCKET"):
        try:
            instrument_imports()
        except BaseException as e:
            print(f"failed to instrument imports: {e}", file=sys.stderr)


auto_instrument_imports()
