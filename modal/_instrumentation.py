import importlib
import json
import os
import queue
import sys
import threading
import time

from modal.config import logger

MODULE_LOAD_START = "module_load_start"
MODULE_LOAD_END = "module_load_end"


class ImportInterceptor(importlib.abc.Loader):
    def __init__(self, tracing_socket):
        self.loading = set()
        self.tracing_socket = tracing_socket
        self.events = queue.Queue(maxsize=1024)
        sender = threading.Thread(target=self._send, daemon=True)
        sender.start()

    def find_module(self, fullname, path=None):
        if fullname in self.loading:
            return None
        return self

    def load_module(self, fullname):
        t0 = time.monotonic()
        self.emit({"timestamp": time.time(), "event": MODULE_LOAD_START, "name": fullname})
        self.loading.add(fullname)
        try:
            module = importlib.import_module(fullname)
            return module
        finally:
            self.loading.remove(fullname)
            latency = time.monotonic() - t0
            self.emit({"timestamp": time.time(), "event": MODULE_LOAD_END, "name": fullname, "latency": latency})

    def emit(self, event):
        try:
            self.events.put(event, timeout=0)
        except queue.Full:
            logger.debug("failed to emit event: queue full")

    def _send(self):
        while True:
            event = self.events.get()
            try:
                msg = json.dumps(event).encode("utf-8")
            except BaseException as e:
                logger.debug(f"failed to serialize event: {e}")
                continue
            try:
                self.tracing_socket.send(msg)
            except OSError as e:
                logger.debug(f"failed to send event: {e}")


def instrument_imports():
    if hasattr(sys, "frozen"):
        raise Exception("unable to patch meta_path: sys is frozen")
    socket_filename = os.environ.get("MODAL_IMPORT_TRACING_SOCKET")
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
