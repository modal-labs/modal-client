# Copyright Modal Labs 2024

import importlib
import importlib.abc
import json
import logging
import os
import queue
import socket
import sys
import threading
import time
import typing
import uuid
from struct import pack

MODULE_LOAD_START = "module_load_start"
MODULE_LOAD_END = "module_load_end"

MESSAGE_HEADER_FORMAT = "<I"
MESSAGE_HEADER_LEN = 4


class ImportInterceptor(importlib.abc.Loader):
    loading: typing.Set[str]
    tracing_socket: socket.socket
    events: queue.Queue

    def __init__(self, tracing_socket: socket.socket):
        self.loading = set()
        self.tracing_socket = tracing_socket
        self.events = queue.Queue(maxsize=16 * 1024)
        sender = threading.Thread(target=self._send, daemon=True)
        sender.start()

    def find_module(self, fullname, path=None):
        if fullname in self.loading:
            return None
        return self

    def load_module(self, fullname):
        t0 = time.monotonic()
        span_id = str(uuid.uuid4())
        self.emit(
            {"span_id": span_id, "timestamp": time.time(), "event": MODULE_LOAD_START, "attributes": {"name": fullname}}
        )
        self.loading.add(fullname)
        try:
            module = importlib.import_module(fullname)
            return module
        finally:
            self.loading.remove(fullname)
            latency = time.monotonic() - t0
            self.emit(
                {
                    "span_id": span_id,
                    "timestamp": time.time(),
                    "event": MODULE_LOAD_END,
                    "attributes": {
                        "name": fullname,
                        "latency": latency,
                    },
                }
            )

    def emit(self, event):
        try:
            self.events.put_nowait(event)
        except queue.Full:
            logging.debug("failed to emit event: queue full")

    def _send(self):
        while True:
            event = self.events.get()
            try:
                msg = json.dumps(event).encode("utf-8")
            except BaseException as e:
                logging.debug(f"failed to serialize event: {e}")
                continue
            try:
                encoded_len = pack(MESSAGE_HEADER_FORMAT, len(msg))
                self.tracing_socket.send(encoded_len + msg)
            except OSError as e:
                logging.debug(f"failed to send event: {e}")


def _instrument_imports(socket_filename: str):
    if hasattr(sys, "frozen"):
        raise Exception("unable to patch meta_path: sys is frozen")
    tracing_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    tracing_socket.connect(socket_filename)
    sys.meta_path = [ImportInterceptor(tracing_socket)] + sys.meta_path  # type: ignore


def instrument_imports():
    socket_filename = os.environ.get("MODAL_TELEMETRY_SOCKET")
    if socket_filename:
        if not supported_python_version():
            logging.debug("unsupported python version, not instrumenting imports")
            return
        try:
            _instrument_imports(socket_filename)
        except BaseException as e:
            logging.warning(f"failed to instrument imports: {e}")


def supported_python_version():
    # TODO(dano): support python 3.12
    return sys.version_info[0] == 3 and sys.version_info[1] <= 11
