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
import typing
from importlib import import_module
from struct import pack
from time import monotonic as time_monotonic, time as time_time
from uuid import uuid4

MODULE_LOAD_START = "module_load_start"
MODULE_LOAD_END = "module_load_end"

MESSAGE_HEADER_FORMAT = "<I"
MESSAGE_HEADER_LEN = 4


class ImportInterceptor(importlib.abc.Loader):
    __slots__ = ("loading", "tracing_socket", "events", "events_put_nowait", "loading_add", "loading_remove")

    loading: typing.Set[str]
    tracing_socket: socket.socket
    events: queue.Queue

    def __init__(self, tracing_socket: socket.socket):
        self.loading = set()
        self.loading_add = self.loading.add
        self.loading_remove = self.loading.remove
        self.tracing_socket = tracing_socket
        self.events = queue.Queue(16 * 1024)
        self.events_put_nowait = self.events.put_nowait()
        sender = threading.Thread(target=self._send, daemon=True)
        sender.start()

    def find_module(self, fullname, path=None):
        if fullname in self.loading:
            return None
        return self

    def load_module(self, fullname):
        t0 = time_monotonic()
        span_id = str(uuid4())
        self.emit(
            {"span_id": span_id, "timestamp": time_time(), "event": MODULE_LOAD_START, "attributes": {"name": fullname}}
        )
        self.loading_add(fullname)
        try:
            module = import_module(fullname)
            return module
        finally:
            self.loading_remove(fullname)
            latency = time_monotonic() - t0
            self.emit(
                {
                    "span_id": span_id,
                    "timestamp": time_time(),
                    "event": MODULE_LOAD_END,
                    "attributes": {
                        "name": fullname,
                        "latency": latency,
                    },
                }
            )

    def emit(self, event):
        try:
            self.events_put_nowait(event)
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
        try:
            _instrument_imports(socket_filename)
        except BaseException as e:
            logging.warning(f"failed to instrument imports: {e}")
