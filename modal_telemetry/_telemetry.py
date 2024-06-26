# Copyright Modal Labs 2024

import importlib
import importlib.abc
import json
import logging
import queue
import threading
import time
import uuid
from struct import pack

MODULE_LOAD_START = "module_load_start"
MODULE_LOAD_END = "module_load_end"

MESSAGE_LEN_FORMAT = "<I"
MESSAGE_LEN_LEN = 4


class ImportInterceptor(importlib.abc.Loader):
    def __init__(self, tracing_socket):
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
                encoded_len = pack(MESSAGE_LEN_FORMAT, len(msg))
                self.tracing_socket.send(encoded_len + msg)
            except OSError as e:
                logging.debug(f"failed to send event: {e}")
