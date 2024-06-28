# Copyright Modal Labs 2024

import json
import logging
import os
import pytest
import queue
import socket
import sys
import tempfile
import threading
import time
import typing
import uuid
from pathlib import Path
from struct import unpack

from modal._telemetry import (
    MESSAGE_HEADER_FORMAT,
    MESSAGE_HEADER_LEN,
    ImportInterceptor,
    instrument_imports,
    supported_platform,
)


class TelemetryConsumer:
    socket_filename: Path
    server: socket.socket
    connections: typing.Set[socket.socket]
    events: queue.Queue
    tmp: tempfile.TemporaryDirectory

    def __init__(self):
        self.stopped = False
        self.events = queue.Queue()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.socket_filename = Path(self.tmp.name) / "telemetry.sock"
        self.connections = set()
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(self.socket_filename.as_posix())
        self.server.listen()
        listener = threading.Thread(target=self._listen, daemon=True)
        listener.start()

    def stop(self):
        self.stopped = True
        self.server.close()
        for conn in list(self.connections):
            conn.close()

    def _listen(self):
        while not self.stopped:
            try:
                conn, _ = self.server.accept()
                receiver = threading.Thread(target=self._recv, args=(conn,), daemon=True)
                receiver.start()
                self.connections.add(conn)
            except OSError as e:
                logging.debug(f"listener got exception, exiting: {e}")
                return

    def _recv(self, conn):
        try:
            buffer = bytearray()
            while not self.stopped:
                try:
                    data = conn.recv(1024)
                except OSError as e:
                    logging.debug(f"connection {conn} got exception, exiting: {e}")
                    return
                buffer.extend(data)
                while True:
                    if len(buffer) <= MESSAGE_HEADER_LEN:
                        break
                    message_len = unpack(MESSAGE_HEADER_FORMAT, buffer[0:MESSAGE_HEADER_LEN])[0]
                    if len(buffer) < message_len + MESSAGE_HEADER_LEN:
                        break
                    message_bytes = buffer[MESSAGE_HEADER_LEN : MESSAGE_HEADER_LEN + message_len]
                    buffer = buffer[MESSAGE_HEADER_LEN + message_len :]
                    message = message_bytes.decode("utf-8").strip()
                    message = json.loads(message)
                    self.events.put(message)
        finally:
            self.connections.remove(conn)


def test_import_tracing(monkeypatch):
    if not supported_platform():
        pytest.skip(f"unsupported platform: {sys.platform}")

    with TelemetryConsumer() as consumer, ImportInterceptor.connect(consumer.socket_filename.absolute().as_posix()):
        from .telemetry import tracing_module_1  # noqa

        expected_messages: list[typing.Dict[str, typing.Any]] = [
            {"event": "module_load_start", "attributes": {"name": "test.telemetry.tracing_module_1"}},
            {"event": "module_load_start", "attributes": {"name": "test.telemetry.tracing_module_2"}},
            {"event": "module_load_end", "attributes": {"name": "test.telemetry.tracing_module_2"}},
            {"event": "module_load_end", "attributes": {"name": "test.telemetry.tracing_module_1"}},
        ]

        for expected_message in expected_messages:
            m = consumer.events.get(timeout=30)
            # skip this test module - behavior seems to vary depending on timing and maybe python version etc
            while m["attributes"]["name"] == "test.telemetry":
                m = consumer.events.get(timeout=30)
            assert m["event"] == expected_message["event"]
            assert m["attributes"]["name"] == expected_message["attributes"]["name"]
            assert m["timestamp"] >= 0
            assert uuid.UUID(m["span_id"])
            if m["event"] == "module_load_end":
                assert m["attributes"]["latency"] >= 0


# For manual testing
def generate_import_telemetry(telemetry_socket):
    instrument_imports(telemetry_socket)
    t0 = time.monotonic()
    import kubernetes  # noqa

    return time.monotonic() - t0


# For manual testing
def main():
    telemetry_socket = os.environ.get("MODAL_TELEMETRY_SOCKET")
    if telemetry_socket:
        latency = generate_import_telemetry(telemetry_socket)
    else:
        with TelemetryConsumer() as consumer:
            latency = generate_import_telemetry(consumer.socket_filename.absolute().as_posix())

            while True:
                try:
                    m = consumer.events.get_nowait()
                    print(m)
                except queue.Empty:
                    break

    print(f"import kubernetes took {latency:.02}s")


if __name__ == "__main__":
    main()
