# Copyright Modal Labs 2024

import json
import logging
import os
import queue
import socket
import sys
import tempfile
import threading
import typing
import uuid
from pathlib import Path
from struct import unpack

from modal._telemetry import MESSAGE_HEADER_FORMAT, MESSAGE_HEADER_LEN


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
    # Delete the `modal._instrument` module in case it has already been imported by other test runs
    sys.modules.pop("modal._instrument", None)

    with TelemetryConsumer() as consumer:
        monkeypatch.setenv("MODAL_TELEMETRY_SOCKET", consumer.socket_filename.absolute().as_posix())

        from modal import _instrument  # noqa

        from .telemetry import tracing_module_1  # noqa

        expected_messages = [
            {"event": "module_load_start", "attributes": {"name": "test.telemetry"}},
            {"event": "module_load_end", "attributes": {"name": "test.telemetry"}},
            {"event": "module_load_start", "attributes": {"name": "test.telemetry.tracing_module_1"}},
            {"event": "module_load_start", "attributes": {"name": "test.telemetry.tracing_module_2"}},
            {"event": "module_load_end", "attributes": {"name": "test.telemetry.tracing_module_2"}},
            {"event": "module_load_end", "attributes": {"name": "test.telemetry.tracing_module_1"}},
        ]

        for expected_message in expected_messages:
            m = consumer.events.get(timeout=30)
            assert m["event"] == expected_message["event"]
            assert m["attributes"] == m["attributes"] | expected_message["attributes"]
            assert m["timestamp"] >= 0
            assert uuid.UUID(m["span_id"])
            if m["event"] == "module_load_end":
                assert m["attributes"]["latency"] >= 0


def generate_modal_import_telemetry():
    from modal import _instrument  # noqa
    import kubernetes  # noqa


def main():
    if "MODAL_TELEMETRY_SOCKET" in os.environ:
        generate_modal_import_telemetry()
    else:
        with TelemetryConsumer() as consumer:
            os.environ["MODAL_TELEMETRY_SOCKET"] = consumer.socket_filename.absolute().as_posix()

            generate_modal_import_telemetry()

            while True:
                try:
                    m = consumer.events.get_nowait()
                    print(m)
                except queue.Empty:
                    break


if __name__ == "__main__":
    main()
