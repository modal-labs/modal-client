# Copyright Modal Labs 2024

import json
import logging
import os
import queue
import socket
import tempfile
import threading
import uuid
from pathlib import Path
from struct import unpack

from modal_telemetry._telemetry import MESSAGE_LEN_FORMAT, MESSAGE_LEN_LEN


class TelemetryConsumer:
    socket_filename: Path
    server: socket.socket
    connections: set[socket.socket]
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
                    if len(buffer) <= MESSAGE_LEN_LEN:
                        break
                    message_len = unpack(MESSAGE_LEN_FORMAT, buffer[0:MESSAGE_LEN_LEN])[0]
                    if len(buffer) < message_len + MESSAGE_LEN_LEN:
                        break
                    message = buffer[MESSAGE_LEN_LEN : MESSAGE_LEN_LEN + message_len]
                    buffer = buffer[MESSAGE_LEN_LEN + message_len :]
                    message = message.decode("utf-8").strip()
                    message = json.loads(message)
                    self.events.put(message)
        finally:
            self.connections.remove(conn)


def test_import_tracing(monkeypatch):
    with TelemetryConsumer() as consumer:
        monkeypatch.setenv("MODAL_TELEMETRY_SOCKET", consumer.socket_filename.absolute().as_posix())

        from modal_telemetry import _instrument  # noqa

        from .supports import module_1  # noqa
        from .supports import module_2  # noqa

        expected_messages = [
            {"event": "module_load_start", "attributes": {"name": "test.supports"}},
            {"event": "module_load_end", "attributes": {"name": "test.supports"}},
            {"event": "module_load_start", "attributes": {"name": "test.supports.module_1"}},
            {"event": "module_load_end", "attributes": {"name": "test.supports.module_1"}},
            {"event": "module_load_start", "attributes": {"name": "test.supports.module_2"}},
            {"event": "module_load_end", "attributes": {"name": "test.supports.module_2"}},
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
    from modal_telemetry import _instrument  # noqa
    import modal  # noqa


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
