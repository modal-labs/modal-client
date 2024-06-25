import json
import logging
import queue
import socket
import tempfile
import threading
from pathlib import Path


class ImportTraceConsumer:
    socket_filename: Path
    server: socket.socket
    connections: set[socket.socket]
    messages: queue.Queue
    tmp: tempfile.TemporaryDirectory

    def __init__(self):
        self.stopped = False
        self.messages = queue.Queue()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.socket_filename = Path(self.tmp.name) / "import_tracing.sock"
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
            while not self.stopped:
                try:
                    message = conn.recv(1024)
                except OSError as e:
                    logging.debug(f"connection {conn} got exception, exiting: {e}")
                    return
                message = json.loads(message.decode("utf-8"))
                logging.debug("recv: {message}")
                self.messages.put(message)
        finally:
            self.connections.remove(conn)


def test_import_tracing(monkeypatch):
    with ImportTraceConsumer() as consumer:
        monkeypatch.setenv("MODAL_IMPORT_TRACING_SOCKET", consumer.socket_filename.absolute().as_posix())

        import modal._instrumentation  # noqa

        from .supports import module_1  # noqa
        from .supports import module_2  # noqa

        expected_messages = [
            {"event": "module_load_start", "name": "test.supports"},
            {"event": "module_load_end", "name": "test.supports"},
            {"event": "module_load_start", "name": "test.supports.module_1"},
            {"event": "module_load_end", "name": "test.supports.module_1"},
            {"event": "module_load_start", "name": "test.supports.module_2"},
            {"event": "module_load_end", "name": "test.supports.module_2"},
        ]

        for expected_message in expected_messages:
            m = consumer.messages.get(timeout=30)
            assert m == m | expected_message
            assert m["timestamp"] >= 0
            if m["event"] == "module_load_end":
                assert m["latency"] >= 0
