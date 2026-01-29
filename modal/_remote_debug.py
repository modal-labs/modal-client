import os
import pdb  # noqa: T100
import queue
import select
import sys
import termios
import threading
import time
import tty
from functools import wraps
from typing import Callable

# PART 1: REMOTE CONTAINER SIDE


class IOQueueWrapper:
    """File-like interface for Pdb using queues."""

    def __init__(self, in_q, out_q):
        self.in_q, self.out_q, self._buf = in_q, out_q, ""

    def write(self, data):
        self.out_q.put({"type": "stdout", "data": data})
        return len(data)

    def flush(self):
        pass

    def readline(self):
        while "\n" not in self._buf:
            try:
                msg = self.in_q.get(timeout=0.1)
                if msg["type"] == "stdin":
                    self._buf += msg["data"]
                elif msg["type"] == "interrupt":
                    raise KeyboardInterrupt()
            except queue.Empty:
                continue
        line, self._buf = self._buf.split("\n", 1)
        return line + "\n"

    def isatty(self):
        return True


class RemotePdb(pdb.Pdb):
    """Custom Pdb for network I/O."""

    def __init__(self, in_q, out_q):
        self.in_q, self.out_q = in_q, out_q
        io = IOQueueWrapper(in_q, out_q)
        super().__init__(stdin=io, stdout=io, skip=["modal.*"])

    def interaction(self, frame, tb):
        self.out_q.put(
            {
                "type": "debug_start",
                "frame": {"file": frame.f_code.co_filename, "line": frame.f_lineno, "func": frame.f_code.co_name},
            }
        )
        try:
            super().interaction(frame, tb)
        finally:
            self.out_q.put({"type": "debug_end"})


class DebugTunnel:
    """Manages debug I/O and heartbeat."""

    def __init__(self, stream):
        self.stream = stream
        self.in_q, self.out_q = queue.Queue(), queue.Queue()
        self._run = False
        self._orig_hook = sys.breakpointhook

    def start(self):
        self._run = True
        threading.Thread(target=self._io_loop, daemon=True).start()
        threading.Thread(target=self._heartbeat, daemon=True).start()
        sys.breakpointhook = lambda: RemotePdb(self.in_q, self.out_q).set_trace(sys._getframe(1))

    def stop(self):
        self._run = False
        sys.breakpointhook = self._orig_hook

    def _io_loop(self):
        while self._run:
            try:
                msg = self.out_q.get(timeout=0.05)
                self.stream.send({"debug_msg": msg})
            except queue.Empty:
                pass
            try:
                for r in self.stream.receive(timeout=0.05):
                    if "debug_in" in r:
                        self.in_q.put(r["debug_in"])
            except Exception:
                pass

    def _heartbeat(self):
        while self._run:
            try:
                self.stream.send({"heartbeat": {"ts": time.time(), "status": "debug"}})
            except Exception:
                pass
            time.sleep(2.0)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
        return False


# PART 2: LOCAL CLIENT SIDE


class TerminalManager:
    """Handles terminal raw mode and keystroke forwarding."""

    def __init__(self, send_fn: Callable):
        self.send = send_fn
        self._orig_term = None
        self._debug_mode = False
        self._run = False

    def enter_debug(self, frame):
        if self._debug_mode:
            return
        self._debug_mode = True
        try:
            self._orig_term = termios.tcgetattr(sys.stdin.fileno())
            tty.setraw(sys.stdin.fileno())
        except Exception:
            return
        self._run = True
        threading.Thread(target=self._forward_input, daemon=True).start()

    def exit_debug(self):
        if not self._debug_mode:
            return
        self._run = False
        self._debug_mode = False
        if self._orig_term:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._orig_term)
            except Exception:
                pass

    def _forward_input(self):
        while self._run:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                try:
                    ch = sys.stdin.read(1)
                    self.send({"type": "interrupt" if ord(ch) == 3 else "stdin", "data": ch})
                except Exception:
                    break

    def output(self, data):
        sys.stdout.write(data.replace("\n", "\r\n") if self._debug_mode else data)
        sys.stdout.flush()

    def cleanup(self):
        if self._orig_term:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._orig_term)
            except Exception:
                pass


class DebugClient:
    """Processes debug messages from remote."""

    def __init__(self, term: TerminalManager):
        self.term = term

    def handle(self, msg):
        t = msg.get("type")
        if t == "debug_start":
            self.term.enter_debug(msg.get("frame", {}))
        elif t == "debug_end":
            self.term.exit_debug()
        elif t == "stdout":
            self.term.output(msg.get("data", ""))


# PART 3: MODAL INTEGRATION


def enable_remote_debugging(func):
    """Decorator to enable debugging for Modal functions."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if os.environ.get("MODAL_INTERACTIVE_DEBUG") != "1":
            return func(*args, **kwargs)

        # Get Modal's gRPC stream (Modal engineers: wire this up properly)
        try:
            from modal._runtime import current_function_call

            stream = current_function_call().grpc_stream
        except Exception:
            stream = None

        if not stream:
            return func(*args, **kwargs)

        with DebugTunnel(stream):
            return func(*args, **kwargs)

    return wrapper


class DebugSession:
    """Context manager for entire Modal app debug sessions."""

    def __init__(self, app):
        self.app = app
        self.term = None
        self.client = None

    def __enter__(self):
        if os.environ.get("MODAL_INTERACTIVE_DEBUG") != "1":
            return self

        stream = self.app._client.stub  # Modal-specific

        self.term = TerminalManager(lambda msg: stream.send({"debug_in": msg}))
        self.client = DebugClient(self.term)

        # Hook into Modal's message processing
        orig_proc = self.app._process_message

        def patched(msg):
            if "debug_msg" in msg:
                self.client.handle(msg["debug_msg"])
            else:
                orig_proc(msg)

        self.app._process_message = patched

        return self

    def __exit__(self, *_):
        if self.term:
            self.term.cleanup()
        return False


def patch_modal_cli():
    """Call this from Modal's CLI entry point when --interactive flag is used."""
    if "--interactive" in sys.argv or "-i" in sys.argv:
        os.environ["MODAL_INTERACTIVE_DEBUG"] = "1"
