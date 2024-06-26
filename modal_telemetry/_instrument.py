# Copyright Modal Labs 2024

import logging
import os
import sys

from modal_telemetry._telemetry import ImportInterceptor


def instrument_imports():
    if hasattr(sys, "frozen"):
        raise Exception("unable to patch meta_path: sys is frozen")
    socket_filename = os.environ.get("MODAL_TELEMETRY_SOCKET")
    if socket_filename:
        import socket

        tracing_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        tracing_socket.connect(socket_filename)
        sys.meta_path = [ImportInterceptor(tracing_socket)] + sys.meta_path  # type: ignore


def auto_instrument_imports():
    if os.environ.get("MODAL_TELEMETRY_SOCKET"):
        try:
            instrument_imports()
        except BaseException as e:
            logging.warning(f"failed to instrument imports: {e}")


auto_instrument_imports()
