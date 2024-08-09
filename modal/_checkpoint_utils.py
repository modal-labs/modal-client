# Copyright Modal Labs 2022

import socket
import struct
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class NetworkConnection:
    remote_addr: str
    status: str


def get_open_connections() -> List[NetworkConnection]:
    # psutil is only supported in Linux. This function is only called inside
    # containers (via _container_io_manager.py).
    from ._vendor import psutil

    open_connections: list[NetworkConnection] = []
    for kind in ["tcp", "udp"]:
        for conn in psutil.net_connections(kind=kind):
            if conn.status in ("ESTABLISHED", "CLOSING", "CLOSE_WAIT"):
                remote_address, remote_port = conn.raddr if conn.raddr else ("Unknown", "Unknown")
                open_connections.append(
                    NetworkConnection(
                        remote_addr=f"{remote_address}:{remote_port}",
                        status=conn.status,
                    )
                )

    return open_connections


# Store the original close method
original_close = socket.socket.close


def patch_socket_close():
    """Patch all socket.close() calls with the aggressive `socket.SO_LINGER` option.
    Any unsent data is discarded and the connection is abruptly terminated with a
    RST (reset) packet instead of the normal FIN (finish) packet. This can cause
    data corruption issue and is only used in Modal to close sockets immediately
    when creating memory snapshots."""

    def new_close(self):
        try:
            self.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        except Exception:
            # Ignore any errors that might occur when setting the socket option
            pass
        original_close(self)

    socket.socket.close = new_close  # type: ignore


def unpatch_socket_close():
    socket.socket.close = original_close  # type: ignore


def is_socket_patched():
    """Function to check if the socket.close() is patched"""
    return socket.socket.close != original_close
