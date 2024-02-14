# Copyright Modal Labs 2022

from dataclasses import dataclass
from typing import List

import psutil


@dataclass(frozen=True)
class NetworkConnection:
    remote_addr: str
    status: str


def get_open_connections() -> List[NetworkConnection]:
    open_connections:list[NetworkConnection] = []
    for kind in ["tcp", "udp"]:
        for conn in psutil.net_connections(kind=kind):
            if conn.status in ("ESTABLISHED", "CLOSING", "CLOSE_WAIT"):
                remote_address, remote_port = conn.raddr if conn.raddr else ("Unknown", "Unknown")
                open_connections.append(
                    NetworkConnection(
                        remote_addr=f"{remote_address}:{remote_port}",
                        status=conn.status,
                    ))

    return open_connections
