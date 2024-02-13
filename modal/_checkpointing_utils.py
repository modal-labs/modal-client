import psutil

from dataclasses import dataclass

@dataclass(frozen=True)
class NetworkConnection:
    remote_addr: str
    status: str


def get_open_connections() -> list[NetworkConnection]:
    open_connections:list[NetworkConnection] = []
    for kind in ["tcp", "udp"]:
        for conn in psutil.net_connections(kind=kind):
            if conn.status == "ESTABLISHED":
                remote_address, remote_port = conn.raddr if conn.raddr else ("Unknown", "Unknown")
                open_connections.append(
                    NetworkConnection(
                        remote_addr=f"{remote_address}:{remote_port}",
                        status=conn.status,
                    ))
        
    return open_connections