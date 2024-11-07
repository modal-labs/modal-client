# Copyright Modal Labs 2024
import modal
import modal.experimental
from modal._utils.async_utils import synchronize_api
from modal.functions import _Function
from modal.queue import _Queue

app = modal.App(name="test")


class GroupInfo:
    def __init__(self, size: int, rank: int, q: modal.Queue):
        self.size = size
        self.rank = rank
        self.q = q

    def init(self):
        import os
        import socket

        def get_i6pn():
            """Returns the ipv6 address assigned to this container."""
            return socket.getaddrinfo("i6pn.modal.local", None, socket.AF_INET6)[0][4][0]

        hostname = socket.gethostname()
        addr_info = get_i6pn()
        # nccl's default host ID is $(hostname)$(cat /proc/sys/kernel/random/boot_id).
        # on runc, if two i6pn-linked containers get scheduled on the same worker,
        # their boot ID and hostname will both be identical, causing nccl to break.
        # As a workaround, we can explicitly specify a unique host ID here.
        # See MOD-4067.
        os.environ["NCCL_HOSTID"] = f"{hostname}{addr_info}"

        if self.rank == 0:
            self.q.put_many([addr_info for _ in range(self.size)])
        main_ip = self.q.get()
        assert main_ip is not None, "Failed to get main i6pn address"

        return self.rank, self.size, main_ip


async def _launch_grouped_function(f: _Function):
    handles = []
    async with _Queue.ephemeral() as q:
        for i in range(f.cluster_size):
            handles.append(await f.spawn(group_info=GroupInfo(f.cluster_size, i, synchronize_api(q))))
    return [await handle.get() for handle in handles]


launch_grouped_function = synchronize_api(_launch_grouped_function)


@app.function()
@modal.experimental.grouped(size=2)
def f(group_info: GroupInfo):
    rank, world_size, main_i6pn = group_info.init()

    print(f"Hello from {rank}/{world_size} " f"({main_i6pn})")

    return rank


@app.local_entrypoint()
def main():
    print(launch_grouped_function(f))
