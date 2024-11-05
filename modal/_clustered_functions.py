from modal._utils.async_utils import synchronize_api


async def _initialize_clustered_function(cluster_id: str, cluster_rank: int, cluster_size: int):
    print(f"Initializing clustered function with cluster ID {cluster_id}, rank {cluster_rank}, size {cluster_size}")


initialize_clustered_function = synchronize_api(_initialize_clustered_function)

# Old code, for reference. TODO(nathan): Delete this once we're done migrating.
#         def get_i6pn():
#             """Returns the ipv6 address assigned to this container."""
#             return socket.getaddrinfo("i6pn.modal.local", None, socket.AF_INET6)[0][4][0]

#         hostname = socket.gethostname()
#         addr_info = get_i6pn()
#         # nccl's default host ID is $(hostname)$(cat /proc/sys/kernel/random/boot_id).
#         # on runc, if two i6pn-linked containers get scheduled on the same worker,
#         # their boot ID and hostname will both be identical, causing nccl to break.
#         # As a workaround, we can explicitly specify a unique host ID here.
#         # See MOD-4067.
#         os.environ["NCCL_HOSTID"] = f"{hostname}{addr_info}"
