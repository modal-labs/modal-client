from modal._utils.async_utils import synchronize_api


async def _initialize_clustered_function(cluster_id: str, cluster_rank: int, cluster_size: int):
    print(f"Initializing clustered function with cluster ID {cluster_id}, rank {cluster_rank}, size {cluster_size}")


initialize_clustered_function = synchronize_api(_initialize_clustered_function)

# def _networked(func):
#     """This function handles i6pn address sharing between the main container and its peers.

#     It pops modal_rank, modal_size, and modal_queue from the kwargs of the function.
#     The container with a modal_rank of 0 is the main container and is responsible for sharing its i6pn address
#     through a modal queue.
#     All containers will block until they are able to acquire the i6pn address from the ephemeral modal queue.
#     This behavior ensures that on entry to the wrapped function, appropriate environment variables will have
#     been set for communication between containers.
#     """

#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         import os
#         import socket

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

#         rank = kwargs.pop("modal_rank", None)
#         size = kwargs.pop("modal_size", None)
#         q = kwargs.pop("modal_q", None)

#         if rank is None or size is None or q is None:
#             raise ValueError("Missing required arguments; `_networked` must be called using `grouped` decorator")
#         elif rank == 0:
#             q.put_many([addr_info for _ in range(size)])
#         main_ip = q.get()
#         assert main_ip is not None, "Failed to get main i6pn address"

#         os.environ["MODAL_MAIN_I6PN"] = f"{main_ip}"
#         os.environ["MODAL_WORLD_SIZE"] = f"{size}"
#         os.environ["MODAL_CONTAINER_RANK"] = f"{rank}"

#         return func(*args, **kwargs)

#     return wrapper
