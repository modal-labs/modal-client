# Copyright Modal Labs 2025
import asyncio

from modal.mount import _create_client_dependency_mounts


def main():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(_create_client_dependency_mounts())

if __name__ == "__main__":
    main()
