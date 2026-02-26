# Copyright Modal Labs 2025
import argparse
import asyncio

from modal.mount import _create_client_dependency_mounts


def main():
    parser = argparse.ArgumentParser(description="Create client dependency mounts")
    parser.add_argument(
        "--dry-run", action="store_true", default=False, help="Run in dry-run mode without making actual changes"
    )
    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(_create_client_dependency_mounts(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
