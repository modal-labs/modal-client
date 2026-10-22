# Copyright Modal Labs 2025
import argparse
import asyncio

from modal.mount import _create_client_dependency_mounts


def main():
    parser = argparse.ArgumentParser(description="Create client dependency mounts")
    parser.add_argument(
        "--dry-run", action="store_true", default=False, help="Run in dry-run mode without making actual changes"
    )
    parser.add_argument(
        "--preview", action="store_true", default=False, help="Also publish (and overwrite) the PREVIEW mounts"
    )
    args = parser.parse_args()

    builder_versions = ["2025.06", "PREVIEW"] if args.preview else ["2025.06"]
    loop = asyncio.get_event_loop()
    loop.run_until_complete(_create_client_dependency_mounts(builder_versions=builder_versions, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
