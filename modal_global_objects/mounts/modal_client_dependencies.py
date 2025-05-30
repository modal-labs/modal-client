# Copyright Modal Labs 2025
import asyncio
import subprocess
import tempfile

from modal.client import Client
from modal.config import config

from modal.image import ImageBuilderVersion, _get_modal_requirements_path
from modal.mount import _create_client_dependency_mounts
from modal_proto import api_pb2


def main():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(_create_client_dependency_mounts())

if __name__ == "__main__":
    main()
