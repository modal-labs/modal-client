# Copyright Modal Labs 2023
import asyncio
import os
import shutil
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple, Union, overload

from click import UsageError

from modal.network_file_system import _NetworkFileSystem
from modal.volume import _Volume
from modal_proto import api_pb2

_Entry = Union[api_pb2.SharedVolumeListFilesEntry, api_pb2.VolumeListFilesEntry]


@overload
def _glob_download(
    volume: _Volume,
    is_file_fn: Callable[[api_pb2.VolumeListFilesEntry], bool],
    remote_glob_path: str,
    local_destination: Path,
    overwrite: bool,
):
    ...


@overload
def _glob_download(
    volume: _NetworkFileSystem,
    is_file_fn: Callable[[api_pb2.SharedVolumeListFilesEntry], bool],
    remote_glob_path: str,
    local_destination: Path,
    overwrite: bool,
):
    ...


async def _glob_download(
    volume,
    is_file_fn,
    remote_glob_path: str,
    local_destination: Path,
    overwrite: bool,
):
    q: asyncio.Queue[Tuple[Optional[Path], Optional[_Entry]]] = asyncio.Queue()
    num_consumers = 10  # concurrency limit

    async def producer():
        async for entry in volume.iterdir(remote_glob_path):
            output_path = local_destination / entry.path
            if output_path.exists():
                if overwrite:
                    if is_file_fn(entry):
                        os.remove(output_path)
                    else:
                        shutil.rmtree(output_path)
                else:
                    raise UsageError(
                        f"Output path '{output_path}' already exists. Use --force to overwrite the output directory"
                    )
            await q.put((output_path, entry))
        # No more entries to process; issue one shutdown message for each consumer.
        for _ in range(num_consumers):
            await q.put((None, None))

    async def consumer():
        while True:
            output_path, entry = await q.get()
            if output_path is None:
                return
            try:
                if is_file_fn(entry):
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with output_path.open("wb") as fp:
                        b = 0
                        async for chunk in volume.read_file(entry.path):
                            b += fp.write(chunk)

                    print(f"Wrote {b} bytes to {output_path}", file=sys.stderr)
            finally:
                q.task_done()

    consumers = [consumer() for _ in range(num_consumers)]
    await asyncio.gather(producer(), *consumers)
