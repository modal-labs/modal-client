# Copyright Modal Labs 2023
import asyncio
import os
import shutil
import sys
from pathlib import Path
from typing import AsyncIterator, Optional, Tuple, Union

from click import UsageError

from modal._utils.async_utils import TaskContext
from modal.network_file_system import _NetworkFileSystem
from modal.volume import FileEntry, FileEntryType, _Volume

PIPE_PATH = Path("-")


async def _volume_download(
    volume: Union[_NetworkFileSystem, _Volume],
    remote_path: str,
    local_destination: Path,
    overwrite: bool,
):
    is_pipe = local_destination == PIPE_PATH

    q: asyncio.Queue[Tuple[Optional[Path], Optional[FileEntry]]] = asyncio.Queue()
    num_consumers = 1 if is_pipe else 10  # concurrency limit for downloading files

    async def producer():
        iterator: AsyncIterator[FileEntry]
        if isinstance(volume, _Volume):
            iterator = volume.iterdir(remote_path, recursive=True)
        else:
            iterator = volume.iterdir(remote_path)  # NFS still supports "glob" paths

        async for entry in iterator:
            if is_pipe:
                await q.put((None, entry))
            else:
                start_path = os.path.dirname(remote_path).split("*")[0]
                rel_path = Path(entry.path).relative_to(start_path.lstrip("/"))
                output_path = local_destination / rel_path
                if output_path.exists():
                    if overwrite:
                        if output_path.is_file():
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
            if entry is None:
                return
            try:
                if is_pipe:
                    if entry.type == FileEntryType.FILE:
                        async for chunk in volume.read_file(entry.path):
                            sys.stdout.buffer.write(chunk)
                else:
                    if entry.type == FileEntryType.FILE:
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        with output_path.open("wb") as fp:
                            b = 0
                            async for chunk in volume.read_file(entry.path):
                                b += fp.write(chunk)
                        print(f"Wrote {b} bytes to {output_path}", file=sys.stderr)
                    elif entry.type == FileEntryType.DIRECTORY:
                        output_path.mkdir(parents=True, exist_ok=True)
            finally:
                q.task_done()

    consumers = [consumer() for _ in range(num_consumers)]
    await TaskContext.gather(producer(), *consumers)
    sys.stdout.flush()
