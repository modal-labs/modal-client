import asyncio
from pathlib import Path
from typing import Optional, Set, Union

from aiostream import stream
from rich.tree import Tree
from watchfiles import awatch

from modal.functions import _Function
from modal.mount import _Mount
from modal.stub import _Stub

from ._output import OutputManager

START = object()
TIMEOUT = object()


async def _sleep(timeout: float):
    await asyncio.sleep(timeout)
    yield TIMEOUT


async def _watch_paths(paths: Set[Union[str, Path]]):
    try:
        async for changes in awatch(*paths, step=500):
            yield changes
    except RuntimeError:
        # Thrown by watchfiles from Rust, when the generator is closed externally.
        pass


def _print_watched_paths(paths: Set[Union[str, Path]], output_mgr: OutputManager, timeout: Optional[float]):
    if timeout:
        msg = f"⚡️ Serving for {timeout} seconds... hit Ctrl-C to stop!"
    else:
        msg = "️️⚡️ Serving... hit Ctrl-C to stop!"

    output_tree = Tree(msg, guide_style="gray50")

    for path in paths:
        output_tree.add(f"Watching {path}.")

    output_mgr.print_if_visible(output_tree)


async def watch(stub: _Stub, output_mgr: OutputManager, timeout: Optional[float]):
    paths = set()

    # Iterate through all mounts for all functions and
    # collect unique directories and files to watch
    for object in stub._blueprint.values():
        if isinstance(object, _Function):
            for mount in object._mounts:
                if isinstance(mount, _Mount):
                    if mount._local_dir is not None:
                        paths.add(mount._local_dir)
                    elif mount._local_file is not None:
                        paths.add(mount._local_file)

    _print_watched_paths(paths, output_mgr, timeout)

    yield START

    timeout_agen = [] if timeout is None else [_sleep(timeout)]

    async with stream.merge(_watch_paths(paths), *timeout_agen).stream() as streamer:
        async for event in streamer:
            yield event
