# Copyright Modal Labs 2022
import asyncio
from collections import defaultdict
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from aiostream import stream
from rich.tree import Tree
from watchfiles import Change, DefaultFilter, awatch
from watchfiles.main import FileChange

from modal.mount import _Mount
from modal.stub import _Stub

from ._output import OutputManager


class AppChange(IntEnum):
    """
    Enum representing the type of a change in the Modal stub serving state.
    """

    START = 1
    TIMEOUT = 2


FileChangeset = Set[FileChange]
ChangeEvent = Union[AppChange, FileChangeset, None]


class StubFilesFilter(DefaultFilter):
    def __init__(
        self,
        *,
        # A directory filter is used to only watch certain files within a directory.
        # Watching specific files is discouraged on Linux, so to watch a file we watch its
        # containing directory and then filter that directory's changes for relevant files.
        # https://github.com/notify-rs/notify/issues/394
        dir_filters: Dict[Path, Optional[Set[str]]],
    ) -> None:
        self.dir_filters = dir_filters
        super().__init__()

    def __call__(self, change: Change, path: str) -> bool:
        p = Path(path)
        if p.name == ".DS_Store":
            return False
        # Vim creates this temporary file to see whether it can write
        # into a target directory.
        elif p.name == "4913":
            return False
        for root, allowlist in self.dir_filters.items():
            if allowlist is not None and root in p.parents and path not in allowlist:
                return False
        return super().__call__(change, path)


async def _sleep(timeout: float):
    await asyncio.sleep(timeout)
    yield AppChange.TIMEOUT


async def _watch_paths(paths: Set[Path], watch_filter: StubFilesFilter):
    try:
        async for changes in awatch(*paths, step=500, watch_filter=watch_filter):
            yield changes
    except RuntimeError:
        # Thrown by watchfiles from Rust, when the generator is closed externally.
        pass


def _print_watched_paths(paths: Set[Path], output_mgr: OutputManager, timeout: Optional[float]):
    if timeout:
        msg = f"⚡️ Serving for {timeout} seconds... hit Ctrl-C to stop!"
    else:
        msg = "️️⚡️ Serving... hit Ctrl-C to stop!"

    output_tree = Tree(msg, guide_style="gray50")

    for path in paths:
        output_tree.add(f"Watching {path}.")

    output_mgr.print_if_visible(output_tree)


def _watch_args_from_mounts(mounts: List[_Mount]) -> Tuple[Set[Path], StubFilesFilter]:
    paths = set()
    dir_filters: Dict[Path, Optional[Set[str]]] = defaultdict(set)
    for mount in mounts:
        # TODO(elias): Make this part of the mount class instead, since it uses so much internals
        for entry in mount._entries:
            path, filter_file = entry.watch_entry()
            paths.add(path)
            if filter_file is None:
                dir_filters[path] = None
            elif dir_filters[path] is not None:
                dir_filters[path].add(filter_file.as_posix())

    watch_filter = StubFilesFilter(dir_filters=dict(dir_filters))
    return paths, watch_filter


async def watch(stub: _Stub, output_mgr: OutputManager, timeout: Optional[float]):
    paths, watch_filter = _watch_args_from_mounts(mounts=stub._local_mounts)

    _print_watched_paths(paths, output_mgr, timeout)

    yield AppChange.START

    timeout_agen = [] if timeout is None else [_sleep(timeout)]
    async with stream.merge(_watch_paths(paths, watch_filter), *timeout_agen).stream() as streamer:
        async for event in streamer:
            yield event
