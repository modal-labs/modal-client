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

from modal.functions import _Function
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


async def _watch_paths(paths: Set[Union[str, Path]], watch_filter: StubFilesFilter):
    try:
        async for changes in awatch(*paths, step=500, watch_filter=watch_filter):
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


def _watch_args_from_mounts(mounts: List[_Mount]) -> Tuple[Set[Union[str, Path]], StubFilesFilter]:
    paths = set()
    dir_filters: Dict[Path, Set[str]] = defaultdict(set)
    for mount in mounts:
        if mount._local_dir is not None:
            paths.add(mount._local_dir)
            dir_filters[Path(mount._local_dir)] = None
        elif mount._local_file is not None:
            parent = Path(mount._local_file).parent
            paths.add(parent)
            if dir_filters[parent] is not None:
                dir_filters[parent].add(str(mount._local_file))

    watch_filter = StubFilesFilter(dir_filters=dict(dir_filters))
    return paths, watch_filter


def _is_local_mount(mount) -> bool:
    if not isinstance(mount, _Mount):
        return False
    # TODO(erikbern): this is pretty ugly, but we want to ignore
    # any _Mount that's just a remote reference. Let's rethink this.
    return getattr(mount, "_local_dir", None) or getattr(mount, "_local_file", None)


async def watch(stub: _Stub, output_mgr: OutputManager, timeout: Optional[float]):
    # Iterate through all mounts for all functions and
    # collect unique directories and files to watch
    all_mounts = []
    for object in stub._blueprint.values():
        if isinstance(object, _Function):
            all_mounts.extend([mount for mount in object._mounts if _is_local_mount(mount)])
    paths, watch_filter = _watch_args_from_mounts(mounts=all_mounts)

    _print_watched_paths(paths, output_mgr, timeout)

    yield AppChange.START

    timeout_agen = [] if timeout is None else [_sleep(timeout)]
    async with stream.merge(_watch_paths(paths, watch_filter), *timeout_agen).stream() as streamer:
        async for event in streamer:
            yield event
