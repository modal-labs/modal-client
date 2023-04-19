# Copyright Modal Labs 2022
from collections import defaultdict
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Set, Tuple

from rich.tree import Tree
from watchfiles import Change, DefaultFilter, awatch

from modal.mount import _Mount

from ._output import OutputManager


_TIMEOUT_SENTINEL = object()


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

        allowlists = set()

        for root, allowlist in self.dir_filters.items():
            # For every filter path that's a parent of the current path...
            if root in p.parents:
                # If allowlist is None, we're watching the directory and we have a match.
                if allowlist is None:
                    return super().__call__(change, path)

                # If not, it's specific files, and we could have a match.
                else:
                    allowlists |= allowlist

        if allowlists and path not in allowlists:
            return False

        return super().__call__(change, path)


async def _watch_paths(paths: Set[Path], watch_filter: StubFilesFilter):
    try:
        async for changes in awatch(*paths, step=500, watch_filter=watch_filter):
            yield changes
    except RuntimeError:
        # Thrown by watchfiles from Rust, when the generator is closed externally.
        pass


def _print_watched_paths(paths: Set[Path], output_mgr: OutputManager):
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


async def watch(mounts: List[_Mount], output_mgr: OutputManager) -> AsyncGenerator[None, None]:
    paths, watch_filter = _watch_args_from_mounts(mounts)

    _print_watched_paths(paths, output_mgr)

    async for event in _watch_paths(paths, watch_filter):
        yield
