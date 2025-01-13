# Copyright Modal Labs 2024
"""Pattern matching library ported from https://github.com/moby/patternmatcher.

This is the same pattern-matching logic used by Docker, except it is written in
Python rather than Go. Also, the original Go library has a couple deprecated
functions that we don't implement in this port.

The main way to use this library is by constructing a `FilePatternMatcher` object,
then asking it whether file paths match any of its patterns.
"""

import os
from abc import abstractmethod
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

from ._utils.pattern_utils import Pattern


class _AbstractPatternMatcher:
    _custom_repr: Optional[str] = None

    def __invert__(self) -> "_AbstractPatternMatcher":
        """Invert the filter. Returns a function that returns True if the path does not match any of the patterns.

        Usage:
        ```python
        from pathlib import Path
        from modal import FilePatternMatcher

        inverted_matcher = ~FilePatternMatcher("**/*.py")

        assert not inverted_matcher(Path("foo.py"))
        ```
        """
        return _CustomPatternMatcher(lambda path: not self(path))

    def _with_repr(self, custom_repr) -> "_AbstractPatternMatcher":
        # use to give an instance of a matcher a custom name - useful for visualizing default values in signatures
        self._custom_repr = custom_repr
        return self

    def __repr__(self) -> str:
        if self._custom_repr:
            return self._custom_repr

        return super().__repr__()

    @abstractmethod
    def __call__(self, path: Path) -> bool:
        ...


class _CustomPatternMatcher(_AbstractPatternMatcher):
    def __init__(self, predicate: Callable[[Path], bool]):
        self._predicate = predicate

    def __call__(self, path: Path) -> bool:
        return self._predicate(path)


class FilePatternMatcher(_AbstractPatternMatcher):
    """
    Allows matching file Path objects against a list of patterns.

    **Usage:**
    ```python
    from pathlib import Path
    from modal import FilePatternMatcher

    matcher = FilePatternMatcher("*.py")

    assert matcher(Path("foo.py"))

    # You can also negate the matcher.
    negated_matcher = ~matcher

    assert not negated_matcher(Path("foo.py"))
    ```
    """

    patterns: list[Pattern]
    _delayed_init: Callable[[], None] = None

    def _set_patterns(self, patterns: Sequence[str]) -> None:
        self.patterns = []
        for pattern in list(patterns):
            pattern = pattern.strip()
            if not pattern:
                continue
            pattern = os.path.normpath(pattern)
            new_pattern = Pattern()
            if pattern[0] == "!":
                if len(pattern) == 1:
                    raise ValueError('Illegal exclusion pattern: "!"')
                new_pattern.exclusion = True
                pattern = pattern[1:]
            # In Python, we can proceed without explicit syntax checking
            new_pattern.cleaned_pattern = pattern
            new_pattern.dirs = pattern.split(os.path.sep)
            self.patterns.append(new_pattern)

    def __init__(self, *pattern: str) -> None:
        """Initialize a new FilePatternMatcher instance.

        Args:
            pattern (str): One or more pattern strings.

        Raises:
            ValueError: If an illegal exclusion pattern is provided.
        """
        self._set_patterns(pattern)

    @classmethod
    def from_file(cls, file_path: Path) -> "FilePatternMatcher":
        """Initialize a new FilePatternMatcher instance from a file.

        The patterns in the file will be read lazily when the matcher is first used.

        Args:
            file_path (Path): The path to the file containing patterns.

        **Usage:**
        ```python
        from pathlib import Path
        from modal import FilePatternMatcher

        matcher = FilePatternMatcher.from_file(Path("/path/to/ignorefile"))
        ```

        """
        uninitialized = cls.__new__(cls)

        def _delayed_init():
            uninitialized._set_patterns(file_path.read_text("utf8").splitlines())
            uninitialized._delayed_init = None

        uninitialized._delayed_init = _delayed_init
        return uninitialized

    def _matches(self, file_path: str) -> bool:
        """Check if the file path or any of its parent directories match the patterns.

        This is equivalent to `MatchesOrParentMatches()` in the original Go
        library. The reason is that `Matches()` in the original library is
        deprecated due to buggy behavior.
        """

        matched = False
        file_path = os.path.normpath(file_path)
        if file_path == ".":
            # Don't let them exclude everything; kind of silly.
            return False
        parent_path = os.path.dirname(file_path)
        if parent_path == "":
            parent_path = "."
        parent_path_dirs = parent_path.split(os.path.sep)

        for pattern in self.patterns:
            # Skip evaluation based on current match status and pattern exclusion
            if pattern.exclusion != matched:
                continue

            match = pattern.match(file_path)

            if not match and parent_path != ".":
                # Check if the pattern matches any of the parent directories
                for i in range(len(parent_path_dirs)):
                    dir_path = os.path.sep.join(parent_path_dirs[: i + 1])
                    if pattern.match(dir_path):
                        match = True
                        break

            if match:
                matched = not pattern.exclusion

        return matched

    def __call__(self, file_path: Path) -> bool:
        if self._delayed_init:
            self._delayed_init()
        return self._matches(str(file_path))


# _with_repr allows us to use this matcher as a default value in a function signature
#  and get a nice repr in the docs and auto-generated type stubs:
NON_PYTHON_FILES = (~FilePatternMatcher("**/*.py"))._with_repr(f"{__name__}.NON_PYTHON_FILES")
_NOTHING = (~FilePatternMatcher())._with_repr(f"{__name__}._NOTHING")  # match everything = ignore nothing


def _ignore_fn(ignore: Union[Sequence[str], Callable[[Path], bool]]) -> Callable[[Path], bool]:
    # if a callable is passed, return it
    # otherwise, treat input as a sequence of patterns and return a callable pattern matcher for those
    if callable(ignore):
        return ignore

    return FilePatternMatcher(*ignore)
