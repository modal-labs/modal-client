# Copyright Modal Labs 2024
"""Pattern matching library ported from https://github.com/moby/patternmatcher.

This is the same pattern-matching logic used by Docker, except it is written in
Python rather than Go. Also, the original Go library has a couple deprecated
functions that we don't implement in this port.

The main way to use this library is by constructing a `FilePatternMatcher` object,
then asking it whether file paths match any of its patterns.
"""

import os
from pathlib import Path
from typing import Callable

from ._utils.pattern_utils import Pattern


class FilePatternMatcher:
    """Allows matching file paths against a list of patterns."""

    def __init__(self, *pattern: str) -> None:
        """Initialize a new FilePatternMatcher instance.

        Args:
            pattern (str): One or more pattern strings.

        Raises:
            ValueError: If an illegal exclusion pattern is provided.
        """
        self.patterns: list[Pattern] = []
        self.exclusions = False
        for p in list(pattern):
            p = p.strip()
            if not p:
                continue
            p = os.path.normpath(p)
            new_pattern = Pattern()
            if p[0] == "!":
                if len(p) == 1:
                    raise ValueError('Illegal exclusion pattern: "!"')
                new_pattern.exclusion = True
                p = p[1:]
                self.exclusions = True
            # In Python, we can proceed without explicit syntax checking
            new_pattern.cleaned_pattern = p
            new_pattern.dirs = p.split(os.path.sep)
            self.patterns.append(new_pattern)

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
        """Check if the path matches any of the patterns.

        Args:
            file_path (Path): The path to check.

        Returns:
            True if the path matches any of the patterns.

        Usage:
        ```python
        from pathlib import Path
        from modal import FilePatternMatcher

        matcher = FilePatternMatcher("*.py")

        assert matcher(Path("foo.py"))
        ```
        """
        return self._matches(str(file_path))

    def __invert__(self) -> Callable[[Path], bool]:
        """Invert the filter. Returns a function that returns True if the path does not match any of the patterns.

        Usage:
        ```python
        from pathlib import Path
        from modal import FilePatternMatcher

        inverted_matcher = ~FilePatternMatcher("**/*.py")

        assert not inverted_matcher(Path("foo.py"))
        ```
        """
        return lambda path: not self(path)
