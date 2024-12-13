# Copyright Modal Labs 2024
from pathlib import Path
from typing import Callable

from .pattern_matcher import PatternMatcher


class LocalFileFilter:
    """Allows checking paths against one or more patterns."""

    def __init__(self, *pattern: str) -> None:
        """Initialize a new LocalFileFilter instance. Calling it will return True if the path matches any of
        the patterns.

        Args:
            pattern (str): One or more pattern strings.
        """
        self.patterns: list[str] = list(pattern)

    def __call__(self, file_path: Path) -> bool:
        """Check if the path matches any of the patterns.

        Args:
            file_path (Path): The path to check.

        Returns:
            True if the path matches any of the patterns.

        Usage:
        ```python

        filter = LocalFileFilter("*.py")

        assert filter(Path("foo.py"))
        ```
        """
        return PatternMatcher(self.patterns).matches(str(file_path))

    def __invert__(self) -> Callable[[Path], bool]:
        """Invert the filter. Returns a function that returns True if the path does not match any of the patterns.

        Usage:
        ```python

        filter = LocalFileFilter("*.py")

        assert not ~filter(Path("foo.py"))
        ```
        """
        return lambda path: not self(path)
