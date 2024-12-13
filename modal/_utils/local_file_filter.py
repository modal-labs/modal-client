# Copyright Modal Labs 2024
from pathlib import Path

from .pattern_matcher import PatternMatcher


class LocalFileFilter:
    """Allows checking paths against one or more patterns."""

    def __init__(self, *pattern: str, invert: bool = False) -> None:
        """Initialize a new LocalFileFilter instance. Calling it will return True if the path matches any of
        the patterns. Uses the underlying PatternMatcher to match the patterns which is dockerignore compatible.

        Args:
            pattern (str): One or more pattern strings.
            invert (bool): If True, the filter will return True for paths that do not match any of the patterns.
        """
        self.patterns: list[str] = list(pattern)
        self.invert: bool = invert

    def __call__(self, file_path: Path) -> bool:
        """Check if the path matches any of the patterns.

        Args:
            file_path (Path): The path to check.

        Returns:
            True if the path matches any of the patterns.
            If self.invert is set to True, returns True if the path does not match any of the patterns instead.

        Usage:
        ```python

        filter = LocalFileFilter("*.py")

        assert filter(Path("foo.py"))
        assert not ~filter(Path("foo.py"))
        ```
        """
        if self.invert:
            return not PatternMatcher(self.patterns).matches(str(file_path))
        return PatternMatcher(self.patterns).matches(str(file_path))

    def __invert__(self) -> "LocalFileFilter":
        """Invert the filter. If self.invert is set to True, returns True if the path does not match any of the patterns
        instead.

        Usage:
        ```python

        filter = LocalFileFilter("*.py")

        assert filter(Path("foo.py"))

        assert not ~filter(Path("foo.py"))
        ```
        """
        return LocalFileFilter(*self.patterns, invert=True)
