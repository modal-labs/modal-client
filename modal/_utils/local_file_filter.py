# Copyright Modal Labs 2024
from pathlib import Path

from .pattern_matcher import PatternMatcher


class LocalFileFilter:
    """Allows checking paths against one or more patterns."""

    def __init__(self, *pattern: str, invert: bool = False) -> None:
        """Initialize a new LocalFileFilter instance.

        Args:
            pattern (str): One or more pattern strings.

        """
        self.patterns: list[str] = list(pattern)
        self.invert: bool = invert

    def __call__(self, file_path: Path) -> bool:
        if self.invert:
            return not PatternMatcher(self.patterns).matches(str(file_path))
        return PatternMatcher(self.patterns).matches(str(file_path))

    def __invert__(self) -> "LocalFileFilter":
        return LocalFileFilter(*self.patterns, invert=True)
