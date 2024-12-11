# Copyright Modal Labs 2024
from pathlib import Path

from .pattern_matcher import PatternMatcher


class LocalFileFilter:
    """Allows checking paths against one or more patterns."""

    def __init__(self, *pattern: str) -> None:
        """Initialize a new LocalFileFilter instance.

        Args:
            pattern (str): One or more pattern strings.

        """
        self.patterns: list[str] = list(pattern)

    def __call__(self, file_path: Path) -> bool:
        return PatternMatcher(self.patterns).matches(str(file_path))
