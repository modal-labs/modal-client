# Copyright Modal Labs 2024
"""Pattern matching library ported from https://github.com/moby/patternmatcher.

This is the same pattern-matching logic used by Docker, except it is written in
Python rather than Go. Also, the original Go library has a couple deprecated
functions that we don't implement in this port.

The main way to use this library is by constructing a `PatternMatcher` object,
then asking it whether file paths match any of its patterns.
"""

import enum
import os
import re
from typing import List, Optional, TextIO

escape_chars = frozenset(".+()|{}$")


class MatchType(enum.IntEnum):
    UNKNOWN = 0
    EXACT = 1
    PREFIX = 2
    SUFFIX = 3
    REGEXP = 4


class Pattern:
    """Defines a single regex pattern used to filter file paths."""

    def __init__(self) -> None:
        """Initialize a new Pattern instance."""
        self.match_type = MatchType.UNKNOWN
        self.cleaned_pattern = ""
        self.dirs: List[str] = []
        self.regexp: Optional[re.Pattern] = None
        self.exclusion = False

    def __str__(self) -> str:
        """Return the cleaned pattern as the string representation."""
        return self.cleaned_pattern

    def compile(self, separator: str) -> None:
        """Compile the pattern into a regular expression.

        Args:
            separator (str): The path separator (e.g., '/' or '\\').

        Raises:
            ValueError: If the pattern is invalid.
        """
        reg_str = "^"
        pattern = self.cleaned_pattern

        esc_separator = separator
        if separator == "\\":
            esc_separator = "\\\\"

        self.match_type = MatchType.EXACT
        i = 0
        pattern_length = len(pattern)
        while i < pattern_length:
            ch = pattern[i]
            if ch == "*":
                if (i + 1) < pattern_length and pattern[i + 1] == "*":
                    # Handle '**'
                    i += 1  # Skip the second '*'
                    # Treat '**/' as '**' so eat the '/'
                    if (i + 1) < pattern_length and pattern[i + 1] == separator:
                        i += 1  # Skip the '/'
                    if i + 1 == pattern_length:
                        # Pattern ends with '**'
                        if self.match_type == MatchType.EXACT:
                            self.match_type = MatchType.PREFIX
                        else:
                            reg_str += ".*"
                            self.match_type = MatchType.REGEXP
                    else:
                        # '**' in the middle
                        reg_str += f"(.*{esc_separator})?"
                        self.match_type = MatchType.REGEXP

                    if i == 1:
                        self.match_type = MatchType.SUFFIX
                else:
                    # Single '*'
                    reg_str += f"[^{esc_separator}]*"
                    self.match_type = MatchType.REGEXP
            elif ch == "?":
                # Single '?'
                reg_str += f"[^{esc_separator}]"
                self.match_type = MatchType.REGEXP
            elif ch in escape_chars:
                reg_str += "\\" + ch
            elif ch == "\\":
                # Escape next character
                if separator == "\\":
                    reg_str += esc_separator
                    i += 1
                    continue
                if (i + 1) < pattern_length:
                    reg_str += "\\" + pattern[i + 1]
                    i += 1  # Skip the escaped character
                    self.match_type = MatchType.REGEXP
                else:
                    reg_str += "\\"
            elif ch == "[" or ch == "]":
                reg_str += ch
                self.match_type = MatchType.REGEXP
            else:
                reg_str += ch
            i += 1

        if self.match_type != MatchType.REGEXP:
            return

        reg_str += "$"

        try:
            self.regexp = re.compile(reg_str)
            self.match_type = MatchType.REGEXP
        except re.error as e:
            raise ValueError(f"Bad pattern: {pattern}") from e

    def match(self, path: str) -> bool:
        """Check if the path matches the pattern."""
        if self.match_type == MatchType.UNKNOWN:
            self.compile(os.path.sep)

        if self.match_type == MatchType.EXACT:
            return path == self.cleaned_pattern
        elif self.match_type == MatchType.PREFIX:
            # Strip trailing '**'
            return path.startswith(self.cleaned_pattern[:-2])
        elif self.match_type == MatchType.SUFFIX:
            # Strip leading '**'
            suffix = self.cleaned_pattern[2:]
            if path.endswith(suffix):
                return True
            # '**/foo' matches 'foo'
            if suffix[0] == os.path.sep and path == suffix[1:]:
                return True
            else:
                return False
        elif self.match_type == MatchType.REGEXP:
            return self.regexp.match(path) is not None
        else:
            return False


class PatternMatcher:
    """Allows checking paths against a list of patterns."""

    def __init__(self, patterns: List[str]) -> None:
        """Initialize a new PatternMatcher instance.

        Args:
            patterns (list): A list of pattern strings.

        Raises:
            ValueError: If an illegal exclusion pattern is provided.
        """
        self.patterns: List[Pattern] = []
        self.exclusions = False
        for pattern in patterns:
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
                self.exclusions = True
            # In Python, we can proceed without explicit syntax checking
            new_pattern.cleaned_pattern = pattern
            new_pattern.dirs = pattern.split(os.path.sep)
            self.patterns.append(new_pattern)

    def matches(self, file_path: str) -> bool:
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


def read_ignorefile(reader: TextIO) -> List[str]:
    """Read an ignore file from a reader and return the list of file patterns to
    ignore, applying the following rules:

    - An UTF8 BOM header (if present) is stripped. (Python does this already)
    - Lines starting with "#" are considered comments and are skipped.

    For remaining lines:

    - Leading and trailing whitespace is removed from each ignore pattern.
    - It uses `os.path.normpath` to get the shortest/cleanest path for ignore
      patterns.
    - Leading forward-slashes ("/") are removed from ignore patterns, so
      "/some/path" and "some/path" are considered equivalent.

    Args:
        reader (file-like object): The input stream to read from.

    Returns:
        list: A list of patterns to ignore.
    """
    if reader is None:
        return []

    excludes: List[str] = []

    for line in reader:
        pattern = line.rstrip("\n\r")

        # Lines starting with "#" are ignored
        if pattern.startswith("#"):
            continue

        pattern = pattern.strip()
        if pattern == "":
            continue

        # Normalize absolute paths to paths relative to the context
        # (taking care of '!' prefix)
        invert = pattern[0] == "!"
        if invert:
            pattern = pattern[1:].strip()

        if len(pattern) > 0:
            pattern = os.path.normpath(pattern)
            pattern = pattern.replace(os.sep, "/")
            if len(pattern) > 1 and pattern[0] == "/":
                pattern = pattern[1:]

        if invert:
            pattern = "!" + pattern

        excludes.append(pattern)

    return excludes
