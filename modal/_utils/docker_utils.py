# Copyright Modal Labs 2024
import re
import shlex
from typing import Sequence

from ..exception import InvalidError


def extract_copy_command_patterns(dockerfile_lines: Sequence[str]) -> list[str]:
    """
    Extract all COPY command sources from a Dockerfile.
    Combines multiline COPY commands into a single line.
    """
    copy_source_patterns: set[str] = set()
    current_command = ""
    copy_pattern = re.compile(r"^\s*COPY\s+(.+)$", re.IGNORECASE)

    # First pass: handle line continuations and collect full commands
    for line in dockerfile_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            # ignore comments and empty lines
            continue

        if current_command:
            # Continue previous line
            current_command += " " + line.rstrip("\\").strip()
        else:
            # Start new command
            current_command = line.rstrip("\\").strip()

        if not line.endswith("\\"):
            # Command is complete

            match = copy_pattern.match(current_command)
            if match:
                args = match.group(1)
                parts = shlex.split(args)

                # COPY --from=... commands reference external sources and do not need a context mount.
                # https://docs.docker.com/reference/dockerfile/#copy---from
                if parts[0].startswith("--from="):
                    current_command = ""
                    continue

                if len(parts) >= 2:
                    # Last part is destination, everything else is a mount source
                    sources = parts[:-1]

                    for source in sources:
                        special_pattern = re.compile(r"^\s*--|\$\s*")
                        if special_pattern.match(source):
                            raise InvalidError(
                                f"COPY command: {source} using special flags/arguments/variables are not supported"
                            )

                        if source == ".":
                            copy_source_patterns.add("./**")
                        else:
                            copy_source_patterns.add(source)

            current_command = ""

    return list(copy_source_patterns)
