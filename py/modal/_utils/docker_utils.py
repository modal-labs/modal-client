# Copyright Modal Labs 2024
import re
import shlex
from collections.abc import Sequence
from pathlib import Path

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
                if any(p.startswith("--from=") for p in parts):
                    current_command = ""
                    continue

                # Strip known COPY flags (--chmod, --chown, --link) before processing sources.
                known_flag_prefixes = ("--chmod=", "--chown=", "--link=")
                parts = [
                    p
                    for p in parts
                    # link has a special handling - it can be "--link" or like "--link=true"
                    if p != "--link" and not any(p.startswith(prefix) for prefix in known_flag_prefixes)
                ]

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


def find_dockerignore_file(context_directory: Path, dockerfile_path: Path | None = None) -> Path | None:
    """
    Find dockerignore file relative to current context directory
    and if dockerfile path is provided, check if specific <dockerfile_name>.dockerignore
    file exists in the same directory as <dockerfile_name>
    Finds the most specific dockerignore file that exists.
    """

    def valid_dockerignore_file(fp):
        # fp has to exist
        if not fp.exists():
            return False
        # fp has to be subpath to current working directory
        if not fp.is_relative_to(context_directory):
            return False

        return True

    generic_name = ".dockerignore"
    possible_locations = []
    if dockerfile_path:
        specific_name = f"{dockerfile_path.name}.dockerignore"
        # 1. check if specific <dockerfile_name>.dockerignore file exists in the same directory as <dockerfile_name>
        possible_locations.append(dockerfile_path.parent / specific_name)
        # 2. check if generic .dockerignore file exists in the same directory as <dockerfile_name>
        possible_locations.append(dockerfile_path.parent / generic_name)

    # 3. check if generic .dockerignore file exists in current working directory
    possible_locations.append(context_directory / generic_name)

    return next((e for e in possible_locations if valid_dockerignore_file(e)), None)
