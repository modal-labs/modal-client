# Copyright Modal Labs 2024
import re
import shlex
from pathlib import Path
from typing import Optional

from .exception import InvalidError


def extract_copy_command_patterns(dockerfile_lines: list[str]) -> list[str]:
    """
    Extract all COPY command sources from a Dockerfile.
    Combines multiline COPY commands into a single line.
    """
    mount_sources = set()
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

                if len(parts) >= 2:
                    # Last part is destination, everything else is a mount source
                    sources = parts[:-1]

                    for source in sources:
                        special_pattern = re.compile(r"^\s*--|\$\s*")
                        if special_pattern.match(source):
                            raise InvalidError(
                                f"COPY command: {source} using special flags/arguments/variables are not supported"
                            )

                        mount_sources.add(source)

            current_command = ""

    return list(mount_sources)


def find_dockerignore_file(dockerfile_path: Path, context_directory: Path) -> Optional[Path]:
    # 1. file next to the Dockerfile but do NOT look for parent directories

    def valid_dockerignore_file(fp):
        # fp has to exist
        if not fp.exists():
            return False
        # fp has to be subpath to current working directory
        if not fp.is_relative_to(context_directory):
            return False

        return True

    generic_name = ".dockerignore"
    specific_name = f"{dockerfile_path.name}.dockerignore"

    possible_locations = [
        # 1. check if specific <dockerfile_name>.dockerignore file exists in the same directory as <dockerfile_name>
        dockerfile_path.parent / specific_name,
        # 2. check if generic .dockerignore file exists in the same directory as <dockerfile_name>
        dockerfile_path.parent / generic_name,
        # 3. check if generic .dockerignore file exists in current working directory
        context_directory / generic_name,
        # 4. ?????? check if specific <dockerfile_name>.dockerignore file exists in current working directory
        context_directory / specific_name,
    ]

    return next((e for e in possible_locations if valid_dockerignore_file(e)), None)
