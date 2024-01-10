# Copyright Modal Labs 2022
import os
import posixpath
from pathlib import PurePath
from typing import TYPE_CHECKING, Dict, List, Tuple, Union

from .exception import InvalidError

if TYPE_CHECKING:
    from .network_file_system import _NetworkFileSystem
    from .volume import _Volume


def validate_mount_points(
    display_name: str, volume_likes: Dict[Union[str, os.PathLike], Union["_Volume", "_NetworkFileSystem"]]
) -> List[Tuple[str, Union["_Volume", "_NetworkFileSystem"]]]:
    """Mount point path validation for volumes and network file systems."""

    validated = []
    for path, vol in volume_likes.items():
        path = PurePath(path).as_posix()
        abs_path = posixpath.abspath(path)

        if path != abs_path:
            raise InvalidError(f"{display_name} {path} must be a canonical, absolute path.")
        elif abs_path == "/":
            raise InvalidError(f"{display_name} {path} cannot be mounted into root directory.")
        elif abs_path == "/root":
            raise InvalidError(f"{display_name} {path} cannot be mounted at '/root'.")
        elif abs_path == "/tmp":
            raise InvalidError(f"{display_name} {path} cannot be mounted at '/tmp'.")
        validated.append((path, vol))
    return validated
