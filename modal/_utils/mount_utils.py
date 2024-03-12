# Copyright Modal Labs 2022
import posixpath
import typing
from pathlib import PurePath, PurePosixPath
from typing import TYPE_CHECKING, Dict, List, Mapping, Tuple, Union

from ..exception import InvalidError
from ..volume import _Volume

if TYPE_CHECKING:
    from ..cloud_bucket_mount import _CloudBucketMount
    from ..network_file_system import _NetworkFileSystem


T = typing.TypeVar("T", bound=Union["_Volume", "_NetworkFileSystem", "_CloudBucketMount"])


def validate_mount_points(
    display_name: str,
    volume_likes: Mapping[Union[str, PurePosixPath], T],
) -> List[Tuple[str, T]]:
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


def validate_volumes(
    volumes: Mapping[Union[str, PurePosixPath], Union["_Volume", "_CloudBucketMount"]],
) -> List[Tuple[str, Union["_Volume", "_NetworkFileSystem", "_CloudBucketMount"]]]:
    if not isinstance(volumes, dict):
        raise InvalidError("volumes must be a dict[str, Volume] where the keys are paths")

    validated_volumes = validate_mount_points("Volume", volumes)
    # We don't support mounting a volume in more than one location
    volume_to_paths: Dict["_Volume", List[str]] = {}
    for path, volume in validated_volumes:
        if isinstance(volume, _Volume):
            volume_to_paths.setdefault(volume, []).append(path)
    for paths in volume_to_paths.values():
        if len(paths) > 1:
            conflicting = ", ".join(paths)
            raise InvalidError(
                f"The same Volume cannot be mounted in multiple locations for the same function: {conflicting}"
            )

    return validated_volumes
