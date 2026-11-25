# Copyright Modal Labs 2024
import pytest
from pathlib import PurePosixPath

from modal._utils.mount_utils import (
    validate_mount_points,
    validate_only_modal_volumes,
    validate_volumes,
)
from modal.cloud_bucket_mount import _CloudBucketMount
from modal.exception import InvalidError
from modal.volume import _Volume


def test_validate_mount_points():
    # valid mount points
    dict_input: dict[str | PurePosixPath, _Volume] = {"/foo/bar": _Volume.from_name("volume", create_if_missing=False)}
    validate_mount_points("Volume", dict_input)

    # invalid list input, should be dicts
    list_input = [_Volume.from_name("volume", create_if_missing=False)]

    with pytest.raises(InvalidError, match="volume_likes"):
        validate_mount_points("Volume", list_input)  # type: ignore


@pytest.mark.parametrize("path", ["/", "/root", "/tmp", "foo/bar"])
def test_validate_mount_points_invalid_paths(path):
    validated_mount_points = {path: _Volume.from_name("volume", create_if_missing=False)}
    with pytest.raises(InvalidError, match="Volume"):
        validate_mount_points("Volume", validated_mount_points)


def test_validate_volumes(client, servicer):
    # valid volume input
    volumes = {"/my/path": _Volume.from_name("foo", create_if_missing=False)}
    validate_volumes(volumes)  # type: ignore

    # invalid non volume input
    not_volumes = {"/my/path": object()}
    with pytest.raises(InvalidError, match="object"):
        validate_volumes(not_volumes)  # type: ignore

    # invalid attempt mount volume twice
    vol = _Volume.from_name("foo", create_if_missing=False)
    bad_path_volumes = {"/my/path": vol, "/my/other/path": vol}
    with pytest.raises(InvalidError, match="Volume"):
        validate_volumes(bad_path_volumes)  # type: ignore


def test_validate_only_modal_volumes():
    valid_volumes = {"/my/path": _Volume.from_name("foo", create_if_missing=False)}
    output = validate_only_modal_volumes(valid_volumes, "test_function")  # type: ignore
    assert output == [("/my/path", valid_volumes["/my/path"])]

    output = validate_only_modal_volumes(None, "test_function")  # type: ignore
    assert not output

    invalid_volumes = {
        "/my/path": _Volume.from_name("foo", create_if_missing=False),
        "/dev/path": _CloudBucketMount(bucket_name="hello_world"),
    }
    with pytest.raises(InvalidError, match="Image.run_commands only supports volumes that are modal.Volume"):
        validate_only_modal_volumes(invalid_volumes, "Image.run_commands")  # type: ignore
