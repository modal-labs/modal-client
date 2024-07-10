import pytest

from modal._utils.mount_utils import validate_mount_points, validate_network_file_systems, validate_volumes
from modal.exception import InvalidError
from modal.network_file_system import _NetworkFileSystem
from modal.volume import _Volume


def test_validate_mount_points():
    # valid mount points
    dict_input = {"/foo/bar": _NetworkFileSystem.from_name("NetworkFileSystem", create_if_missing=False)}
    validate_mount_points("NetworkFileSystem", dict_input)

    # invalid list input, should be dicts
    list_input = [_NetworkFileSystem.from_name("NetworkFileSystem", create_if_missing=False)]

    with pytest.raises(InvalidError, match="NetworkFileSystem"):
        validate_mount_points("NetworkFileSystem", list_input)


@pytest.mark.parametrize("path", ["/", "/root", "/tmp", "foo/bar"])
def test_validate_mount_points_invalid_paths(path):
    validated_mount_points = {path: _NetworkFileSystem.from_name("NetworkFileSystem", create_if_missing=False)}
    with pytest.raises(InvalidError, match="NetworkFileSystem"):
        validate_mount_points("NetworkFileSystem", validated_mount_points)


def test_validate_network_file_systems(client, servicer):
    # valid network_file_systems input
    network_file_systems = {"/my/path": _NetworkFileSystem.from_name("foo", create_if_missing=False)}
    validate_network_file_systems(network_file_systems)

    # invalid non network_file_systems input
    not_network_file_systems = {"/my/path": _Volume.from_name("foo", create_if_missing=False)}
    with pytest.raises(InvalidError, match="_Volume"):
        validate_network_file_systems(not_network_file_systems)


def test_validate_volumes(client, servicer):
    # valid volume input
    volumes = {"/my/path": _Volume.from_name("foo", create_if_missing=False)}
    validate_volumes(volumes)

    # invalid non volume input
    not_volumes = {"/my/path": _NetworkFileSystem.from_name("foo", create_if_missing=False)}
    with pytest.raises(InvalidError, match="_NetworkFileSystem"):
        validate_volumes(not_volumes)

    # invalid attempt mount volume twice
    vol = _Volume.from_name("foo", create_if_missing=False)
    with pytest.raises(InvalidError, match="Volume"):
        validate_volumes({"/my/path": vol, "/my/other/path": vol})
