# Copyright Modal Labs 2024
import pytest
from collections.abc import Mapping

from modal._utils.mount_utils import validate_mount_points, validate_network_file_systems, validate_volumes
from modal.exception import InvalidError
from modal.network_file_system import NetworkFileSystem
from modal.volume import Volume


def test_validate_mount_points():
    # valid mount points
    dict_input: Mapping = {"/foo/bar": NetworkFileSystem.from_name("NetworkFileSystem", create_if_missing=False)}
    validate_mount_points("NetworkFileSystem", dict_input)

    # invalid list input, should be dicts
    list_input = [NetworkFileSystem.from_name("NetworkFileSystem", create_if_missing=False)]

    with pytest.raises(InvalidError, match="volume_likes"):
        validate_mount_points("NetworkFileSystem", list_input)  # type: ignore


@pytest.mark.parametrize("path", ["/", "/root", "/tmp", "foo/bar"])
def test_validate_mount_points_invalid_paths(path):
    validated_mount_points: Mapping = {path: NetworkFileSystem.from_name("NetworkFileSystem", create_if_missing=False)}
    with pytest.raises(InvalidError, match="NetworkFileSystem"):
        validate_mount_points("NetworkFileSystem", validated_mount_points)


def test_validate_network_file_systems(client, servicer):
    # valid network_file_systems input
    network_file_systems: Mapping = {"/my/path": NetworkFileSystem.from_name("foo", create_if_missing=False)}
    validate_network_file_systems(network_file_systems)

    # invalid non network_file_systems input
    not_network_file_systems = {"/my/path": Volume.from_name("foo", create_if_missing=False)}
    with pytest.raises(InvalidError, match="Volume"):
        validate_network_file_systems(not_network_file_systems)  # type: ignore


def test_validate_volumes(client, servicer):
    # valid volume input
    volumes: Mapping = {"/my/path": Volume.from_name("foo", create_if_missing=False)}
    validate_volumes(volumes)

    # invalid non volume input
    not_volumes = {"/my/path": NetworkFileSystem.from_name("foo", create_if_missing=False)}
    with pytest.raises(InvalidError, match="NetworkFileSystem"):
        validate_volumes(not_volumes)  # type: ignore

    # invalid attempt mount volume twice
    vol = Volume.from_name("foo", create_if_missing=False)
    bad_path_volumes: Mapping = {"/my/path": vol, "/my/other/path": vol}
    with pytest.raises(InvalidError, match="Volume"):
        validate_volumes(bad_path_volumes)
