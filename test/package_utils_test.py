# Copyright Modal Labs 2022
import pytest

from modal_utils.package_utils import get_module_mount_info
from modal.exception import ModuleNotMountable


def test_get_module_mount_info():
    res = get_module_mount_info("modal")
    assert len(res) == 1
    assert res[0][0] == True

    res = get_module_mount_info("asyncio")
    assert len(res) == 1
    assert res[0][0] == True

    res = get_module_mount_info("six")
    assert len(res) == 1
    assert res[0][0] == False

    with pytest.raises(ModuleNotMountable, match="aiohttp can't be mounted because it contains binary file"):
        get_module_mount_info("aiohttp")
