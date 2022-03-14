from modal_utils.package_utils import get_module_mount_info


def test_get_module_mount_info():
    res = get_module_mount_info("modal")
    assert len(res) == 1
    assert res[0][0] == "modal"

    res = get_module_mount_info("asyncio")
    assert len(res) == 1
    assert res[0][0] == "asyncio"
