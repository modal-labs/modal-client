from modal._package_utils import get_package_deps_mount_info


def test_get_package_deps_mount_info():
    res = get_package_deps_mount_info("modal")
    assert len(res) == 1
    assert res[0][0] == "modal"

    res = get_package_deps_mount_info("asyncio")
    assert len(res) == 1
    assert res[0][0] == "asyncio"
