from modal_utils.app_utils import is_valid_deployment_name


def test_deployment_name():
    assert is_valid_deployment_name("banana")
    assert is_valid_deployment_name("BaNaNa")
    assert not is_valid_deployment_name(" ")
    assert not is_valid_deployment_name("ban/ana")
