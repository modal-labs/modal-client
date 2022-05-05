from modal_utils.app_utils import is_valid_subdomain_label


def test_deployment_name():
    assert is_valid_subdomain_label("banana")
    assert is_valid_subdomain_label("foo-123-456")
    assert not is_valid_subdomain_label("BaNaNa")
    assert not is_valid_subdomain_label(" ")
    assert not is_valid_subdomain_label("ban/ana")
