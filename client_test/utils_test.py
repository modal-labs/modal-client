from modal_utils.app_utils import is_valid_app_name, is_valid_subdomain_label


def test_subdomain_label():
    assert is_valid_subdomain_label("banana")
    assert is_valid_subdomain_label("foo-123-456")
    assert not is_valid_subdomain_label("BaNaNa")
    assert not is_valid_subdomain_label(" ")
    assert not is_valid_subdomain_label("ban/ana")


def test_app_name():
    assert is_valid_app_name("baNaNa")
    assert is_valid_app_name("foo-123_456")
    assert is_valid_app_name("a" * 64)
    assert not is_valid_app_name("hello world")
    assert not is_valid_app_name("a" * 65)
