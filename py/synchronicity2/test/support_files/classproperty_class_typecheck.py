# Copyright Modal Labs 2026
"""Consumer typing checks for generated classproperty_class wrappers."""

import classproperty_class
from typing_extensions import assert_type


def _sync_usage() -> None:
    assert_type(classproperty_class.Service.manager, classproperty_class.Manager)
    assert_type(classproperty_class.Service.manager.describe(), str)
    assert_type(classproperty_class.Service.default_name, str)

    service = classproperty_class.Service()
    assert_type(service.manager, classproperty_class.Manager)
    assert_type(service.default_name, str)
