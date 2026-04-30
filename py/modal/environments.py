# Copyright Modal Labs 2025
from ._environments import (
    _create_environment,
    _delete_environment,
    _Environment,
    _EnvironmentManager,
    _EnvironmentMembersManager,
    _list_environments,
    _update_environment,
)
from ._utils.async_utils import synchronize_api, synchronizer

EnvironmentManager = synchronize_api(_EnvironmentManager, target_module=__name__)
EnvironmentMembersManager = synchronize_api(_EnvironmentMembersManager, target_module=__name__)
Environment = synchronize_api(_Environment, target_module=__name__)

# These functions are structurally internal (they deal in protobuf types) but users have come
# to use them as they did not have a private naming convention, and we didn't have an alternate
# public Environment for some time. We will deprecate them gracefully as we build out Environment.
create_environment = synchronizer.create_blocking(_create_environment, name="create_environment")
delete_environment = synchronizer.create_blocking(_delete_environment, name="delete_environment")
list_environments = synchronizer.create_blocking(_list_environments, name="list_environments")
update_environment = synchronizer.create_blocking(_update_environment, name="update_environment")

__all__ = [
    "Environment",
    "create_environment",
    "delete_environment",
    "list_environments",
    "update_environment",
]
