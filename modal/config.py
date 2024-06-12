# Copyright Modal Labs 2022
r"""Modal intentionally keeps configurability to a minimum.

The main configuration options are the API tokens: the token id and the token secret.
These can be configured in two ways:

1. By running the `modal token set` command.
   This writes the tokens to `.modal.toml` file in your home directory.
2. By setting the environment variables `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`.
   This takes precedence over the previous method.

.modal.toml
---------------

The `.modal.toml` file is generally stored in your home directory.
It should look like this::

```toml
[default]
token_id = "ak-12345..."
token_secret = "as-12345..."
```

You can create this file manually, or you can run the `modal token set ...`
command (see below).

Setting tokens using the CLI
----------------------------

You can set a token by running the command::

```bash
modal token set \
  --token-id <token id> \
  --token-secret <token secret>
```

This will write the token id and secret to `.modal.toml`.

If the token id or secret is provided as the string `-` (a single dash),
then it will be read in a secret way from stdin instead.

Other configuration options
---------------------------

Other possible configuration options are:

* `loglevel` (in the .toml file) / `MODAL_LOGLEVEL` (as an env var).
  Defaults to `WARNING`. Set this to `DEBUG` to see internal messages.
* `logs_timeout` (in the .toml file) / `MODAL_LOGS_TIMEOUT` (as an env var).
  Defaults to 10.
  Number of seconds to wait for logs to drain when closing the session,
  before giving up.
* `automount` (in the .toml file) / `MODAL_AUTOMOUNT` (as an env var).
  Defaults to True.
  By default, Modal automatically mounts modules imported in the current scope, that
  are deemed to be "local". This can be turned off by setting this to False.
* `force_build` (in the .toml file) / `MODAL_FORCE_BUILD` (as an env var).
  Defaults to False.
  When set, ignores the Image cache and builds all Image layers. Note that this
  will break the cache for all images based on the rebuilt layers, so other images
  may rebuild on subsequent runs / deploys even if the config is reverted.
* `traceback` (in the .toml file) / `MODAL_TRACEBACK` (as an env var).
  Defaults to False. Enables printing full tracebacks on unexpected CLI
  errors, which can be useful for debugging client issues.

Meta-configuration
------------------

Some "meta-options" are set using environment variables only:

* `MODAL_CONFIG_PATH` lets you override the location of the .toml file,
  by default `~/.modal.toml`.
* `MODAL_PROFILE` lets you use multiple sections in the .toml file
  and switch between them. It defaults to "default".
"""

import logging
import os
import typing
import warnings
from textwrap import dedent
from typing import Any, Dict, Optional

from google.protobuf.empty_pb2 import Empty

from modal_proto import api_pb2

from ._utils.logger import configure_logger
from .exception import InvalidError, deprecation_warning

# Locate config file and read it

user_config_path: str = os.environ.get("MODAL_CONFIG_PATH") or os.path.expanduser("~/.modal.toml")


def _is_remote() -> bool:
    # We want to prevent read/write on a modal config file in the container
    # environment, both because that doesn't make sense and might cause weird
    # behavior, and because we want to keep the `toml` dependency out of the
    # container runtime.
    return os.environ.get("MODAL_IS_REMOTE") == "1"


def _read_user_config():
    if not _is_remote() and os.path.exists(user_config_path):
        # Defer toml import so we don't need it in the container runtime environment
        import toml

        with open(user_config_path) as f:
            return toml.load(f)
    else:
        return {}


_user_config = _read_user_config()


async def _lookup_workspace(server_url: str, token_id: str, token_secret: str) -> api_pb2.WorkspaceNameLookupResponse:
    from .client import _Client

    credentials = (token_id, token_secret)
    async with _Client(server_url, api_pb2.CLIENT_TYPE_CLIENT, credentials) as client:
        return await client.stub.WorkspaceNameLookup(Empty())


def config_profiles():
    """List the available modal profiles in the .modal.toml file."""
    return _user_config.keys()


def _config_active_profile() -> str:
    for key, values in _user_config.items():
        if values.get("active", False) is True:
            return key
    else:
        return "default"


def config_set_active_profile(env: str) -> None:
    """Set the user's active modal profile by writing it to the `.modal.toml` file."""
    if env not in _user_config:
        raise KeyError(env)

    for key, values in _user_config.items():
        values.pop("active", None)

    _user_config[env]["active"] = True
    _write_user_config(_user_config)


def _check_config() -> None:
    num_profiles = len(_user_config)
    num_active = sum(v.get("active", False) for v in _user_config.values())
    if num_active > 1:
        raise InvalidError(
            "More than one Modal profile is active. "
            "Please fix with `modal profile activate` or by editing your Modal config file "
            f"({user_config_path})."
        )
    elif num_profiles > 1 and num_active == 0 and _profile == "default":
        # Eventually we plan to have num_profiles > 1 with num_active = 0 be an error
        # But we want to give users time to activate one of their profiles without disruption
        message = dedent(
            """
            Support for using an implicit 'default' profile is deprecated.
            Please use `modal profile activate` to activate one of your profiles.
            (Use `modal profile list` to see the options.)

            This will become an error in a future update.
            """
        )
        deprecation_warning((2024, 2, 6), message, show_source=False)


_profile = os.environ.get("MODAL_PROFILE") or _config_active_profile()

# Define settings


def _to_boolean(x: object) -> bool:
    return str(x).lower() not in {"", "0", "false"}


class _Setting(typing.NamedTuple):
    default: typing.Any = None
    transform: typing.Callable[[str], typing.Any] = lambda x: x  # noqa: E731


_SETTINGS = {
    "loglevel": _Setting("WARNING", lambda s: s.upper()),
    "log_format": _Setting("STRING", lambda s: s.upper()),
    "server_url": _Setting("https://api.modal.com"),
    "token_id": _Setting(),
    "token_secret": _Setting(),
    "task_id": _Setting(),
    "task_secret": _Setting(),
    "serve_timeout": _Setting(transform=float),
    "sync_entrypoint": _Setting(),
    "logs_timeout": _Setting(10, float),
    "image_id": _Setting(),
    "automount": _Setting(True, transform=_to_boolean),
    "heartbeat_interval": _Setting(15, float),
    "function_runtime": _Setting(),
    "function_runtime_debug": _Setting(False, transform=_to_boolean),  # For internal debugging use.
    "environment": _Setting(),
    "default_cloud": _Setting(None, transform=lambda x: x if x else None),
    "worker_id": _Setting(),  # For internal debugging use.
    "restore_state_path": _Setting("/__modal/restore-state.json"),
    "force_build": _Setting(False, transform=_to_boolean),
    "traceback": _Setting(False, transform=_to_boolean),
    "image_builder_version": _Setting(),
}


class Config:
    """Singleton that holds configuration used by Modal internally."""

    def __init__(self):
        pass

    def get(self, key, profile=None, use_env=True):
        """Looks up a configuration value.

        Will check (in decreasing order of priority):
        1. Any environment variable of the form MODAL_FOO_BAR (when use_env is True)
        2. Settings in the user's .toml configuration file
        3. The default value of the setting
        """
        if profile is None:
            profile = _profile
        s = _SETTINGS[key]
        env_var_key = "MODAL_" + key.upper()
        if use_env and env_var_key in os.environ:
            return s.transform(os.environ[env_var_key])
        elif profile in _user_config and key in _user_config[profile]:
            return s.transform(_user_config[profile][key])
        else:
            return s.default

    def override_locally(self, key: str, value: str):
        # Override setting in this process by overriding environment variable for the setting
        #
        # Does NOT write back to settings file etc.
        try:
            self.get(key)
            os.environ["MODAL_" + key.upper()] = value
        except KeyError:
            # Override env vars not available in config, e.g. NVIDIA_VISIBLE_DEVICES.
            # This is used for restoring env vars from a memory snapshot.
            os.environ[key.upper()] = value

    def __getitem__(self, key):
        return self.get(key)

    def __repr__(self):
        return repr(self.to_dict())

    def to_dict(self):
        return {key: self.get(key) for key in _SETTINGS.keys()}


config = Config()

# Logging

logger = logging.getLogger("modal-client")
configure_logger(logger, config["loglevel"], config["log_format"])

# Utils to write config


def _store_user_config(
    new_settings: Dict[str, Any], profile: Optional[str] = None, active_profile: Optional[str] = None
):
    """Internal method, used by the CLI to set tokens."""
    if profile is None:
        profile = _profile
    user_config = _read_user_config()
    user_config.setdefault(profile, {}).update(**new_settings)
    if active_profile is not None:
        for prof_name, prof_config in user_config.items():
            if prof_name == active_profile:
                prof_config["active"] = True
            else:
                prof_config.pop("active", None)
    _write_user_config(user_config)


def _write_user_config(user_config):
    if _is_remote():
        raise InvalidError("Can't update config file in remote environment.")

    # Defer toml import so we don't need it in the container runtime environment
    import toml

    with open(user_config_path, "w") as f:
        toml.dump(user_config, f)


# Make sure all deprecation warnings are shown
# See https://docs.python.org/3/library/warnings.html#overriding-the-default-filter
warnings.filterwarnings(
    "default",
    category=DeprecationWarning,
    module="modal",
)
