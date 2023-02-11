# Copyright Modal Labs 2022
"""Modal intentionally keeps configurability to a minimum.

The main configuration options are the API tokens: the token id and the token secret.
These can be configured in two ways:

1. By running the ``modal token set`` command.
   This writes the tokens to ``.modal.toml`` file in your home directory.
2. By setting the environment variables ``MODAL_TOKEN_ID`` and ``MODAL_TOKEN_SECRET``.
   This takes precedence over the previous method.

.modal.toml
---------------

The ``.modal.toml`` file is generally stored in your home directory.
It should look like this::

```toml
[default]
token_id = "ak-12345..."
token_secret = "as-12345..."
```

You can create this file manually, or you can run the ``modal token set ...``
command (see below).

Setting tokens using the CLI
----------------------------

You can set a token by running the command::

```bash
modal token set \
  --token-id <token id> \
  --token-secret <token secret>
```

This will write the token id and secret to ``.modal.toml``.

If the token id or secret is provided as the string ``-`` (a single dash),
then it will be read in a secret way from stdin instead.

Other configuration options
---------------------------

Other possible configuration options are:

* ``loglevel`` (in the .toml file) / ``MODAL_LOGLEVEL`` (as an env var).
  Defaults to ``WARNING``.
  Set this to ``DEBUG`` to see a bunch of internal output.
* ``logs_timeout`` (in the .toml file) / ``MODAL_LOGS_TIMEOUT`` (as an env var).
  Defaults to 10.
  Number of seconds to wait for logs to drain when closing the session,
  before giving up.
* ``automount`` (in the .toml file) / ``MODAL_AUTOMOUNT`` (as an env var).
  Defaults to True.
  By default, Modal automatically mounts modules imported in the current scope, that
  are deemed to be "local". This can be turned off by setting this to False.
* ``server_url`` (in the .toml file) / ``MODAL_SERVER_URL`` (as an env var).
  Defaults to ``https://api.modal.com``.
  Not typically meant to be used.

Meta-configuration
------------------

Some "meta-options" are set using environment variables only:

* ``MODAL_CONFIG_PATH`` lets you override the location of the .toml file,
  by default ``~/.modal.toml``.
* ``MODAL_ENV`` lets you use multiple sections in the .toml file
  and switch between them. It defaults to "default".
"""

import logging
import os
import typing
import warnings

import toml


from ._traceback import setup_rich_traceback

# Locate config file and read it

user_config_path: str = os.environ.get("MODAL_CONFIG_PATH") or os.path.expanduser("~/.modal.toml")


def _read_user_config():
    if os.path.exists(user_config_path):
        with open(user_config_path) as f:
            return toml.load(f)
    else:
        return {}


_user_config = _read_user_config()


def config_envs():
    """List the available modal envs in the .modal.toml file."""
    return _user_config.keys()


def _config_active_env():
    for key, values in _user_config.items():
        if values.get("active", False) is True:
            return key
    else:
        return "default"


def config_set_active_env(env: str):
    """Set the user's active modal env by writing it to the `.modal.toml` file."""
    if env not in _user_config:
        raise KeyError(env)

    for key, values in _user_config.items():
        values.pop("active", None)

    _user_config[env]["active"] = True
    _write_user_config(_user_config)


_env = os.environ.get("MODAL_ENV", _config_active_env())

# Define settings


class _Setting(typing.NamedTuple):
    default: typing.Any = None
    transform: typing.Callable[[str], typing.Any] = lambda x: x  # noqa: E731


_SETTINGS = {
    "loglevel": _Setting("WARNING", lambda s: s.upper()),
    "server_url": _Setting("https://api.modal.com"),
    "token_id": _Setting(),
    "token_secret": _Setting(),
    "task_id": _Setting(),
    "task_secret": _Setting(),
    "sync_entrypoint": _Setting(),
    "logs_timeout": _Setting(10, float),
    "image_python_version": _Setting(),
    "image_id": _Setting(),
    "serve_timeout": _Setting(transform=float),
    "automount": _Setting(True, transform=lambda x: x not in ("", "0")),
    "tracing_enabled": _Setting(False, transform=lambda x: x not in ("", "0")),
    "profiling_enabled": _Setting(False, transform=lambda x: x not in ("", "0")),
}


class Config:
    """Singleton that holds configuration used by Modal internally."""

    def __init__(self):
        pass

    def get(self, key, env=None):
        """Looks up a configuration value.

        Will check (in decreasing order of priority):
        1. Any environment variable of the form MODAL_FOO_BAR
        2. Settings in the user's .toml configuration file
        3. The default value of the setting
        """
        if env is None:
            env = _env
        s = _SETTINGS[key]
        env_var_key = "MODAL_" + key.upper()
        if env_var_key in os.environ:
            return s.transform(os.environ[env_var_key])
        elif env in _user_config and key in _user_config[env]:
            return s.transform(_user_config[env][key])
        else:
            return s.default

    def __getitem__(self, key):
        return self.get(key)

    def __repr__(self):
        return repr({key: self.get(key) for key in _SETTINGS.keys()})


config = Config()

# Logging

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z")
logger = logging.getLogger("modal-client")
log_level_numeric = logging.getLevelName(config["loglevel"])
logger.setLevel(log_level_numeric)

# Utils to write config


def _store_user_config(new_settings, env=None):
    """Internal method, used by the CLI to set tokens."""
    if env is None:
        env = _env
    user_config = _read_user_config()
    user_config.setdefault(env, {}).update(**new_settings)
    _write_user_config(user_config)


def _write_user_config(user_config):
    with open(user_config_path, "w") as f:
        toml.dump(user_config, f)


# Make sure all deprecation warnings are shown
# See https://docs.python.org/3/library/warnings.html#overriding-the-default-filter
warnings.filterwarnings(
    "default",
    category=DeprecationWarning,
    module="modal",
)

# Set up rich tracebacks, but only on user's end.
if _user_config:
    setup_rich_traceback()
