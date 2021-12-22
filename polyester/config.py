"""
Configuration
=============

Polyester intentionally keeps configurability to a minimum.
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

  [default]
  token_id = "ak-12345..."
  token_secret = "as-12345..."

You can create this file manually, or you can run the ``modal token set ...``
command (see below).

Setting tokens using the CLI
----------------------------

You can set a token by running the command::

  modal token set \\
      --token-id <token id> \\
      --token-secret <token secret>

This will write the token id and secret to ``.modal.toml``.

If the token id or secret is provided as the string ``-`` (a single dash),
then it will be read in a secret way from stdin instead.

Other configuration options
---------------------------

Other possible configuration options are:

* ``loglevel`` (in the .toml file) / ``MODAL_LOGLEVEL`` (as an env var).
  Defaults to ``WARNING``.
  Set this to ``DEBUG`` to see a bunch of internal output.
* ``server_url`` (in the .toml file) / ``MODAL_SERVER_URL`` (as an env var).
  Defaults to ``https://api.modal.com``.
  Not typically meant to be used.

Meta-configuration
------------------

Some "meta-options" are set using environment variables only:

* ``MODAL_CONFIG_PATH`` lets you override the location of the .toml file,
  by default ``~/.modal.toml``.
* ``MODAL_ENVIRONMENT`` lets you use multiple sections in the .toml file
  and switch between them. It defaults to "default".
"""

import logging
import os
import typing

import toml

try:
    import icecream

    icecream.install()
except ImportError:
    pass


_user_config_path = os.environ.get("MODAL_CONFIG_PATH")
if _user_config_path is None:
    _user_config_path = os.path.expanduser("~/.modal.toml")


def _read_user_config():
    if os.path.exists(_user_config_path):
        return toml.load(open(_user_config_path))
    else:
        return {}


_user_config = _read_user_config()
_env = os.environ.get("MODAL_ENV", "default")


class Setting(typing.NamedTuple):
    default: typing.Any = None
    transform: typing.Callable[[str], typing.Any] = lambda x: x


_SETTINGS = {
    "loglevel": Setting("WARNING", lambda s: s.upper()),
    "server_url": Setting("https://api.modal.com"),
    "token_id": Setting(),
    "token_secret": Setting(),
    "task_id": Setting(),
    "task_secret": Setting(),
    "sync_entrypoint": Setting(),
    "logs_timeout": Setting(10, float),
    "image_python_version": Setting(),
    "image_id": Setting(),
}


class Config:
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

logging.basicConfig(
    level=config["loglevel"], format="%(threadName)s %(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z"
)
logger = logging.getLogger()


def store_user_config(new_settings, env=None):
    """Internal method, used by the CLI to set tokens."""
    if env is None:
        env = _env
    user_config = _read_user_config()
    user_config.setdefault(env, {}).update(**new_settings)
    with open(_user_config_path, "w") as f:
        toml.dump(user_config, f)
