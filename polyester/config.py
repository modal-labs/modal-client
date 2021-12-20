import logging
import os
import typing

import toml

try:
    import icecream

    icecream.install()
except ImportError:
    pass


_user_config_path = os.environ.get("POLYESTER_CONFIG_PATH")
if _user_config_path is None:
    _user_config_path = os.path.expanduser("~/.polyester.toml")


def _read_user_config():
    if os.path.exists(_user_config_path):
        return toml.load(open(_user_config_path))
    else:
        return {}


_user_config = _read_user_config()
_env = os.environ.get("POLYESTER_ENV", "default")


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
}


class Config:
    def __init__(self):
        pass

    def get(self, key, env=None):
        """Looks up a configuration value.

        Will check (in decreasing order of priority):
        1. Any environment variable of the form POLYESTER_FOO_BAR
        2. Settings in the user's .toml configuration file
        3. The default value of the setting
        """
        if env is None:
            env = _env
        s = _SETTINGS[key]
        env_var_key = "POLYESTER_" + key.upper()
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
    """Used by the CLI to store values."""
    if env is None:
        env = _env
    user_config = _read_user_config()
    user_config.setdefault(env, {}).update(**new_settings)
    with open(_user_config_path, "w") as f:
        toml.dump(user_config, f)
