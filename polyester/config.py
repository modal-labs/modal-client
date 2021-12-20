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


def read_user_config():
    if os.path.exists(_user_config_path):
        return toml.load(open(_user_config_path))
    else:
        return {}


_user_config = read_user_config()
_env = os.environ.get("POLYESTER_ENV", "default")


class Config(typing.NamedTuple):
    key: str
    default: typing.Any = None
    transform: typing.Callable[[str], typing.Any] = lambda x: x


_CONFIG = [
    Config("loglevel", "WARNING", lambda s: s.upper()),
    Config("server_url", "https://api.modal.com"),
    Config("token_id"),
    Config("token_secret"),
    Config("task_id"),
    Config("task_secret"),
    Config("sync_entrypoint"),
    Config("logs_timeout", 10, float),
    Config("image_python_version"),
]


config = {}

for c in _CONFIG:
    env_var_key = "POLYESTER_" + c.key.upper()
    if env_var_key in os.environ:
        config[c.key] = c.transform(os.environ[env_var_key])
    elif _env in _user_config and c.key in _user_config[_env]:
        config[c.key] = c.transform(_user_config[_env][c.key])
    else:
        config[c.key] = c.default


logging.basicConfig(
    level=config["loglevel"], format="%(threadName)s %(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z"
)
logger = logging.getLogger()


def store_user_config(new_settings, env=_env):
    user_config = read_user_config()
    user_config.setdefault(env, {}).update(**new_settings)
    with open(_user_config_path, "w") as f:
        toml.dump(user_config, f)
