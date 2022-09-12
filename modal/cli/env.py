import typer

from modal.config import _env, config_envs, config_set_active_env

env_cli = typer.Typer(name="env", help="Set the current environment.", no_args_is_help=True)


@env_cli.command(help="Change the active Modal environment.")
def activate(env: str = typer.Argument(..., help="Modal environment to activate.")):
    config_set_active_env(env)


@env_cli.command(help="Print the active Modal environment.")
def current():
    print(_env)


@env_cli.command(help="List all Modal environments that are defined.")
def list():
    for env in config_envs():
        print(f"{env} [active]" if _env == env else env)
