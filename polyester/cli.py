import typer

from .config import config, store_user_config

app = typer.Typer()
sub_apps = {}
for cmd in ["token", "config"]:
    sub_apps[cmd] = typer.Typer()
    app.add_typer(sub_apps[cmd], name=cmd)


@sub_apps["token"].command()
def set(token_id: str, token_secret: str, env: str = "default"):
    # TODO: let's verify that these creds are good
    store_user_config({"token_id": token_id, "token_secret": token_secret}, env=env)


@sub_apps["config"].command()
def show():
    # This is just a test command
    print(config)


def main():
    app()


if __name__ == "__main__":
    app()
