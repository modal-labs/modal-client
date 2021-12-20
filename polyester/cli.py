import typer

from .config import config, store_user_config

app = typer.Typer()

token_app = typer.Typer()
app.add_typer(token_app, name="token")
config_app = typer.Typer()
app.add_typer(config_app, name="config")


@token_app.command()
def set(token_id: str, token_secret: str, env: str = None):
    # TODO: let's verify that these creds are good
    store_user_config({"token_id": token_id, "token_secret": token_secret}, env=env)


@config_app.command()
def show():
    # This is just a test command
    print(config)


def main():
    app()


if __name__ == "__main__":
    app()
