import typer

from .config import config, store_user_config

app = typer.Typer()


@app.command()
def creds_set(token_id: str, token_secret: str, env: str = "default"):
    # TODO: let's verify that these creds are good
    store_user_config({"token_id": token_id, "token_secret": token_secret}, env=env)


@app.command()
def config_show():
    # This is just a test command
    print(config)


def main():
    app()


if __name__ == "__main__":
    app()
