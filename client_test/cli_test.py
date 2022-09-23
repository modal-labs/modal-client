import pytest_asyncio
import typer.testing

from modal import cli
from modal.client import Client

dummy_app_file = """
import modal

stub = modal.Stub("my_app")
"""


@pytest_asyncio.fixture
async def set_env_client(aio_client):
    try:
        Client.set_env_client(aio_client)
        yield
    finally:
        Client.set_env_client(None)


def test_app_deploy_success(servicer, mock_dir, monkeypatch, set_env_client):
    runner = typer.testing.CliRunner()
    with mock_dir({"myapp.py": dummy_app_file}) as root_dir:
        monkeypatch.chdir(root_dir)
        res = runner.invoke(cli.entrypoint_cli, ["app", "deploy", "myapp.py"])
        assert res.exit_code == 0

    assert "my_app" in servicer.deployed_apps


def test_app_deploy_with_name(servicer, mock_dir, monkeypatch, set_env_client):
    runner = typer.testing.CliRunner()
    with mock_dir({"myapp.py": dummy_app_file}) as root_dir:
        monkeypatch.chdir(root_dir)
        res = runner.invoke(cli.entrypoint_cli, ["app", "deploy", "myapp.py", "--name", "my_app_foo"])
        assert res.exit_code == 0

    assert "my_app_foo" in servicer.deployed_apps


dummy_aio_app_file = """
from modal.aio import AioStub

stub = AioStub("my_aio_app")
"""


def test_aio_app_deploy_success(servicer, mock_dir, monkeypatch, set_env_client):
    runner = typer.testing.CliRunner()
    with mock_dir({"myaioapp.py": dummy_aio_app_file}) as root_dir:
        monkeypatch.chdir(root_dir)
        res = runner.invoke(cli.entrypoint_cli, ["app", "deploy", "myaioapp.py"])
        assert res.exit_code == 0

    assert "my_aio_app" in servicer.deployed_apps


def test_app_deploy_no_such_module():
    runner = typer.testing.CliRunner()
    res = runner.invoke(cli.entrypoint_cli, ["app", "deploy", "does_not_exist.py"])
    assert res.exit_code == 1
    assert "No module named" in res.stdout


def test_secret_list(servicer, set_env_client):
    runner = typer.testing.CliRunner()
    res = runner.invoke(cli.entrypoint_cli, ["secret", "list"])
    assert res.exit_code == 0
    assert "dummy-secret-0" not in res.stdout
    servicer.created_secrets = 2

    res = runner.invoke(cli.entrypoint_cli, ["secret", "list"])
    assert res.exit_code == 0
    assert "dummy-secret-0" in res.stdout
    assert "dummy-secret-1" in res.stdout


def test_secret_create(servicer, set_env_client):
    runner = typer.testing.CliRunner()
    res = runner.invoke(cli.entrypoint_cli, ["secret", "create", "foo"])
    assert res.exit_code == 2  # fail without any keys

    res = runner.invoke(cli.entrypoint_cli, ["secret", "create", "foo", "bar=baz"])
    assert res.exit_code == 0
    assert servicer.created_secrets == 1
