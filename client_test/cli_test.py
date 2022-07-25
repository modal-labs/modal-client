import pytest

import typer.testing

from modal import cli
from modal.client import Client

dummy_app_file = """
import modal

stub = modal.Stub("my_app")
"""


@pytest.fixture
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
        res = runner.invoke(cli.app_app, ["deploy", "myapp.py"])
        assert res.exit_code == 0

    assert "my_app" in servicer.deployed_apps


def test_app_deploy_with_name(servicer, mock_dir, monkeypatch, set_env_client):
    runner = typer.testing.CliRunner()
    with mock_dir({"myapp.py": dummy_app_file}) as root_dir:
        monkeypatch.chdir(root_dir)
        res = runner.invoke(cli.app_app, ["deploy", "myapp.py", "--name", "my_app_foo"])
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
        res = runner.invoke(cli.app_app, ["deploy", "myaioapp.py"])
        assert res.exit_code == 0

    assert "my_aio_app" in servicer.deployed_apps


def test_app_deploy_no_such_module():
    runner = typer.testing.CliRunner()
    res = runner.invoke(cli.app_app, ["deploy", "does_not_exist.py"])
    assert res.exit_code == 1
    assert "No module named" in res.stdout
