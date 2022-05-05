import typer.testing

import modal_proto.api_pb2_grpc
from modal import cli

dummy_app_file = """
import modal

app = modal.App("my-app")
"""


def test_app_deploy_success(servicer, mock_dir, monkeypatch):
    monkeypatch.setattr(modal_proto.api_pb2_grpc, "ModalClientStub", lambda _: servicer)

    runner = typer.testing.CliRunner()
    with mock_dir({"myapp.py": dummy_app_file}) as root_dir:
        monkeypatch.chdir(root_dir)
        res = runner.invoke(cli.app_app, ["deploy", "myapp.py"])
        assert res.exit_code == 0

    assert "my-app" in servicer.deployed_apps


dummy_aio_app_file = """
from modal.aio import AioApp

app = AioApp("my-aio-app")
"""


def test_aio_app_deploy_success(servicer, mock_dir, monkeypatch):
    monkeypatch.setattr(modal_proto.api_pb2_grpc, "ModalClientStub", lambda _: servicer)

    runner = typer.testing.CliRunner()
    with mock_dir({"myaioapp.py": dummy_aio_app_file}) as root_dir:
        monkeypatch.chdir(root_dir)
        res = runner.invoke(cli.app_app, ["deploy", "myaioapp.py"])
        assert res.exit_code == 0

    assert "my-aio-app" in servicer.deployed_apps


def test_app_deploy_no_such_module():
    runner = typer.testing.CliRunner()
    res = runner.invoke(cli.app_app, ["deploy", "does_not_exist.py"])
    assert res.exit_code == 1
    assert "No module named" in res.stdout
