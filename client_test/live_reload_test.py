# Copyright Modal Labs 2023
import copy
import pytest
from pathlib import Path

from watchfiles import Change

from modal._watcher import AppChange


@pytest.fixture(scope="function")
def clean_up_file_changes():
    change_reversions: dict[Path, str] = {}
    yield change_reversions
    for path, contents in change_reversions.items():
        path.write_text(contents)


def dummy():
    pass


def test_reload_of_function_defs(clean_up_file_changes, client, monkeypatch, servicer, test_dir):
    from client_test.supports.live_reload_tests.webhook import stub

    stub_file = test_dir / "supports" / "live_reload_tests" / "webhook.py"
    app_functions_snapshots = []

    def make_revertable_file_modification(path: Path, new_contents: str) -> None:
        if path not in clean_up_file_changes:
            clean_up_file_changes[path] = path.read_text()
        path.write_text(new_contents)

    async def fake_watch(stub, output_mgr, timeout):
        """
        Provide a fixed list of events to stub.serve, recording the state of the app's
        Modal functions in between events.
        """
        # begin serving
        yield AppChange.START
        app_functions_snapshots.append(copy.deepcopy(servicer.app_functions))
        # Change the served module
        make_revertable_file_modification(
            stub_file,
            new_contents="""
import modal
stub = modal.Stub()

@stub.webhook(method="POST")  # change method from GET to POST
def dummy():
    pass
        """,
        )
        yield {(Change.modified, str(stub_file))}
        app_functions_snapshots.append(copy.deepcopy(servicer.app_functions))
        # stop the served app
        yield AppChange.TIMEOUT

    monkeypatch.setattr("modal._watcher.watch", fake_watch)

    stub.serve(client=client, timeout=None)

    assert len(app_functions_snapshots) == 2
    assert app_functions_snapshots[0]["fu-1"].webhook_config.method == "GET"
    assert app_functions_snapshots[1]["fu-1"].webhook_config.method == "POST"


def test_reload_on_invalid_syntax(clean_up_file_changes, client, monkeypatch, servicer, test_dir):
    import importlib

    # HACK: There is some test interference here, probably caused by the use of exec/compile in live-reloading.
    importlib.import_module("client_test.supports.live_reload_tests.syntax")
    from client_test.supports.live_reload_tests.syntax import stub

    stub_file = test_dir / "supports" / "live_reload_tests" / "syntax.py"
    app_functions_snapshots = []

    def make_revertable_file_modification(path: Path, new_contents: str) -> None:
        if path not in clean_up_file_changes:
            clean_up_file_changes[path] = path.read_text()
        path.write_text(new_contents)

    async def fake_watch(stub, output_mgr, timeout):
        """
        Provide a fixed list of events to stub.serve, recording the state of the app's
        Modal functions in between events.
        """
        # begin serving
        yield AppChange.START
        app_functions_snapshots.append(copy.deepcopy(servicer.app_functions))
        # Change the served module but introduce a syntax error
        make_revertable_file_modification(
            stub_file,
            new_contents="""
import modal
stub = modal.Stub()

@stub.webhook(method="GET"))  # )) syntax error
def dummy():
    pass
        """,
        )
        yield {(Change.modified, str(stub_file))}
        app_functions_snapshots.append(copy.deepcopy(servicer.app_functions))
        # Change the served module to include a valid AST but invalid Python syntax.
        make_revertable_file_modification(
            stub_file,
            new_contents="return 42",
        )
        yield {(Change.modified, str(stub_file))}
        app_functions_snapshots.append(copy.deepcopy(servicer.app_functions))
        # stop the served app
        yield AppChange.TIMEOUT

    monkeypatch.setattr("modal._watcher.watch", fake_watch)

    stub.serve(client=client, timeout=None)

    assert len(app_functions_snapshots) == 3
    assert all(snap["fu-1"].webhook_config.method == "DELETE" for snap in app_functions_snapshots)
