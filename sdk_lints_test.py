import importlib.util
import pytest
from pathlib import Path

import invoke

CLIENT_ROOT = Path(__file__).parent


def _load_tasks(tasks_dir: Path, module_name: str):
    # Both task files are named `tasks.py`, so load them under distinct names.
    spec = importlib.util.spec_from_file_location(module_name, tasks_dir / "tasks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "tasks_dir, task_name",
    [
        (CLIENT_ROOT, "lint_protos"),
        (CLIENT_ROOT, "lint_versions"),
        (CLIENT_ROOT, "lint_changelogs"),
        (CLIENT_ROOT / "py", "check_copyright"),
    ],
)
def test_client_sdk_lint(tasks_dir, task_name, monkeypatch):
    tasks = _load_tasks(tasks_dir, f"{tasks_dir.name}_tasks")
    monkeypatch.chdir(tasks_dir)
    getattr(tasks, task_name)(invoke.Context())


def test_copyright_check_sees_client_sources():
    # Guards against the Bazel data dependencies silently dropping client/py sources.
    py_root = CLIENT_ROOT / "py"
    packages = {path.relative_to(py_root).parts[0] for path in py_root.glob("*/**/*.py")}
    assert {"modal", "modal_docs", "modal_global_objects", "modal_version", "synchronicity2", "test"} <= packages
