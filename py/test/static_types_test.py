# Copyright Modal Labs 2024
import importlib
import os
import pkgutil
import pytest
import shutil
import subprocess
import sys
from pathlib import Path

import modal
from test.supports.skip import skip_windows

project_root = Path(modal.__file__).parent.parent
supports_dir = Path(__file__).parent / "supports"


def _find_modal_modules(root: str = "modal") -> list[str]:
    modules = []
    path = importlib.import_module(root).__path__
    for _, name, is_pkg in pkgutil.iter_modules(path):
        full_name = f"{root}.{name}"
        if is_pkg:
            modules.extend(_find_modal_modules(full_name))
        else:
            modules.append(full_name)
    return modules


def _has_wrapped_types(module_name: str) -> bool:
    from synchronicity.synchronizer import SYNCHRONIZER_ATTR

    if module_name.startswith("modal.cli."):  # CLI handlers are not called into directly
        return False
    module = importlib.import_module(module_name)
    return any(
        hasattr(obj, "__module__")
        and obj.__module__ == module_name
        and not name.startswith("_")  # Avoid deprecation of _App.__getattr__
        and hasattr(obj, SYNCHRONIZER_ATTR)
        for name, obj in vars(module).items()
    )


@pytest.fixture(scope="module")
def type_check_root(tmp_path_factory) -> Path:
    """A scratch project containing a copy of the `modal` package with generated type stubs.

    Stubs are generated into a copy rather than in place so the test works when the installed
    package is read-only (e.g. inside a Bazel sandbox) and never dirties the source tree.
    """
    root = tmp_path_factory.mktemp("static_types")
    shutil.copytree(
        Path(modal.__file__).parent,
        root / "modal",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyi"),
    )
    shutil.copy(project_root / "pyproject.toml", root / "pyproject.toml")
    (root / "test" / "supports").mkdir(parents=True)
    for name in ("type_assertions.py", "type_assertions_negative.py"):
        shutil.copy(supports_dir / name, root / "test" / "supports" / name)

    modules = [m for m in _find_modal_modules() if _has_wrapped_types(m)]
    # The copy shadows the installed package so the stubs are written next to the copied modules.
    p = subprocess.run(
        [sys.executable, "-m", "synchronicity.type_stubs", *modules],
        cwd=root,
        env=_subprocess_env(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf8",
    )
    assert p.returncode == 0, f"type stub generation failed:\n{p.stdout}"
    return root


def _subprocess_env(root: Path) -> dict[str, str]:
    env = {**os.environ, "MYPYPATH": str(root)}
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(root), env.get("PYTHONPATH")]))
    return env


def _run_mypy(root: Path, target: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "mypy", "--config-file", "pyproject.toml", target],
        cwd=root,
        env=_subprocess_env(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf8",
    )


@pytest.mark.skipif(sys.version_info[:2] >= (3, 14), reason="type stub generation is broken in Python 3.14+")
@skip_windows("Type tests fail on windows since they don't exclude non-windows features")
def test_remote_call_keeps_original_return_value(type_check_root):
    p = _run_mypy(type_check_root, "test/supports/type_assertions.py")
    print(p.stdout)
    assert p.returncode == 0


@pytest.mark.skipif(sys.version_info[:2] >= (3, 14), reason="type stub generation is broken in Python 3.14+")
@skip_windows("Type tests fail on windows since they don't exclude non-windows features")
def test_negative_assertions(type_check_root):
    p = _run_mypy(type_check_root, "test/supports/type_assertions_negative.py")
    stdout = p.stdout
    print(stdout)
    assert p.returncode == 1
    assert "Found 6 errors in 1 file" in stdout
    assert 'Unexpected keyword argument "b" for "__call__"' in stdout
    assert 'Argument "a" to "__call__" of "__remote_spec" has incompatible type "int"' in stdout
    assert 'Unexpected keyword argument "c" for "local" of "Function"' in stdout
    assert 'Argument "a" to "local" of "Function" has incompatible type "int"' in stdout
    assert 'Unexpected keyword argument "e" for "aio" of "__remote_spec"' in stdout
    assert 'Argument "a" to "aio" of "__remote_spec" has incompatible type "float"' in stdout
