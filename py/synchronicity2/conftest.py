# Copyright Modal Labs 2026
"""Root pytest configuration: vendored ``mylib.synchronicity2`` and README weather wrappers."""

from __future__ import annotations

import pytest
import shutil
import subprocess
import sys
from pathlib import Path

from test.integration.test_utils import GENERATED_DIR, codegen_command, codegen_env

MYLIB_SYNCHRONICITY2 = "mylib.synchronicity2"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _ensure_mylib_synchronicity2_under_generated(generated: Path) -> None:
    """Populate ``generated/mylib/synchronicity2/``."""
    generated.mkdir(parents=True, exist_ok=True)
    from synchronicity2.codegen.runtime_vendor import vendor_runtime

    vendor_runtime(target_package=MYLIB_SYNCHRONICITY2, output_base=generated)


def _sync_weather_impl_into_generated(generated: Path) -> None:
    """Copy ``support_files/mylib/_weather_impl.py`` into ``generated/mylib/`` (single import root)."""
    root = _repo_root()
    src = root / "test" / "support_files" / "mylib" / "_weather_impl.py"
    dest_dir = generated / "mylib"
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_dir / "_weather_impl.py")


def _generate_mylib_weather_wrappers(generated: Path) -> None:
    """Compile ``mylib.weather`` from the support-file impl copied under ``generated/mylib/``."""
    root = _repo_root()
    support = root / "test" / "support_files"
    _ensure_mylib_synchronicity2_under_generated(generated)
    _sync_weather_impl_into_generated(generated)
    # generated first so ``mylib`` resolves to vendored runtime + impl + generated wrappers
    env = codegen_env(str(generated), str(support), str(root))
    result = subprocess.run(
        [
            *codegen_command(),
            "wrappers",
            "-m",
            "mylib._weather_impl",
            "--runtime-package",
            MYLIB_SYNCHRONICITY2,
            "-o",
            str(generated),
        ],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to generate mylib.weather wrappers:\n{result.stderr}\n{result.stdout}")


def _ensure_generated_on_sys_path(generated: Path) -> None:
    g = str(generated)
    if g not in sys.path:
        sys.path.insert(0, g)


@pytest.hookimpl(optionalhook=True)
def pytest_markdown_docs_globals():
    """Vendor + generate README ``mylib`` examples for ``pytest --markdown-docs``."""
    generated = GENERATED_DIR
    if not (generated / "mylib" / "weather.py").is_file():
        _generate_mylib_weather_wrappers(generated)
    _ensure_generated_on_sys_path(generated)
    support = str(_repo_root() / "test" / "support_files")
    if support not in sys.path:
        sys.path.insert(0, support)
    return {}


@pytest.fixture(scope="session", autouse=True)
def _vendor_mylib_synchronicity2_for_generated_dir():
    """Autouse: keep ``generated/mylib/synchronicity2/`` present (integration may wipe and rebuild)."""
    generated = GENERATED_DIR
    _ensure_mylib_synchronicity2_under_generated(generated)
    _ensure_generated_on_sys_path(generated)
    yield
