# Copyright Modal Labs 2026
"""Tests for vendoring the runtime and ``--runtime-package`` / ``compile_modules``.

These are infrastructure tests and intentionally do not follow the four-tier
runtime / pyright (impl, wrapper, usage) layout used by scenario integration tests.
"""

import subprocess
import sys
from pathlib import Path

from synchronicity2.codegen.runtime_vendor import vendor_runtime
from test.integration.test_utils import codegen_command, codegen_env


def _run_codegen_wrappers(module_name: str, module_dir: Path) -> subprocess.CompletedProcess[str]:
    project_root = Path(__file__).parent.parent.parent
    env = codegen_env(str(module_dir), str(project_root))

    return subprocess.run(
        [
            *codegen_command(),
            "wrappers",
            "-m",
            module_name,
            "--synchronizer-module",
            "test_generated_synchronizer",
            "--stdout",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(project_root),
    )


def test_vendor_runtime_writes_expected_files(tmp_path: Path) -> None:
    dest = vendor_runtime(target_package="vendored_test.sync_rt", output_base=tmp_path)
    assert dest == tmp_path / "vendored_test" / "sync_rt"
    assert (tmp_path / "vendored_test" / "__init__.py").is_file()
    for name in ("module.py", "types.py", "descriptor.py", "synchronizer.py", "__init__.py"):
        assert (dest / name).is_file()
    text = (dest / "__init__.py").read_text(encoding="utf-8")
    assert "FunctionWithAio" not in text
    assert "Module" in text
    assert "get_synchronizer" not in text
    assert "_preregister_synchronizer" not in text


def test_cli_vendor_invocation(tmp_path: Path) -> None:
    ok = subprocess.run(
        [
            sys.executable,
            "-m",
            "synchronicity2.codegen",
            "vendor",
            "cli_vendor_test.sync_rt",
            "-o",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert ok.returncode == 0, ok.stderr
    assert (tmp_path / "cli_vendor_test" / "sync_rt" / "types.py").is_file()

    bad = subprocess.run(
        [
            sys.executable,
            "-m",
            "synchronicity2.codegen",
            "vendor",
            "not-a-package!",
            "-o",
            str(tmp_path / "o2"),
        ],
        capture_output=True,
        text=True,
    )
    assert bad.returncode != 0


def test_cli_wrappers_reports_unsupported_default_expr(tmp_path: Path) -> None:
    module_file = tmp_path / "bad_defaults_impl.py"
    module_file.write_text(
        """
import time

from synchronicity2 import Module

wrapper_module = Module("bad_defaults")

@wrapper_module.wrap_function()
async def bad_default(value: float = time.time()) -> float:
    return value
""".strip()
        + "\n"
    )

    project_root = Path(__file__).parent.parent.parent
    env = codegen_env(str(tmp_path), str(project_root))

    result = subprocess.run(
        [
            *codegen_command(),
            "wrappers",
            "-m",
            "bad_defaults_impl",
            "--synchronizer-module",
            "bad_defaults_synchronizer",
            "--stdout",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(project_root),
    )

    assert result.returncode != 0
    assert "Error:" in result.stderr
    assert "parameter 'value'" in result.stderr
    assert "unsupported default expression" in result.stderr


def test_cli_wrappers_rejects_implementation_importing_wrapper_module(tmp_path: Path) -> None:
    (tmp_path / "wrapper_import_cycle.py").write_text("SENTINEL = True\n")
    (tmp_path / "wrapper_import_cycle_impl.py").write_text(
        """
import wrapper_import_cycle

from synchronicity2 import Module

wrapper_module = Module("wrapper_import_cycle")

@wrapper_module.wrap_function()
async def value() -> int:
    return 1
""".strip()
        + "\n"
    )

    result = _run_codegen_wrappers("wrapper_import_cycle_impl", tmp_path)

    assert result.returncode != 0
    assert "Implementation imports loaded generated wrapper module(s) during codegen" in result.stderr
    assert "wrapper_import_cycle" in result.stderr


def test_cli_wrappers_rejects_annotation_fallback_importing_wrapper_module(tmp_path: Path) -> None:
    (tmp_path / "annotation_wrapper_cycle.py").write_text(
        """
class PublicType:
    pass
""".strip()
        + "\n"
    )
    (tmp_path / "annotation_wrapper_cycle_impl.py").write_text(
        """
from synchronicity2 import Module

wrapper_module = Module("annotation_wrapper_cycle")

@wrapper_module.wrap_function()
async def value(arg: "annotation_wrapper_cycle.PublicType | None" = None) -> int:
    return 1
""".strip()
        + "\n"
    )

    result = _run_codegen_wrappers("annotation_wrapper_cycle_impl", tmp_path)

    assert result.returncode != 0
    assert "Implementation annotations may not import generated wrapper module" in result.stderr
    assert "annotation_wrapper_cycle" in result.stderr


def test_cli_wrappers_warns_for_inherited_wrapper_annotation_and_treats_it_as_identity(tmp_path: Path) -> None:
    (tmp_path / "inherited_wrapper_impl.py").write_text(
        """
from synchronicity2 import Module

wrapper_module = Module("inherited_wrapper")

@wrapper_module.wrap_class()
class WrappedBase:
    pass

class UnwrappedChild(WrappedBase):
    pass

@wrapper_module.wrap_function()
async def passthrough(value: UnwrappedChild | int) -> UnwrappedChild | int:
    return value
""".strip()
        + "\n"
    )

    result = _run_codegen_wrappers("inherited_wrapper_impl", tmp_path)

    assert result.returncode == 0, result.stderr
    assert "subclass inherited_wrapper_impl.UnwrappedChild of wrapped implementation class" in result.stderr
    assert "not directly wrapped" in result.stderr
    assert "typing.Union[inherited_wrapper_impl.UnwrappedChild, int]" in result.stdout
    assert "typing.Union[inherited_wrapper.WrappedBase, int]" not in result.stdout
    assert 'hasattr(_v, "_impl_instance")' not in result.stdout
