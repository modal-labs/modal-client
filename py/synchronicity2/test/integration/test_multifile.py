# Copyright Modal Labs 2026
"""Integration tests for multifile code generation."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from test.integration.test_utils import GENERATED_DIR, PROJECT_ROOT, SUPPORT_FILES, check_pyright


def test_runtime(generated_wrappers):
    from multifile import a as multifile_a, b as multifile_b

    a = multifile_a.A(value=100)
    assert a.get_value() == 100

    b = multifile_b.B(name="test_b")
    assert b.get_name() == "test_b"

    b_from_a = multifile_a.get_b()
    assert b_from_a.get_name() == "test"

    a_from_b = multifile_b.get_a()
    assert a_from_b.get_value() == 42

    generated_dir = generated_wrappers
    module_a = generated_dir / "multifile/a.py"
    module_b = generated_dir / "multifile/b.py"
    assert module_a.exists()
    assert module_b.exists()
    module_a_src = module_a.read_text()
    module_b_src = module_b.read_text()
    assert "class A:" in module_a_src
    assert "class B:" in module_b_src
    assert "class A" not in module_b_src
    assert "class B" not in module_a_src
    assert "def get_b(" in module_a_src
    assert "def get_a(" in module_b_src


def test_codegen_module_not_imported_at_runtime(generated_wrappers):
    """Importing generated wrappers does not import synchronicity2.codegen."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        check_imports_script = tmppath / "check_imports.py"
        check_imports_script.write_text(
            """
import sys
from multifile.a import A
from multifile.b import B

sync_modules = [m for m in sys.modules.keys() if m.startswith('synchronicity2')]
print("SYNC_MODULES:", ','.join(sorted(sync_modules)))

assert 'synchronicity2.codegen' not in sys.modules, "synchronicity2.codegen should not be imported at runtime"
assert 'synchronicity2.synchronizer' in sys.modules, "synchronicity2.synchronizer should be imported at runtime"

print("IMPORT_CHECK_SUCCESS")
"""
        )

        project_root = PROJECT_ROOT
        support_files = SUPPORT_FILES
        generated_dir = GENERATED_DIR

        import os

        env = os.environ.copy()
        pythonpath_parts = [str(project_root), str(support_files), str(generated_dir)]
        if "PYTHONPATH" in env:
            pythonpath_parts.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

        result_imports = subprocess.run(
            [sys.executable, str(check_imports_script)],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            env=env,
        )

        print(f"Import check output:\n{result_imports.stdout}")
        if result_imports.returncode != 0:
            print(f"Import check stderr:\n{result_imports.stderr}")

        assert result_imports.returncode == 0
        assert "IMPORT_CHECK_SUCCESS" in result_imports.stdout
        assert "synchronicity2.codegen" not in result_imports.stdout


def test_pyright_implementation():
    support_files = Path(__file__).parent.parent / "support_files"
    check_pyright(
        [
            support_files / "multifile_impl/_a.py",
            support_files / "multifile_impl/_b.py",
            support_files / "multifile_impl/__init__.py",
        ]
    )


def test_pyright_wrapper(generated_wrappers):
    module_a = generated_wrappers / "multifile/a.py"
    module_b = generated_wrappers / "multifile/b.py"
    assert module_a.exists()
    assert module_b.exists()
    project_root = generated_wrappers.parent
    support_files = Path(__file__).parent.parent / "support_files"
    pythonpath = os.pathsep.join([str(project_root), str(support_files)])
    check_pyright([module_a, module_b], pythonpath)


def test_pyright_usage():
    from importlib.util import find_spec

    spec = find_spec("multifile_typecheck")
    assert spec and spec.origin
    check_pyright([Path(spec.origin)])
