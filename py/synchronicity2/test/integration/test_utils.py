# Copyright Modal Labs 2026
"""Shared utility functions for integration tests."""

import contextlib
import os
import pytest
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
SUPPORT_FILES = PROJECT_ROOT / "test" / "support_files"
# Bazel runs tests from a read-only tree, so generated code goes to the per-test scratch dir there.
GENERATED_DIR = (
    Path(os.environ["TEST_TMPDIR"]) / "generated" if os.environ.get("TEST_TMPDIR") else PROJECT_ROOT / "generated"
)


def _runfile(rloc: str) -> str:
    """Resolve a Bazel runfiles path, failing if it is missing (i.e. a missing ``data`` dependency)."""
    from python.runfiles import Runfiles  # type: ignore[import-not-found]

    runfiles = Runfiles.Create()
    path = runfiles.Rlocation(rloc) if runfiles else None
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"runfile {rloc!r} missing (got {path!r})")
    return path


def codegen_command() -> list[str]:
    """Command prefix that runs the synchronicity2 codegen CLI.

    Codegen can run on a newer Python than the tests, to check that generated wrappers work on older Pythons than
    codegen itself requires: ``RLOC_SYNCHRONICITY2_CODEGEN`` is a Bazel runfiles path to a codegen executable,
    ``SYNCHRONICITY2_CODEGEN_PYTHON`` an interpreter to run ``-m synchronicity2.codegen`` with.
    """
    if rloc := os.environ.get("RLOC_SYNCHRONICITY2_CODEGEN"):
        return [_runfile(rloc)]
    return [os.environ.get("SYNCHRONICITY2_CODEGEN_PYTHON", sys.executable), "-m", "synchronicity2.codegen"]


def codegen_env(*pythonpath: str) -> dict[str, str]:
    """Environment for running ``codegen_command()`` with ``pythonpath`` importable."""
    env = dict(os.environ)
    # The Bazel codegen executable brings its own Python and packages, which the test interpreter's paths would shadow.
    inherited = [] if os.environ.get("RLOC_SYNCHRONICITY2_CODEGEN") else [env.get("PYTHONPATH")]
    env["PYTHONPATH"] = os.pathsep.join(p for p in [*pythonpath, *inherited] if p)
    return env


def _pyright(args: list[str], extra_pythonpath: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run pyright with generated wrappers and support files importable."""
    env = dict(os.environ)
    parts = [extra_pythonpath, str(GENERATED_DIR), str(SUPPORT_FILES), env.get("PYTHONPATH")]
    env["PYTHONPATH"] = os.pathsep.join(p for p in parts if p)
    # Under Bazel, node comes from the toolchain in the runfiles rather than being downloaded by pyright-python.
    if rloc := os.environ.get("RLOC_SYNCHRONICITY2_NODE"):
        node = Path(_runfile(rloc.removeprefix("external/")))
        env["PATH"] = os.pathsep.join([str(node.parent), env.get("PATH", "")])
    return subprocess.run(
        [sys.executable, "-m", "pyright", "--pythonpath", sys.executable, *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(PROJECT_ROOT),
    )


def check_pyright(module_paths: list[Path], extra_pythonpath: str | None = None) -> str:
    """Run pyright on generated code to check for type errors.

    Args:
        module_paths: List of module file paths to check
        extra_pythonpath: Additional path to add to PYTHONPATH

    Returns:
        Pyright output if successful

    Raises:
        pytest.fail if pyright fails
    """
    result = _pyright([str(p) for p in module_paths], extra_pythonpath)

    if result.returncode != 0:
        print("  ✗ Pyright validation failed!")
        print(f"    Output: {result.stdout}")
        if result.stderr:
            print(f"    Stderr: {result.stderr}")
        pytest.fail("Pyright validation failed")

    print("  ✓ Pyright validation passed")
    return result.stdout


def check_pyright_with_xfail(script_module: str):
    """Pyright checks a file with expected failures
    The expected failures should be documented using comments of the form
    # xfail: <error pattern>
    Where <error pattern> should match the subsequent line's pyright error loosely
    """
    from importlib.util import find_spec

    spec = find_spec(script_module)
    assert spec
    assert spec.has_location
    script_path = spec.origin
    assert script_path
    source = Path(script_path).read_text()
    lines = source.split("\n")
    xfail_starts = [(i, line.lstrip()) for i, line in enumerate(lines) if line.lstrip().startswith("# xfail:")]

    xfail_searches = []
    for line_i, line in xfail_starts:
        current_line = line
        endline_i = line_i

        while current_line.endswith("\\"):  # continuation character
            endline_i += 1
            current_line = lines[endline_i]
            assert current_line.lstrip().startswith("#")  # next line must be a comment

        # skip any lines with comments or empty lines before the offending python lines
        offending_line_i = endline_i
        while current_line.startswith("#") or len(current_line.strip()) == 0:
            offending_line_i += 1
            current_line = lines[offending_line_i]

        chunks = [
            lines[i].lstrip().removeprefix("#").lstrip().removeprefix("xfail:").lstrip().rstrip().removesuffix("\\")
            for i in range(line_i, endline_i + 1)
        ]
        failure_string = " ".join(chunks)

        xfail_searches.append((offending_line_i, failure_string))

    if not xfail_searches:
        raise Exception("The supplied type assertion file has no xfail assertions")

    result = _pyright([script_path])
    assert result.returncode != 0
    output = result.stdout
    expected_num_errors = len(xfail_searches)
    if not re.search(rf"{expected_num_errors} errors?,", output):
        # TODO: be helpful and show the only the errors we didn't expect here:
        pytest.fail(f"expected {expected_num_errors} errors, got:\n{output}", pytrace=False)

    for line_i, error_string in xfail_searches:
        line_no = line_i + 1
        # pyright may print a differently spelled path to the same file (e.g. Windows short names), so match on the name
        mo = re.search(rf"[\\/]{re.escape(Path(script_path).name)}:{line_no}:\d+ - error: ([^\n]*)", output)
        if not mo:
            pytest.fail(f"expected type failure on line {line_no}, but did not find one")

        actual_error = mo.group(1)
        escaped_pattern = re.escape(error_string)
        reduced_spaces, _ = re.subn(r"(\\\ )+", lambda _: r"\s+", escaped_pattern)
        error_did_match = re.search(reduced_spaces, actual_error, flags=re.S)
        if not error_did_match:
            pytest.fail(f"expected error {error_string!r}, found {actual_error!r}")


def close_synchronizers(*module_names: str) -> None:
    """Close synchronizers exported by generated instance modules."""
    import importlib

    for module_name in module_names:
        importlib.import_module(module_name).synchronizer._close_loop()


@contextlib.contextmanager
def closing_synchronizers(*module_names: str):
    """Wrap a runtime test body and close the synchronizers it starts."""
    try:
        yield
    finally:
        close_synchronizers(*module_names)
