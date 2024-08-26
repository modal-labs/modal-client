# Copyright Modal Labs 2024
import pytest
import subprocess
from test.supports.skip import skip_old_py, skip_windows


@pytest.fixture(scope="module")
def generate_type_stubs():
    subprocess.check_call(["inv", "type-stubs"])


@skip_windows("Type tests fail on windows since they don't exclude non-windows features")
@skip_old_py("can't generate type stubs w/ Concatenate on <3.10", (3, 10))
@pytest.mark.usefixtures("generate_type_stubs")
def test_remote_call_keeps_original_return_value():
    subprocess.check_call(["mypy", "test/supports/type_assertions.py"])


@skip_windows("Type tests fail on windows since they don't exclude non-windows features")
@skip_old_py("can't generate type stubs w/ Concatenate on <3.10", (3, 10))
@pytest.mark.usefixtures("generate_type_stubs")
def test_negative_assertions():
    p = subprocess.Popen(
        ["mypy", "test/supports/type_assertions_negative.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf8",
    )
    stdout, _ = p.communicate()
    assert p.returncode == 1
    print(stdout)
    assert "Found 6 errors in 1 file" in stdout
    assert 'Unexpected keyword argument "b" for "__call__"' in stdout
    assert 'Argument "a" to "__call__" of "__remote_spec" has incompatible type "int"' in stdout
    assert 'Unexpected keyword argument "c" for "local" of "Function"' in stdout
    assert 'Argument "a" to "local" of "Function" has incompatible type "int"' in stdout
    assert 'Unexpected keyword argument "e" for "aio" of "__remote_spec"' in stdout
    assert 'Argument "a" to "aio" of "__remote_spec" has incompatible type "float"' in stdout
