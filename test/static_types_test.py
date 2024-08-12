import pytest
import subprocess


@pytest.fixture(scope="module")
def generate_type_stubs():
    pass  # subprocess.call(["inv", "type-stubs"])


@pytest.mark.usefixtures("generate_type_stubs")
def test_remote_call_keeps_original_return_value():
    subprocess.check_call(["mypy", "test/supports/type_assertions.py"])


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
