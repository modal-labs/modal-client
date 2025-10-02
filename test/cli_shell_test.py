# Copyright Modal Labs 2025


def test_shell_kwargs_compatible_with_sandbox_create():
    """Test that _ShellKwargs fields are compatible with _Sandbox._create.

    This is a contract test that ensures the fields in _ShellKwargs dataclass
    are a subset of the parameters accepted by _Sandbox._create, which is the internal
    method used by modal shell to create sandboxes.
    """
    import dataclasses
    import inspect

    from modal.cli.shell import _ShellKwargs
    from modal.sandbox import _Sandbox

    # Get parameter names from _Sandbox._create (internal method used by modal shell)
    sig = inspect.signature(_Sandbox._create)
    sandbox_params = set(sig.parameters.keys())

    # Get all field names from _ShellKwargs dataclass
    shell_kwargs_fields = {field.name for field in dataclasses.fields(_ShellKwargs)}

    # Assert that all _ShellKwargs fields are valid _Sandbox._create parameters
    # Note: We use _Sandbox._create (not .create) because it exposes 'mounts' which
    # is specifically used by modal shell (see comment in sandbox.py:431-435)
    assert shell_kwargs_fields.issubset(sandbox_params), (
        f"_ShellKwargs fields {shell_kwargs_fields - sandbox_params} are not valid "
        f"_Sandbox._create parameters. This indicates an incompatibility between "
        f"_ShellKwargs and _Sandbox._create."
    )
