# Copyright Modal Labs 2023
import importlib
import os
from enum import IntEnum
from types import ModuleType

from modal_docs.mdmd import mdmd
from modal_docs.mdmd.signatures import parse_params_from_signature, strip_signature


def test_simple_function():
    def foo():
        pass

    assert (
        mdmd.function_str("bar", foo)
        == """```python
bar()
```\n\n"""
    )


def test_simple_async_function():
    async def foo():
        pass

    assert (
        mdmd.function_str("bar", foo)
        == """```python
bar()
```\n\n"""
    )


def test_async_gen_function():
    async def foo():
        yield

    assert (
        mdmd.function_str("bar", foo)
        == """```python
bar()
```\n\n"""
    )


def test_complex_function_signature():
    def foo(a: str, *args, **kwargs):
        pass

    assert (
        mdmd.function_str("foo", foo)
        == """```python
foo(a, *args, **kwargs)
```\n\n"""
    )


def test_complex_function_signature_with_line_hidden():
    def foo(
        a: str,
        *args,  # mdmd:line-hidden
        **kwargs,
    ):
        pass

    assert (
        mdmd.function_str("foo", foo)
        == """```python
foo(a, **kwargs)
```\n\n"""
    )


def test_function_has_docstring():
    def foo():
        """short description

        longer description"""

    assert (
        mdmd.function_str("foo", foo)
        == """```python
foo()
```
short description

longer description
"""
    )


def test_simple_class_with_docstring():
    class Foo:
        """The all important Foo"""

        def bar(self, baz: str):
            """Bars the foo with the baz"""

    # Exact snapshot includes leading newline from `parts` and method signatures from `strip_signature`.
    expected = (
        "\n```python\nclass Foo(object)\n```\n\nThe all important Foo\n\n\n### bar\n\n"
        "```python\nbar(self, baz)\n```\nBars the foo with the baz\n"
    )
    assert mdmd.class_str("Foo", Foo) == expected


def test_class_docstring_examples_become_usage():
    """Class docstrings use the same Example(s) extraction as :func:`parse_docstring`."""

    class Foo:
        """Widget class.

        Example:
            ```python
            w = Foo()
            ```
        """

    out = mdmd.class_str("Foo", Foo)
    assert "**Usage**" in out
    assert "w = Foo()" in out


def test_simple_class_with_docstring_with_line_hidden():
    class Foo:
        """The all important Foo mdmd:line-hidden"""

        def bar(self, baz: str):
            """Bars the foo with the baz

            This won't be included mdmd:line-hidden
            """

    assert (
        mdmd.class_str("Foo", Foo)
        == """
```python
class Foo(object)
```


### bar

```python
bar(self, baz)
```
Bars the foo with the baz
"""
    )


def test_enum():
    class Eee(IntEnum):
        FOO = 1
        BAR = 2
        XYZ = 3

    expected = """
```python
class bar(enum.IntEnum)
```

An enumeration.

The possible values are:

* `FOO`
* `BAR`
* `XYZ`
"""

    assert mdmd.class_str("bar", Eee) == expected


def test_class_with_classmethod():
    class Foo:
        @classmethod
        def create_foo(cls, some_arg):
            pass

    assert (
        mdmd.class_str("Foo", Foo)
        == """
```python
class Foo(object)
```


### create_foo

```python
create_foo(cls, some_arg)
```

"""
    )


def test_class_with_baseclass_includes_base_methods():
    class Foo:
        def foo(self):
            pass

    class Bar(Foo):
        def bar(self):
            pass

    out = mdmd.class_str("Bar", Bar)
    assert "foo(self)" in out


def test_module(monkeypatch):
    test_data_dir = os.path.join(os.path.dirname(__file__), "mdmd_data")
    monkeypatch.chdir(test_data_dir)
    monkeypatch.syspath_prepend(test_data_dir)
    test_module = importlib.import_module("foo")
    expected_output = open("./foo-expected.md").read()
    assert mdmd.module_str("foo", test_module) == expected_output


def test_docstring_format_reindents_code():
    assert (
        mdmd.clean_docstring(
            """```python
        foo
            bar
        ```"""
        )
        == """```python
foo
    bar
```
"""
    )


def test_exported_module_filter_includes_all_reexports():
    module = ModuleType("example_module", "Module docs\nmdmd:exported")

    class Reexported:
        pass

    Reexported.__module__ = "elsewhere"
    setattr(module, "Reexported", Reexported)
    setattr(module, "Hidden", Reexported)
    setattr(module, "__all__", ["Reexported"])

    assert mdmd.default_filter(module, "Reexported")
    assert not mdmd.default_filter(module, "Hidden")


def test_docstring_format_hides_exported_directive():
    assert mdmd.clean_docstring("Visible\nmdmd:exported\nStill visible") == "Visible\nStill visible\n"


def test_synchronicity_async_and_blocking_interfaces():
    from synchronicity import Synchronizer

    class Foo:
        """docky mcdocface"""

        async def foo(self):
            pass

        def bar(self):
            pass

    s = Synchronizer()
    BlockingFoo = s.create_blocking(Foo, "BlockingFoo")

    assert (
        mdmd.class_str("BlockingFoo", BlockingFoo)
        == """
```python
class BlockingFoo(object)
```

docky mcdocface


### foo

```python
foo(self)
```


### bar

```python
bar(self)
```

"""
    )


def test_synchronicity_constructors():
    from synchronicity import Synchronizer

    class Foo:
        """docky mcdocface"""

        def __init__(self):
            """constructy mcconstructorface"""

    s = Synchronizer()
    BlockingFoo = s.create_blocking(Foo, "BlockingFoo")

    assert (
        mdmd.class_str("BlockingFoo", BlockingFoo)
        == """
```python
class BlockingFoo(object)
```

docky mcdocface

```python
__init__(self)
```
constructy mcconstructorface
"""
    )


def test_get_all_signature_comments():
    def foo(
        # prefix comment
        one,  # one comment
        two,  # two comment
        # postfix comment
    ) -> str:  # return value comment
        pass

    assert (
        mdmd.function_str("foo", foo)
        == """```python
foo(one, two)
```

"""
    )


def test_classproperty_instance_manager():
    """Ensure mdmd documents a classproperty that returns a manager *instance* (not a class)."""
    from synchronicity import Synchronizer, classproperty

    class _MyManager:
        """Manager docs"""

        async def create(self, name: str) -> None:
            """Create a thing."""

        async def delete(self, name: str) -> None:
            """Delete a thing."""

    s = Synchronizer()

    class _MyResource:
        """Resource docs"""

        @classproperty
        @classmethod
        def objects(cls) -> _MyManager:
            return _MyManager()

    MyResource = s.create_blocking(_MyResource, "MyResource")

    result = mdmd.class_str("MyResource", MyResource)
    assert "## objects" in result
    assert "objects: _MyManager" in result
    assert "class objects" not in result
    assert "### objects.create" in result
    assert "### objects.delete" in result
    assert "Manager docs" in result


def test_property_namespace_decl_override():
    from synchronicity import Synchronizer

    class _FilesystemNamespace:
        """mdmd:namespace
        Namespace docs"""

        def read(self, path: str) -> bytes:
            """Read a file."""

    s = Synchronizer()

    class _Resource:
        @property
        def filesystem(self) -> _FilesystemNamespace:
            return _FilesystemNamespace()

    Resource = s.create_blocking(_Resource, "Resource")

    result = mdmd.class_str("Resource", Resource)
    assert "## filesystem" in result
    assert "filesystem: FilesystemNamespace" in result
    assert "class filesystem" not in result
    assert "### filesystem.read" in result
    assert "Namespace docs" in result


def test_get_decorators():
    BLA = 1

    def my_deco(arg):
        def wrapper(f):
            return f

        return wrapper

    @my_deco(BLA)
    def foo():
        pass

    assert (
        mdmd.function_str("foo", foo)
        == """```python
foo()
```

"""
    )


def test_parse_params_from_signature_multiline_comments_and_hidden_lines():
    signature = """
async def create(
    *args: str,
    app: Optional["modal.app._App"] = None,
    # comment to ignore
    name: Optional[str] = None,
    image: Optional[_Image] = None,  # mdmd:line-hidden
) -> "_Sandbox":
"""

    assert [(p.name, p.type, p.default) for p in parse_params_from_signature(signature)] == [
        ("*args", "str", None),
        ("app", "Optional[modal.app._App]", "None"),
        ("name", "Optional[str]", "None"),
    ]


def test_parse_params_from_signature_single_line():
    signature = "def f(a: int, b: str = 'x', **kwargs: object):"
    assert [(p.name, p.type, p.default) for p in parse_params_from_signature(signature)] == [
        ("a", "int", None),
        ("b", "str", "'x'"),
        ("**kwargs", "object", None),
    ]


def test_parse_params_from_signature_without_trailing_colon():
    signature = "def __init__(self, x: int) -> None"
    assert [(p.name, p.type, p.default) for p in parse_params_from_signature(signature)] == [
        ("self", "", None),
        ("x", "int", None),
    ]


def test_parse_docstring_sections():
    docstring = """
    Create a new Sandbox to run untrusted, arbitrary code.

    The Sandbox's corresponding container will be created asynchronously.

    Args:
        *args: Set the CMD of the Sandbox, overriding any CMD of the container image.
        app (modal.App | None): Associate the sandbox with an app. Required unless creating from a container.
        network_file_systems:

    Returns:
        A `Sandbox` object representing the created sandbox which can be used to interact with the sandbox.

    Raises:
        AlreadyExistsError: If a sandbox with the same name already exists.

    Example:
        ```python
        app = modal.App.lookup("sandbox-hello-world", create_if_missing=True)
        sandbox = modal.Sandbox.create("echo", "hello world", app=app)
        print(sandbox.stdout.read())
        sandbox.wait()
        ```
    """
    parsed = mdmd.parse_docstring(
        "create",
        "def create(*args, app: modal.App | None = None, network_file_systems = None):",
        docstring,
    )
    assert parsed.name == "create"
    assert len(parsed.params) == 3
    assert parsed.params[0].name == "*args"
    assert parsed.params[0].type == ""
    assert parsed.params[0].description == "Set the CMD of the Sandbox, overriding any CMD of the container image."
    assert parsed.params[1].name == "app"
    assert parsed.params[1].type == "modal.App | None"
    assert (
        parsed.params[1].description == "Associate the sandbox with an app. Required unless creating from a container."
    )
    assert parsed.params[2].name == "network_file_systems"
    assert parsed.params[2].type == ""
    assert parsed.params[2].description is None
    assert len(parsed.raises) == 1
    assert parsed.raises[0].type == "AlreadyExistsError"
    assert parsed.raises[0].description == "If a sandbox with the same name already exists."
    assert parsed.returns and not parsed.returns.startswith("Returns:")
    assert "A `Sandbox` object representing the created sandbox" in parsed.returns
    assert parsed.examples and not parsed.examples.startswith("Example:")
    assert parsed.examples.startswith("```python")
    assert 'sandbox = modal.Sandbox.create("echo", "hello world", app=app)' in parsed.examples


def test_parse_docstring_examples_blank_line_inside_fenced_code():
    """Blank lines inside ``` blocks must not end the Examples section early."""
    docstring = """
    Do the thing.

    Examples:
        ```py notest
        x = 1

        y = 2
        ```
    """
    parsed = mdmd.parse_docstring("f", "def f():", docstring)
    assert parsed.examples is not None
    assert "x = 1" in parsed.examples
    assert "y = 2" in parsed.examples
    assert parsed.examples.count("```") == 2


def test_parse_docstring_examples_blank_line_between_blocks_and_prose():
    """Blank lines after a fenced block must not end Examples; prose + more fences follow."""
    docstring = """
    Run the app.

    Args:
        client: The client.

    Returns:
        Nothing.

    Examples:
        ```python notest
        with app.run():
            f()
        ```

        To enable logs, use `modal.enable_output()`:

        ```python notest
        with modal.enable_output():
            with app.run():
                f()
        ```

        More prose.

        ```shell
        python app.py
        ```
    """
    parsed = mdmd.parse_docstring("run", "def run(self, client):", docstring)
    assert parsed.examples is not None
    assert "with app.run():" in parsed.examples
    assert "modal.enable_output()" in parsed.examples
    assert "More prose." in parsed.examples
    assert "python app.py" in parsed.examples
    assert parsed.examples.count("```") == 6


def test_parse_docstring_yields_section():
    docstring = """
    Stream records.

    Yields:
        bytes: Next chunk from the stream.
    """
    parsed = mdmd.parse_docstring("stream", "def stream():", docstring)
    assert parsed.returns and not parsed.returns.startswith("Yields:")
    assert parsed.returns == "bytes: Next chunk from the stream."


def test_parse_docstring_see_also_section():
    docstring = """
    Do the thing.

    Args:
        x: The input.

    Returns:
        The result.

    Raises:
        ValueError: If x is bad.

    See Also:
        - [`other_func`](/docs/reference/other_func)
        - [`Foo`](/docs/reference/Foo)
    """
    parsed = mdmd.parse_docstring("do_thing", "def do_thing(x):", docstring)
    assert parsed.see_also is not None
    assert not parsed.see_also.startswith("See Also:")
    assert "- [`other_func`](/docs/reference/other_func)" in parsed.see_also
    assert "- [`Foo`](/docs/reference/Foo)" in parsed.see_also
    # The See Also content must not leak into the description.
    assert "other_func" not in (parsed.description or "")

    rendered = mdmd._markdown_body_from_parsed_doc(parsed)
    assert "**See Also**" in rendered
    # See Also must come after Parameters, Returns, and Raises.
    assert (
        rendered.index("**Parameters**")
        < rendered.index("**Returns**")
        < rendered.index("**Raises**")
        < rendered.index("**See Also**")
    )


def test_function_str_escapes_braces_in_parameter_attributes_for_svelte():
    """Braces in defaults (e.g. ``{}``) must not be parsed as Svelte ``{expr}``."""

    def foo(a={}):
        """Doc.

        Args:
            a: The thing.
        """

    out = mdmd.function_str("foo", foo)
    assert 'defaultValue="&#123;&#125;"' in out
    assert 'defaultValue="{}"' not in out


def test_strip_signature_wraps_long_call_form_to_80_columns():
    """Long stripped call forms should break on whitespace so each line fits 80 columns."""
    params = ", ".join(f"a{n}" for n in range(50))
    signature = f"def many_params({params}):"
    out = strip_signature(signature)
    assert "\n" in out
    for line in out.splitlines():
        assert len(line) <= 80, repr(line)


def test_strip_signature_preserves_bare_star_separator():
    """Bare * separator for keyword-only args must appear in the output."""
    signature = "def update_autoscaler(self, *, min_containers=None, max_containers=None):"
    out = strip_signature(signature)
    assert "*, " in out or out.endswith(", *)")
    assert "update_autoscaler(self, *, min_containers=None, max_containers=None)" in out


def test_parse_docstring_multiline_arg_and_raise_descriptions():
    docstring = """
    Do thing.

    Args:
        config (dict[str, str]):
            Configuration for the operation.
            Used by multiple subsystems.
        verbose: Enable extra logging.

    Raises:
        ValueError:
            If the configuration is invalid.
            If required keys are missing.
    """
    parsed = mdmd.parse_docstring("do_thing", "def do_thing(config: dict[str, str], verbose):", docstring)
    assert len(parsed.params) == 2
    assert parsed.params[0].name == "config"
    assert parsed.params[0].type == "dict[str, str]"
    assert parsed.params[0].description == "Configuration for the operation. Used by multiple subsystems."
    assert parsed.params[1].name == "verbose"
    assert parsed.params[1].description == "Enable extra logging."
    assert len(parsed.raises) == 1
    assert parsed.raises[0].type == "ValueError"
    assert parsed.raises[0].description == "If the configuration is invalid. If required keys are missing."
