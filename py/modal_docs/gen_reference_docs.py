# Copyright Modal Labs 2023
import importlib
import inspect
import json
import os
import sys
import warnings
from typing import NamedTuple

from synchronicity.synchronizer import FunctionWithAio

from .mdmd.mdmd import (
    Category,
    class_str,
    default_filter,
    function_str,
    module_items,
    module_str,
    object_is_private,
    package_filter,
)


class DocItem(NamedTuple):
    label: str
    category: Category
    document: str
    in_sidebar: bool = True


def validate_doc_item(docitem: DocItem) -> DocItem:
    # Check that unwanted strings aren't leaking into our docs.
    bad_strings = [
        # Presence of a to-do inside a `DocItem` usually indicates it's been
        # placed inside a function signature definition or right underneath it, before the body.
        # Fix by moving the to-do into the body or above the signature.
        "TODO:"
    ]
    for line in docitem.document.splitlines():
        for bad_str in bad_strings:
            if bad_str in line:
                msg = f"Found unwanted string '{bad_str}' in content for item '{docitem.label}'. Problem line: {line}"
                raise ValueError(msg)
    return docitem


# Curated landing page for the Python SDK reference. Each section has a title and
# either a table of entries or a set of subsections (each with its own table).
# Entries are (name, path, description): `path` is appended to `_INTRO_BASE` to
# form the link, and `description` may contain inline Markdown.
_INTRO_BASE = "/docs/sdk/py/latest"

_INTRO_FRONTMATTER = {
    "description": (
        "Complete API reference for the Modal Python SDK. "
        "Documentation for App, Function, Image, Sandbox, Volume, and other Modal primitives."
    ),
}

_INTRO_PREAMBLE = """\
# Python SDK Reference

This is the API reference for the [`modal`](https://pypi.org/project/modal/)
Python SDK, which allows you to programmatically interact with Modal.
"""

_INTRO_SECTIONS: list[dict] = [
    {
        "title": "Application construction",
        "entries": [
            ("App", "modal.App", "The main unit of deployment for code on Modal"),
            ("App.function", "modal.App#function", "Decorator for registering a function with an App"),
            ("App.cls", "modal.App#cls", "Decorator for registering a class with an App"),
            ("App.server", "modal.App#server", "Decorator for registering a server with an App"),
        ],
    },
    {
        "title": "Serverless execution",
        "entries": [
            ("Function", "modal.Function", "A serverless function backed by an autoscaling container pool"),
            ("Cls", "modal.Cls", "A serverless class supporting parametrization and lifecycle hooks"),
            ("Server", "modal.Server", "A serverless HTTP application with low-latency request routing"),
        ],
    },
    {
        "title": "Extended Function configuration",
        "subsections": [
            {
                "title": "Class parametrization",
                "entries": [
                    ("parameter", "modal.parameter", "Used to define class parameters, akin to a Dataclass field"),
                ],
            },
            {
                "title": "Lifecycle hooks",
                "entries": [
                    ("enter", "modal.enter", "Decorator for a method that will be executed during container startup"),
                    ("exit", "modal.exit", "Decorator for a method that will be executed during container shutdown"),
                    ("method", "modal.method", "Decorator for exposing a method as an invokable function"),
                ],
            },
            {
                "title": "Web integrations",
                "entries": [
                    (
                        "fastapi_endpoint",
                        "modal.fastapi_endpoint",
                        "Decorator for exposing a simple FastAPI-based endpoint",
                    ),
                    ("asgi_app", "modal.asgi_app", "Decorator for functions that construct an ASGI web application"),
                    ("wsgi_app", "modal.wsgi_app", "Decorator for functions that construct a WSGI web application"),
                    ("web_server", "modal.web_server", "Decorator for functions that construct an HTTP web server"),
                ],
            },
            {
                "title": "Function semantics",
                "entries": [
                    (
                        "batched",
                        "modal.batched",
                        "Decorator that enables [dynamic input batching](/docs/guide/dynamic-batching)",
                    ),
                    (
                        "concurrent",
                        "modal.concurrent",
                        "Decorator that enables [input concurrency](/docs/guide/concurrent-inputs)",
                    ),
                ],
            },
            {
                "title": "Scheduling",
                "entries": [
                    ("Cron", "modal.Cron", "A schedule that runs based on cron syntax"),
                    ("Period", "modal.Period", "A schedule that runs at a fixed interval"),
                ],
            },
            {
                "title": "Exception handling",
                "entries": [
                    ("Retries", "modal.Retries", "Function retry policy for input failures"),
                ],
            },
        ],
    },
    {
        "title": "Sandboxed execution",
        "entries": [
            ("Sandbox", "modal.Sandbox", "An interface for restricted code execution"),
            (
                "ContainerProcess",
                "modal.container_process#modalcontainer_processcontainerprocess",
                "An object representing a sandboxed process",
            ),
            ("FileIO", "modal.file_io#modalfile_iofileio", "A handle for a file in the Sandbox filesystem"),
        ],
    },
    {
        "title": "Container configuration",
        "entries": [
            ("Image", "modal.Image", "An API for specifying container images"),
            ("Secret", "modal.Secret", "A pointer to secrets that will be exposed as environment variables"),
        ],
    },
    {
        "title": "Data primitives",
        "subsections": [
            {
                "title": "Persistent storage",
                "entries": [
                    ("Volume", "modal.Volume", "Distributed storage supporting highly performant parallel reads"),
                    (
                        "CloudBucketMount",
                        "modal.CloudBucketMount",
                        "Storage backed by a third-party cloud bucket (S3, etc.)",
                    ),
                ],
            },
            {
                "title": "In-memory storage",
                "entries": [
                    ("Dict", "modal.Dict", "A distributed key-value store"),
                    ("Queue", "modal.Queue", "A distributed FIFO queue"),
                ],
            },
        ],
    },
    {
        "title": "Account configuration",
        "entries": [
            ("Workspace", "modal.Workspace", "Workspace-level configuration and observability"),
            ("Environment", "modal.Environment", "Manage workspace subdivisions"),
        ],
    },
    {
        "title": "Networking",
        "entries": [
            ("Proxy", "modal.Proxy", "An object that provides a static outbound IP address for containers"),
            ("forward", "modal.forward", "A context manager for publicly exposing a port from a container"),
        ],
    },
]


def _intro_entry_table(entries) -> str:
    rows = ["|  |  |", "| --- | --- |"]
    for name, path, description in entries:
        rows.append(f"| [`{name}`]({_INTRO_BASE}/{path}) | {description} |")
    return "\n".join(rows)


def get_intro_docs() -> str:
    """Render the curated Python SDK reference landing page (`intro.md`)."""
    frontmatter = "\n".join(f"{key}: {value}" for key, value in _INTRO_FRONTMATTER.items())
    sections = [f"---\n{frontmatter}\n---\n\n{_INTRO_PREAMBLE}"]
    for section in _INTRO_SECTIONS:
        if "entries" in section:
            sections.append(f"## {section['title']}\n\n" + _intro_entry_table(section["entries"]) + "\n")
        else:
            block = f"## {section['title']}\n"
            for sub in section["subsections"]:
                block += f"\n### {sub['title']}\n\n" + _intro_entry_table(sub["entries"]) + "\n"
            sections.append(block)
    return "\n".join(sections)


def run(output_dir: str | None = None):
    """Generate Modal docs."""
    import modal

    ordered_doc_items: list[DocItem] = []
    documented_items = set()

    def filter_already_documented(module, name):
        item = getattr(module, name)
        try:
            if item in documented_items:
                return False
        except TypeError:  # unhashable stuff
            print(f"Warning: could not document item {name}: {item}:")
            return False
        documented_items.add(item)
        return True

    def module_doc_filter(module, name):
        return default_filter(module, name)

    def top_level_filter(module, name):
        item = getattr(module, name)
        if object_is_private(name, item) or inspect.ismodule(item):
            return False
        return package_filter("modal") and filter_already_documented(module, name)

    base_title_level = "#"
    # Standalone module reference pages, sorted into the sidebar with top-level `modal.X` entries.
    module_docs = [
        ("modal.billing", "modal.billing"),
        ("modal.call_graph", "modal.call_graph"),
        ("modal.config", "modal.config"),
        ("modal.container_process", "modal.container_process"),
        ("modal.exception", "modal.exception"),
        ("modal.file_io", "modal.file_io"),
        ("modal.io_streams", "modal.io_streams"),
        ("modal.types", "modal.types"),
    ]
    # These aren't defined in `modal`, but should still be documented as top-level entries.
    forced_members: set[str] = set()
    # These are excluded from the sidebar, typically to 'soft release' some documentation.
    sidebar_excluded: set[str] = {"modal.NetworkFileSystem"}

    for title, modulepath in module_docs:
        module = importlib.import_module(modulepath)
        document = module_str(modulepath, module, title_level=base_title_level, filter_items=module_doc_filter)
        if document:
            document = f"<script>\n    import Parameter from '$lib/ui/docs/Parameter.svelte';\n</script>\n\n{document}"
            ordered_doc_items.append(
                validate_doc_item(
                    DocItem(
                        label=title,
                        category=Category.MODULE,
                        document=document,
                        in_sidebar=title not in sidebar_excluded,
                    )
                )
            )

    def f(module, member_name):
        return top_level_filter(module, member_name) or (member_name in forced_members)

    # now add all remaining top level modal.X entries
    for qual_name, item_name, item in module_items(modal, filter_items=f):
        if object_is_private(item_name, item):
            continue  # skip stuff that's part of explicit `handle_objects` above

        title = f"modal.{item_name}"
        if inspect.isclass(item):
            content = f"{base_title_level} {qual_name}\n\n" + class_str(item_name, item, base_title_level)
            category = Category.CLASS
        elif inspect.isroutine(item) or isinstance(item, FunctionWithAio):
            content = f"{base_title_level} {qual_name}\n\n" + function_str(item_name, item)
            category = Category.FUNCTION
        elif inspect.ismodule(item):
            continue  # skipping imported modules
        else:
            warnings.warn(f"Not sure how to document: {item_name} ({item})")
            continue
        content = f"<script>\n    import Parameter from '$lib/ui/docs/Parameter.svelte';\n</script>\n\n{content}"
        ordered_doc_items.append(
            validate_doc_item(
                DocItem(
                    label=title,
                    category=category,
                    document=content,
                    in_sidebar=title not in sidebar_excluded,
                )
            )
        )
    ordered_doc_items.sort()

    # TODO: add some way of documenting our .aio sub-methods

    make_markdown_docs(
        ordered_doc_items,
        output_dir,
    )


def make_markdown_docs(items: list[DocItem], output_dir: str = None):
    def _write_file(rel_path: str, data: str):
        if output_dir is None:
            print(f"<<< {rel_path}")
            print(data)
            print(f">>> {rel_path}")
            return

        filename = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as fp:
            fp.write(data)

    sidebar_items = []
    for item in items:
        if item.in_sidebar:
            sidebar_items.append(
                {
                    "label": item.label,
                    "category": item.category.value,
                }
            )
        _write_file(f"{item.label}.md", item.document)

    sidebar_data = {"items": sidebar_items}
    _write_file("sidebar.json", json.dumps(sidebar_data))

    # The curated reference landing page, generated as `intro.md` alongside the
    # per-symbol pages (mirroring how the CLI reference index is generated).
    _write_file("intro.md", get_intro_docs())


if __name__ == "__main__":
    # running this module outputs docs to stdout for inspection, useful for debugging
    run(None if len(sys.argv) <= 1 else sys.argv[1])
