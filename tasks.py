# Copyright Modal Labs 2022
# Copyright (c) Modal Labs 2022

import ast
import datetime
import importlib
import os
import pkgutil
import re
import subprocess
import sys
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Generator, List, Optional

import requests
from invoke import task
from rich.console import Console
from rich.table import Table

year = datetime.date.today().year
copyright_header_start = "# Copyright Modal Labs"
copyright_header_full = f"{copyright_header_start} {year}"


@contextmanager
def python_file_as_executable(path: Path) -> Generator[Path, None, None]:
    if sys.platform == "win32":
        # windows can't just run shebang:ed python files, so we create a .bat file that calls it
        src = f"""@echo off
{sys.executable} {path}
"""
        with NamedTemporaryFile(mode="w", suffix=".bat", encoding="ascii", delete=False) as f:
            f.write(src)

        try:
            yield Path(f.name)
        finally:
            Path(f.name).unlink()
    else:
        yield path


@task
def protoc(ctx):
    protoc_cmd = f"{sys.executable} -m grpc_tools.protoc"
    input_files = "modal_proto/api.proto modal_proto/options.proto"
    py_protoc = (
        protoc_cmd + " --python_out=. --grpclib_python_out=." + " --grpc_python_out=. --mypy_out=. --mypy_grpc_out=."
    )
    print(py_protoc)
    # generate grpcio and grpclib proto files:
    ctx.run(f"{py_protoc} -I . {input_files}")

    # generate modal-specific wrapper around grpclib api stub using custom plugin:
    grpc_plugin_pyfile = Path(__file__).parent / "protoc_plugin" / "plugin.py"

    with python_file_as_executable(grpc_plugin_pyfile) as grpc_plugin_executable:
        ctx.run(
            f"{protoc_cmd} --plugin=protoc-gen-modal-grpclib-python={grpc_plugin_executable}"
            + f" --modal-grpclib-python_out=. -I . {input_files}"
        )


@task
def lint(ctx, fix=False):
    ctx.run(f"ruff . {'--fix' if fix else ''}", pty=True)


@task
def lint_protos(ctx):
    proto_fname = "modal_proto/api.proto"
    with open(proto_fname) as f:
        proto_text = f.read()

    sections = ["import", "enum", "message", "service"]
    section_regex = "|".join(sections)
    matches = re.findall(rf"^((?:{section_regex})\s+(?:\w+))", proto_text, flags=re.MULTILINE)
    entities = [tuple(e.split()) for e in matches]

    console = Console()

    def get_first_lineno_with_prefix(text: str, prefix: str) -> int:
        lines = text.split("\n")
        for lineno, line in enumerate(lines):
            if re.match(rf"^{prefix}", line):
                return lineno
        raise RuntimeError(f"Failed to find line starting with `{prefix}` (this shouldn't happen)")

    section_order = {key: i for i, key in enumerate(sections)}
    for (a_type, a_name), (b_type, b_name) in zip(entities[:-1], entities[1:]):
        if (section_order[a_type] > section_order[b_type]) or (a_type == b_type and a_name > b_name):
            # This is a simplistic and sort of hacky of way of identifying the "out of order" entity,
            # as the latter one may be the one that is misplaced. Doesn't seem worth the effort though.
            lineno = get_first_lineno_with_prefix(proto_text, f"{a_type} {a_name}")
            console.print(f"[bold red]Proto lint error:[/bold red] {proto_fname}:{lineno}")
            console.print(f"\nThe {a_name} {a_type} proto is out of order relative to the {b_name} {b_type}.")
            console.print(
                "\nProtos should be organized into the following sections:", *sections, sep="\n - ", style="dim"
            )
            console.print("\nWithin sections, protos should be lexicographically sorted by name.", style="dim")
            sys.exit(1)

    service_chunks = re.findall(r"service \w+ {(.+)}", proto_text, flags=re.DOTALL)
    for service_text in service_chunks:
        rpcs = re.findall(r"^\s*rpc\s+(\w+)", service_text, flags=re.MULTILINE)
        for rpc_a, rpc_b in zip(rpcs[:-1], rpcs[1:]):
            if rpc_a > rpc_b:
                lineno = get_first_lineno_with_prefix(proto_text, rf"\s*rpc\s+{rpc_a}")
                console.print(f"[bold red]Proto lint error:[/bold red] {proto_fname}:{lineno}")
                console.print(f"\nThe {rpc_a} rpc proto is out of order relative to the {rpc_b} rpc.")
                console.print("\nRPC definitions should be ordered within each service proto.", style="dim")
                sys.exit(1)


@task
def type_check(ctx):
    type_stubs(ctx)
    # mypy will not check the *implementation* (.py) for files that also have .pyi type stubs
    mypy_exclude_list = [
        "playground",
        "venv312",
        "venv311",
        "venv310",
        "venv39",
        "venv38",
        "test/cls_test.py",  # blocked by mypy bug: https://github.com/python/mypy/issues/16527
        "test/supports/type_assertions_negative.py",
    ]
    excludes = " ".join(f"--exclude {path}" for path in mypy_exclude_list)
    ctx.run(f"mypy . {excludes}", pty=True)

    # use pyright for checking implementation of those files
    pyright_allowlist = [
        "modal/functions.py",
        "modal/runtime/_asgi.py",
        "modal/_utils/__init__.py",
        "modal/_utils/async_utils.py",
        "modal/_utils/grpc_testing.py",
        "modal/_utils/hash_utils.py",
        "modal/_utils/http_utils.py",
        "modal/_utils/name_utils.py",
        "modal/_utils/logger.py",
        "modal/_utils/mount_utils.py",
        "modal/_utils/package_utils.py",
        "modal/_utils/rand_pb_testing.py",
        "modal/_utils/shell_utils.py",
        "test/cls_test.py",  # see mypy bug above - but this works with pyright, so we run that instead
        "modal/runtime/_container_io_manager.py",
        "modal/io_streams.py",
    ]
    ctx.run(f"pyright {' '.join(pyright_allowlist)}", pty=True)


@task
def check_copyright(ctx, fix=False):
    invalid_files = []
    d = str(Path(__file__).parent)
    for root, dirs, files in os.walk(d):
        fns = [
            os.path.join(root, fn)
            for fn in files
            if (
                fn.endswith(".py")
                # jupytext notebook formatted .py files can't be detected as notebooks if we put a
                # copyright comment at the top
                and not fn.endswith(".notebook.py")
                # vendored code has a different copyright
                and "_vendor" not in root
                and "protoc_plugin" not in root
                # third-party code (i.e., in a local venv) has a different copyright
                and "/site-packages/" not in root
                and "/build/" not in root
                and "/.venv/" not in root
                and not re.search(r"/venv[0-9]*/", root)
            )
        ]
        for fn in fns:
            first_line = open(fn).readline()
            if not first_line.startswith(copyright_header_start):
                if fix:
                    print(f"Fixing {fn}...")
                    content = copyright_header_full + "\n" + open(fn).read()
                    with open(fn, "w") as g:
                        g.write(content)
                else:
                    invalid_files.append(fn)

    if invalid_files:
        for fn in invalid_files:
            print("Missing copyright:", fn)

        raise Exception(
            f"{len(invalid_files)} are missing copyright headers!" " Please run `inv check-copyright --fix`"
        )


@task
def publish_base_mounts(ctx, no_confirm=False):
    from urllib.parse import urlparse

    from modal import config

    server_url = config.config["server_url"]
    if "localhost" not in urlparse(server_url).netloc and not no_confirm:
        answer = input(f"Modal server URL is '{server_url}' not localhost. Continue operation? [y/N]: ")
        if answer.upper() not in ["Y", "YES"]:
            exit("Aborting task.")
    for mount in ["modal_client_package", "python_standalone"]:
        ctx.run(f"{sys.executable} {Path(__file__).parent}/modal_global_objects/mounts/{mount}.py", pty=True)


@task
def update_build_number(ctx, new_build_number: Optional[int] = None):
    from modal_version import build_number as current_build_number

    new_build_number = int(new_build_number) if new_build_number else current_build_number + 1
    assert new_build_number > current_build_number

    # Add the current Git SHA to the file, so concurrent publish actions of the
    # client package result in merge conflicts.
    git_sha = ctx.run("git rev-parse --short=7 HEAD", hide="out").stdout.rstrip()

    with open("modal_version/_version_generated.py", "w") as f:
        f.write(
            f"""\
{copyright_header_full}

# Note: Reset this value to -1 whenever you make a minor `0.X` release of the client.
build_number = {new_build_number}  # git: {git_sha}
"""
        )


@task
def create_alias_package(ctx):
    from modal_version import __version__

    os.makedirs("alias-package", exist_ok=True)
    with open("alias-package/setup.py", "w") as f:
        f.write(
            f"""\
{copyright_header_full}
from setuptools import setup
setup(version="{__version__}")
"""
        )
    with open("alias-package/setup.cfg", "w") as f:
        f.write(
            f"""\
[metadata]
name = modal-client
author = Modal Labs
author_email = support@modal.com
description = Legacy name for the Modal client
long_description = This is a legacy compatibility package that just requires the `modal` client library.
            In versions before 0.51, the official name of the client library was called `modal-client`.
            We have renamed it to `modal`, but this library is kept updated for compatibility.
long_description_content_type = text/markdown
project_urls =
    Homepage = https://modal.com

[options]
install_requires =
    modal=={__version__}
"""
        )
    with open("alias-package/pyproject.toml", "w") as f:
        f.write(
            """\
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"
"""
        )


@task
def type_stubs(ctx):
    # We only generate type stubs for modules that contain synchronicity wrapped types
    from synchronicity.synchronizer import SYNCHRONIZER_ATTR

    stubs_to_remove = []
    for root, _, files in os.walk("modal"):
        for file in files:
            if file.endswith(".pyi"):
                stubs_to_remove.append(os.path.abspath(os.path.join(root, file)))
    for path in sorted(stubs_to_remove):
        os.remove(path)
        print(f"Removed {path}")

    def find_modal_modules(root: str = "modal"):
        modules = []
        path = importlib.import_module(root).__path__
        for _, name, is_pkg in pkgutil.iter_modules(path):
            full_name = f"{root}.{name}"
            if is_pkg:
                modules.extend(find_modal_modules(full_name))
            else:
                modules.append(full_name)
        return modules

    def get_wrapped_types(module_name: str) -> List[str]:
        module = importlib.import_module(module_name)
        return [
            name
            for name, obj in vars(module).items()
            if not module_name.startswith("modal.cli.")  # TODO we don't handle typer-wrapped functions well
            and hasattr(obj, "__module__")
            and obj.__module__ == module_name
            and not name.startswith("_")  # Avoid deprecation of _App.__getattr__
            and hasattr(obj, SYNCHRONIZER_ATTR)
        ]

    modules = [m for m in find_modal_modules() if len(get_wrapped_types(m))]
    subprocess.check_call(["python", "-m", "synchronicity.type_stubs", *modules])
    ctx.run("ruff format modal/ --exclude=*.py --no-respect-gitignore", pty=True)


@task
def update_changelog(ctx, sha: str = ""):
    # Parse a commit message for a GitHub PR number, defaulting to most recent commit
    res = ctx.run(f"git log --pretty=format:%s -n 1 {sha}", hide="stdout", warn=True)
    if res.exited:
        print("Failed to extract changelog update!")
        print("Last 5 commits:")
        res = ctx.run("git log --pretty=oneline -n 5")
        return
    m = re.search(r"\(#(\d+)\)$", res.stdout)
    if m:
        pull_number = m.group(1)
    else:
        print("Aborting: No PR number in commit message")
        return

    # Get the corresponding PR description via the GitHub API
    url = f"https://api.github.com/repos/modal-labs/modal-client/pulls/{pull_number}"
    headers = {"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}", "Accept": "application/vnd.github.v3+json"}
    response = requests.get(url, headers=headers).json()
    pr_description = response.get("body")
    if pr_description is None:
        print("Aborting: No PR description in response from GitHub API")
        return

    # Parse the PR description to get a changelog update
    comment_pattern = r"<!--.+?-->"
    pr_description = re.sub(comment_pattern, "", pr_description, flags=re.DOTALL)

    changelog_pattern = r"## Changelog\s*(.+)$"
    m = re.search(changelog_pattern, pr_description, flags=re.DOTALL)
    if m:
        update = m.group(1).strip()
    else:
        print("Aborting: No changelog section in PR description")
        return
    if not update:
        print("Aborting: Empty changelog in PR description")
        return

    # Read the existing changelog and split after the header so we can prepend new content
    with open("CHANGELOG.md", "r") as fid:
        content = fid.read()
    token_pattern = "<!-- NEW CONTENT GENERATED BELOW. PLEASE PRESERVE THIS COMMENT. -->"
    m = re.search(token_pattern, content)
    if m:
        break_idx = m.span()[1]
        header = content[:break_idx]
        previous_changelog = content[break_idx:]
    else:
        print("Aborting: Could not find token in existing changelog to mark insertion spot")

    # Build the new changelog and write it out
    from modal_version import __version__

    date = datetime.datetime.now().strftime("%Y-%m-%d")
    new_section = f"### {__version__} ({date})\n\n{update}"
    final_content = f"{header}\n\n{new_section}\n\n{previous_changelog}"
    with open("CHANGELOG.md", "w") as fid:
        fid.write(final_content)


@task
def show_deprecations(ctx):
    def get_modal_source_files() -> list[str]:
        source_files: list[str] = []
        for root, _, files in os.walk("modal"):
            for file in files:
                if file.endswith(".py"):
                    source_files.append(os.path.join(root, file))
        return source_files

    class FunctionCallVisitor(ast.NodeVisitor):
        def __init__(self, fname):
            self.fname = fname
            self.deprecations = []
            self.assignments = {}
            self.current_class = None
            self.current_function = None

        def visit_ClassDef(self, node):
            self.current_class = node.name
            self.generic_visit(node)
            self.current_class = None

        def visit_FunctionDef(self, node):
            self.current_function = node.name
            self.assignments["__doc__"] = ast.get_docstring(node)
            self.generic_visit(node)
            self.current_function = None
            self.assignments.pop("__doc__", None)

        def visit_Assign(self, node):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.assignments[target.id] = node.value
            self.generic_visit(node)

        def visit_Attribute(self, node):
            self.assignments[node.attr] = node.value
            self.generic_visit(node)

        def visit_Call(self, node):
            func_name_to_level = {
                "deprecation_warning": "[yellow]warning[/yellow]",
                "deprecation_error": "[red]error[/red]",
            }
            if isinstance(node.func, ast.Name) and node.func.id in func_name_to_level:
                depr_date = date(*(elt.n for elt in node.args[0].elts))
                function = (
                    f"{self.current_class}.{self.current_function}" if self.current_class else self.current_function
                )
                message = node.args[1]
                if isinstance(message, ast.Name):
                    message = self.assignments.get(message.id, "")
                if isinstance(message, ast.Attribute):
                    message = self.assignments.get(message.attr, "")
                if isinstance(message, ast.Constant):
                    message = message.s
                elif isinstance(message, ast.JoinedStr):
                    message = "".join(v.s for v in message.values if isinstance(v, ast.Constant))
                else:
                    message = str(message)
                message = message.replace("\n", " ")
                if len(message) > (max_length := 80):
                    message = message[:max_length] + "..."
                level = func_name_to_level[node.func.id]
                self.deprecations.append((str(depr_date), level, f"{self.fname}:{node.lineno}", function, message))

    files = get_modal_source_files()
    deprecations = []
    for fname in files:
        with open(fname) as f:
            tree = ast.parse(f.read())
        visitor = FunctionCallVisitor(fname)
        visitor.visit(tree)
        deprecations.extend(visitor.deprecations)

    console = Console()
    table = Table("Date", "Level", "Location", "Function", "Message")
    for row in sorted(deprecations, key=lambda r: r[0]):
        table.add_row(*row)
    console.print(table)
