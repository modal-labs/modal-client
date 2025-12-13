# Copyright Modal Labs 2022

import ast
import datetime
import importlib
import os
import pkgutil
import re
import subprocess
import sys
from collections.abc import Generator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from tempfile import NamedTemporaryFile

import requests
from invoke import call, task
from packaging.version import Version
from rich.console import Console
from rich.table import Table

# Set working directory to the root of the client repository.
original_cwd = Path.cwd()
project_root = Path(os.path.dirname(__file__))
os.chdir(project_root)


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
    """Compile protocol buffer files for gRPC and Modal-specific wrappers.

    Generates Python stubs for api.proto."""
    # Suppress pkg_resources deprecation warning from grpcio-tools
    # See: https://github.com/grpc/grpc/issues/33570
    protoc_env = {"PYTHONWARNINGS": "ignore:pkg_resources is deprecated"}
    protoc_cmd = f"{sys.executable} -m grpc_tools.protoc"
    client_proto_files = "modal_proto/api.proto"
    task_command_router_proto_file = "modal_proto/task_command_router.proto"
    py_protoc = (
        protoc_cmd + " --python_out=. --grpclib_python_out=." + " --grpc_python_out=. --mypy_out=. --mypy_grpc_out=."
    )
    print(py_protoc)
    # generate grpcio and grpclib proto files:
    ctx.run(f"{py_protoc} -I . {client_proto_files} {task_command_router_proto_file}", env=protoc_env)

    # generate modal-specific wrapper around grpclib api stub using custom plugin:
    grpc_plugin_pyfile = Path("protoc_plugin/plugin.py")

    with python_file_as_executable(grpc_plugin_pyfile) as grpc_plugin_executable:
        ctx.run(
            f"{protoc_cmd} --plugin=protoc-gen-modal-grpclib-python={grpc_plugin_executable}"
            + f" --modal-grpclib-python_out=. -I . {client_proto_files}",
            env=protoc_env,
        )


@task(
    help={
        "fix": "Auto-fix issues if possible",
    },
)
def lint(ctx, fix=False):
    """Run linter on all files."""
    ctx.run(f"ruff check {'--fix' if fix else ''}", pty=True, echo=True)
    ctx.run(f"ruff format {'' if fix else '--diff'}", pty=True, echo=True)


def lint_protos_impl(ctx, proto_fname: str):
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
def lint_protos(ctx):
    """Lint protocol buffer files.

    Ensures imports/enums/messages/services are ordered correctly and RPCs are alphabetized.
    """
    lint_protos_impl(ctx, "modal_proto/api.proto")
    lint_protos_impl(ctx, "modal_proto/task_command_router.proto")


@task
def type_stubs(ctx):
    """Generate type stub files (.pyi) for synchronicity-wrapped Modal modules.

    We only generate type stubs for modules that contain synchronicity wrapped types.
    """
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

    def get_wrapped_types(module_name: str) -> list[str]:
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


@task(type_stubs)
def type_check(ctx):
    """Run static type checking.

    Uses mypy for most files, but since mypy will not check the *implementation* (.py) for files that also have .pyi
    type stubs, we use pyright for checking the implementation of those files.
    """
    mypy_exclude_list = [
        "playground",
        "venv312",
        "venv311",
        "venv310",
        "venv39",
        "venv38",
        "test/cls_test.py",  # blocked by mypy bug: https://github.com/python/mypy/issues/16527
        "test/supports/sibling_hydration_app.py",  # blocked by mypy bug: https://github.com/python/mypy/issues/16527
        "test/supports/type_assertions_negative.py",
    ]
    excludes = " ".join(f"--exclude {path}" for path in mypy_exclude_list)
    ctx.run(f"mypy . {excludes}", pty=True)

    pyright_allowlist = [
        "modal/_functions.py",
        "modal/_runtime/asgi.py",
        "modal/_runtime/user_code_imports.py",
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
        "modal/_runtime/container_io_manager.py",
        "modal/io_streams.py",
        "modal/image.py",
        "modal/file_io.py",
        "modal/cli/import_refs.py",
        "modal/snapshot.py",
        "modal/config.py",
        "modal/object.py",
        "modal/_type_manager.py",
        "modal/container_process.py",
    ]
    ctx.run(f"pyright {' '.join(pyright_allowlist)}", pty=True)


@task(
    help={
        "pytest_args": "Arguments to pass to pytest",
    },
)
def test(ctx, pytest_args="-v"):
    """Run all tests."""
    ctx.run(f"pytest {pytest_args}", pty=sys.platform != "win32")  # win32 doesn't support the 'pty' module


@task(
    help={
        "fix": "Automatically add missing headers",
    },
)
def check_copyright(ctx, fix=False):
    """Verify all Python files have correct copyright headers.

    Excludes generated, vendored, and third-party code."""
    invalid_files = []
    for root, dirs, files in os.walk("."):
        fns = [
            os.path.join(root, fn)
            for fn in files
            if (
                fn.endswith(".py")
                # jupytext notebook formatted .py files can't be detected as notebooks if we put a
                # copyright comment at the top
                and not fn.endswith(".notebook.py")
                # ignore generated protobuf code
                and "/modal_proto" not in root
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

        raise Exception(f"{len(invalid_files)} are missing copyright headers! Please run `inv check-copyright --fix`")


@task(
    pre=[
        call(check_copyright, fix=True),
        call(lint, fix=True),
        lint_protos,
        type_check,
    ]
)
def pre_pr_checks(ctx):
    """Run all pre-PR validation checks.

    Auto-fixes anything that can be auto-fixed."""
    ...


def _check_prod(no_confirm: bool):
    from urllib.parse import urlparse

    from modal import config

    server_url = config.config["server_url"]
    if "localhost" not in urlparse(server_url).netloc and not no_confirm:
        answer = input(f"🚨 Modal server URL is '{server_url}' not localhost. Continue operation? [y/N]: ")
        if answer.upper() not in ["Y", "YES"]:
            exit("Aborting task.")
    return server_url


@task
def publish_base_mounts(ctx, no_confirm: bool = False):
    """Publish the client mount and other mounts."""
    _check_prod(no_confirm)
    for mount in ["modal_client_package", "python_standalone", "modal_client_dependencies"]:
        ctx.run(f"{sys.executable} modal_global_objects/mounts/{mount}.py", pty=True)


@task(
    help={
        "name": "Image name (e.g. 'debian_slim')",
        "builder_version": "Docker builder version",
        "allow_global_deployment": "Required flag to confirm global deployment",
    },
)
def publish_base_images(
    ctx,
    name: str,
    builder_version: str = "2024.10",
    allow_global_deployment: bool = False,
    no_confirm: bool = False,
) -> None:
    """Publish base images. For example, `inv publish-base-images debian_slim`.

    These should be published as global deployments. However, publishing global
    deployments is *risky* because it would affect all workspaces. Pass the
    `--allow-global-deployment` flag to confirm this behavior."""
    if not allow_global_deployment:
        console = Console()
        console.print("This is a dry run. Rerun with `--allow-global-deployment` to publish.", style="yellow")

    _check_prod(no_confirm)
    ctx.run(
        f"python -m modal_global_objects.images.base_images {name}",
        pty=True,
        env={
            "MODAL_IMAGE_ALLOW_GLOBAL_DEPLOYMENT": "1" if allow_global_deployment else "",
            "MODAL_IMAGE_BUILDER_VERSION": builder_version,
        },
    )


version_file_contents_template = '''\
# Copyright Modal Labs 2025
"""Supplies the current version of the modal client library."""

__version__ = "{}"
'''


@task(
    help={
        "force": "Bump even if version file was modified in last commit",
    },
)
def bump_dev_version(ctx, dry_run: bool = False, force: bool = False):
    """Automatically increment the modal version, handling dev releases (but not other pre-releases).

    This only has an effect when the version file was not modified by the most recent git commit
    (unless `force` is True).

    The version will always be in development after this runs. In the context of the modal client
    release process, manually updating the version file to a non-development version will trigger
    a "real" release. Otherwise we'll push the development version to PyPI.
    """
    version_file = "modal_version/__init__.py"
    commit_files = ctx.run("git diff --name-only HEAD~1 HEAD", hide="out").stdout.splitlines()
    if version_file in commit_files and not force:
        print(f"Aborting: {version_file} was modified by the most recent commit")
        return

    from modal_version import __version__

    v = Version(__version__)

    if v.is_prerelease:
        if not v.is_devrelease:
            raise RuntimeError("We only know how to auto-bump dev versions")
        # For dev releases, increment the dev suffix
        next_version = f"{v.major}.{v.minor}.{v.micro}.dev{v.dev + 1}"
    else:
        # If the most recent commit was *not* a dev release, start the next cycle
        next_version = f"{v.major}.{v.minor}.{v.micro + 1}.dev0"

    version_file_contents = version_file_contents_template.format(next_version)
    if dry_run:
        print(f"Would update {version_file} to the following:")
        print(version_file_contents)
        return

    with open(version_file, "w") as f:
        f.write(version_file_contents)


@task
def get_release_tag(ctx):
    """Optionally print a tag name for the current modal client version."""
    from modal_version import __version__

    v = Version(__version__)
    if not v.is_devrelease:
        print(f"v{v}")


@task(
    help={
        "sha": "Commit SHA (defaults to the most recent commit)",
    },
)
def update_changelog(ctx, sha: str = ""):
    """Update CHANGELOG.md from GitHub PR description.

    Parse a commit message for a GitHub PR number. Requires GITHUB_TOKEN environment variable."""
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

    # Parse the PR description to get a changelog update, which is all text between
    # the changelog header and any auto comments appended by Cursor

    changelog_pattern = r"## Changelog\s*(.*?)(?:<!--\s*\w*CURSOR\w*\s*-->|$)"
    m = re.search(changelog_pattern, pr_description, flags=re.DOTALL)
    if m:
        update = m.group(1)
    else:
        print("Aborting: No changelog section in PR description")
        return

    # Remove any HTML comments
    comment_pattern = r"<!--.+?-->"
    update = re.sub(comment_pattern, "", update, flags=re.DOTALL).strip()

    if not update:
        print("Aborting: Empty changelog in PR description")
        return

    # Read the existing changelog and split after the header so we can prepend new content
    with open("CHANGELOG.md") as fid:
        content = fid.read()
    token_pattern = "<!-- NEW CONTENT GENERATED BELOW. PLEASE PRESERVE THIS COMMENT. -->"
    m = re.search(token_pattern, content)
    if m:
        break_idx = m.span()[1]
        header = content[:break_idx]
        previous_changelog = content[break_idx:]
    else:
        print("Aborting: Could not find token in existing changelog to mark insertion spot")
        return

    # Build the new changelog and write it out
    from modal_version import __version__

    date = datetime.datetime.now().strftime("%Y-%m-%d")
    new_section = f"#### {__version__} ({date})\n\n{update}"
    final_content = f"{header}\n\n{new_section}\n{previous_changelog}"
    with open("CHANGELOG.md", "w") as fid:
        fid.write(final_content)


@task
def show_deprecations(ctx):
    """Analyze Modal source code and display all deprecation warnings/errors.

    Shows deprecation date, level, location, function, and message in a formatted table."""

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
                # We may add a flag to make renamed_parameter error instead of warn
                # in which case this would get a little bit more complicated.
                "renamed_parameter": "[yellow]warning[/yellow]",
            }
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in func_name_to_level
                and isinstance(node.args[0], ast.Tuple)
            ):
                depr_date = date(*(getattr(elt, "n") for elt in node.args[0].elts))
                function = (
                    f"{self.current_class}.{self.current_function}" if self.current_class else self.current_function
                )
                if node.func.id == "renamed_parameter":
                    old_name = getattr(node.args[1], "s")
                    new_name = getattr(node.args[2], "s")
                    message = f"Renamed parameter: {old_name} -> {new_name}"
                else:
                    message = node.args[1]
                    # Handle a few different ways that the message can get passed to the deprecation helper
                    # since it's not always a literal string (e.g. it's often a functions .__doc__ attribute)
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
