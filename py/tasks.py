# Copyright Modal Labs 2022

import ast
import datetime
import importlib
import os
import pkgutil
import re
import subprocess
import sys
from contextlib import suppress
from datetime import date
from pathlib import Path

from invoke import call, task
from invoke.exceptions import Exit, UnexpectedExit

# Set working directory to the root of the client repository.
original_cwd = Path.cwd()
project_root = Path(os.path.dirname(__file__))
os.chdir(project_root)


year = datetime.date.today().year
copyright_header_start = "# Copyright Modal Labs"
copyright_header_full = f"{copyright_header_start} {year}"


# Generated code from protobuf 5.26 onward calls this with the protoc version that produced it, and refuses to load on
# an older runtime. Older generated code carries no version.
GENCODE_STAMP_RE = re.compile(r"ValidateProtobufRuntimeVersion\(\s*\S+,\s*(\d+),\s*(\d+),\s*(\d+)")

# `inv update-proto-pins` writes requirements.protos.txt from PROTOS_REQUIREMENTS and MINIMUM_PROTOBUF, plus the
# grpcio-tools pins it computes. The file is used for:
# 1. Container entrypoint testing: For each Python version, runs
#    `uv run --with-requirements requirements.protos.txt scripts/compile_protos.py` to generate the protos,
#    then imports the container entrypoint with an image builder version's dependencies.
# 2. Single-Python CI jobs: Installs the client with `--overrides requirements.protos.txt`, for type checks,
#    docstring tests, JS and Go SDK tests, and mount and image publishing.
# 3. Publishing to PyPI: Generates the protos with the minimum supported Python and its pins.
# 4. `inv protoc --isolated`: Generates the protos locally with the current Python's pins.
# The Bazel unit tests do not use it. Their dependencies come from client/py/uv.lock.
PROTOS_REQUIREMENTS = [
    "mypy-protobuf~=3.3.0  # TODO: can't use mypy-protobuf>=3.4 because of protobuf==3.20 support",
    "grpclib==0.4.7;python_version<'3.14'",
    "grpclib==0.4.9;python_version>='3.14'",
    "setuptools<82",
]

# The oldest protobuf runtime each supported Python version is tested against. These are support decisions, so
# `inv update-proto-pins` never changes them. For a supported Python version missing here, it suggests the oldest
# protobuf that works and stops. Each Python's grpcio-tools pin is the newest release whose generated code imports
# under this protobuf. From protobuf 5.26 onward, generated code requires a runtime no older than the protoc that
# produced it, which can be newer than grpcio-tools' declared protobuf minimum.
# Python 3.10 is the only version that tests protobuf 3.x, pinned to 3.20.3, the last 3.x release.
# TODO: when Python 3.10 is dropped, remove its entry, raise the protobuf floor in pyproject.toml to the oldest
# remaining pin, and upgrade mypy-protobuf.
MINIMUM_PROTOBUF = {
    "3.10": "3.20.3",
    "3.11": "4.21.3",
    "3.12": "4.21.6",
    "3.13": "5.27.2",
    "3.14": "6.31.1",
    "3.15": "7.35.1",
}


def _supported_pythons() -> list[str]:
    """The Python versions in pyproject.toml's requires-python, oldest first."""
    requires_python = re.search(r'requires-python = ">=3\.(\d+),<3\.(\d+)"', Path("pyproject.toml").read_text())
    if requires_python is None:
        raise Exit("Could not parse requires-python from pyproject.toml")
    return [f"3.{minor}" for minor in range(int(requires_python[1]), int(requires_python[2]))]


@task(
    help={
        "skip_mypy": "skip generating MyPy files",
        "isolated": "generate with the same environment as testing",
    },
)
def protoc(ctx, skip_mypy: bool = False, isolated: bool = False):
    """Compile protocol buffer files for gRPC and Modal-specific wrappers.

    Generates Python stubs for api.proto."""

    if isolated:
        print("Generating protos in isolated environment from requirements.protos.txt")
        major = sys.version_info.major
        minor = sys.version_info.minor
        skip_mypy_flag = " --skip-mypy" if skip_mypy else ""
        ctx.run(
            f"uv run --python {major}.{minor} --with-requirements requirements.protos.txt "
            f"scripts/compile_protos.py{skip_mypy_flag}",
            echo=True,
        )
    else:
        print("Generating protos in the current environment")
        cmd = f"{sys.executable} scripts/compile_protos.py"
        if skip_mypy:
            cmd = f"{cmd} --skip-mypy"
        ctx.run(cmd, echo=True)


@task
def update_proto_pins(ctx):
    """Write requirements.protos.txt from PROTOS_REQUIREMENTS and MINIMUM_PROTOBUF in tasks.py.

    For each supported Python, pins protobuf to its MINIMUM_PROTOBUF entry and grpcio-tools to the newest release
    whose generated code imports under it. For a supported Python missing from MINIMUM_PROTOBUF, suggests the oldest
    protobuf that works and stops without writing.

    Pins are resolved for Linux x86-64, where CI uses them, and the task runs there too, because it generates and
    imports the protos with each candidate.
    """
    import platform
    import tempfile

    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.utils import canonicalize_name
    from packaging.version import Version

    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise Exit(
            "Run `inv update-proto-pins` on Linux x86-64. Some pinned grpcio-tools releases have no wheels elsewhere,"
            " for example on Apple Silicon."
        )

    pyproject = Path("pyproject.toml").read_text()
    pythons = _supported_pythons()

    requirements_path = Path("requirements.protos.txt")
    # Requirement strings without their comments, for resolution.
    other_requirements = [line.split("#", 1)[0].strip() for line in PROTOS_REQUIREMENTS]
    other_names = {canonicalize_name(Requirement(requirement).name) for requirement in other_requirements}

    client_protobuf = re.search(r'"protobuf([^"]*)"', pyproject)
    if client_protobuf is None:
        raise Exit("Could not find the protobuf dependency in pyproject.toml")
    client_protobuf_spec = SpecifierSet(client_protobuf[1])

    # Releases are discovered by resolving with uv, so only releases the configured package index serves are
    # considered. `--only-binary grpcio-tools` limits candidates to releases with a wheel for the target Python.
    def resolve(py: str, requirements: list[str], *args: str) -> dict[str, str] | None:
        """Resolve `requirements` with PROTOS_REQUIREMENTS for Python `py`.

        Returns pinned versions by package name, or None if unsatisfiable.
        """
        result = subprocess.run(
            [
                "uv", "pip", "compile", "-", "--python-version", py, "--python-platform", "x86_64-manylinux_2_28",
                "--only-binary", "grpcio-tools", "--no-header", "--no-annotate", *args,
            ],
            input="\n".join([*requirements, *other_requirements]), capture_output=True, text=True,
        )  # fmt: skip
        if result.returncode:
            return None
        pins = (line.split("==", 1) for line in result.stdout.splitlines() if "==" in line)
        return {canonicalize_name(name.strip()): version.strip() for name, version in pins}

    env = {**os.environ, "PYTHONWARNINGS": "ignore:pkg_resources is deprecated"}

    def generate(py: str, tools: str, out: Path) -> str | None:
        """Generate the protos into `out`. Returns an error message on failure."""
        result = subprocess.run(
            [
                "uv", "run", "--no-project", "--python", py,
                "--with", f"grpcio-tools=={tools}", "--with", "setuptools<82",
                "python", "-m", "grpc_tools.protoc", f"--python_out={out}", "-I", "..",
                "modal_proto/api.proto", "modal_proto/task_command_router.proto",
            ],
            capture_output=True, text=True, env=env,
        )  # fmt: skip
        if result.returncode:
            return (result.stderr.strip().splitlines() or ["unknown error"])[-1].strip()
        return None

    def gencode_stamp(out: Path) -> Version | None:
        match = GENCODE_STAMP_RE.search((out / "modal_proto" / "api_pb2.py").read_text())
        return Version(".".join(match.groups())) if match else None

    def imports(py: str, protobuf: Version, out: Path) -> bool:
        result = subprocess.run(
            [
                "uv", "run", "--no-project", "--python", py, "--with", f"protobuf=={protobuf}",
                "python", "-c", "import modal_proto.api_pb2, modal_proto.task_command_router_pb2",
            ],
            cwd=out, capture_output=True, text=True, env=env,
        )  # fmt: skip
        return result.returncode == 0

    def oldest_protobuf(py: str) -> Version:
        """The oldest protobuf that loads code generated by the first grpcio-tools release with a wheel for `py`."""
        # Resolving the direct requirements to their lowest versions picks the first grpcio-tools release with a
        # wheel for `py`, and the lowest protobuf release that it, pyproject.toml and the other requirements accept.
        protobuf_req = f"protobuf{client_protobuf_spec}"
        # Hold the other requirements at the versions CI installs, the newest that fit, since the lowest-direct
        # resolution would otherwise also take their oldest releases, which may not even build.
        newest = resolve(py, ["grpcio-tools", protobuf_req])
        if newest is None:
            raise Exit(f"py{py}: no grpcio-tools release with a wheel for this Python fits the other requirements")
        held = sorted({f"{name}=={newest[name]}" for name in other_names if name in newest})
        lowest = resolve(py, ["grpcio-tools", protobuf_req, *held], "--resolution", "lowest-direct")
        if lowest is None:
            raise Exit(f"py{py}: no grpcio-tools release with a wheel for this Python fits the other requirements")
        first = lowest["grpcio-tools"]
        with tempfile.TemporaryDirectory() as tmp:
            if error := generate(py, first, Path(tmp)):
                raise Exit(f"py{py}: grpcio-tools {first} failed to generate protos: {error}")
            stamp = gencode_stamp(Path(tmp))
        if stamp and stamp > Version(lowest["protobuf"]):
            # The generated code needs a newer runtime than the resolved one, so take the oldest release from there.
            lowest = resolve(
                py,
                [f"grpcio-tools=={first}", protobuf_req, f"protobuf>={stamp}", *held],
                "--resolution",
                "lowest-direct",
            )
            if lowest is None:
                raise Exit(f"py{py}: no protobuf release from {stamp} fits the other requirements")
        return Version(lowest["protobuf"])

    unsupported = sorted(set(MINIMUM_PROTOBUF) - set(pythons))
    if unsupported:
        raise Exit(f"Remove MINIMUM_PROTOBUF entries for Python versions outside requires-python: {unsupported}")
    missing = [py for py in pythons if py not in MINIMUM_PROTOBUF]
    if missing:
        suggestions = "\n".join(f'    "{py}": "{oldest_protobuf(py)}",' for py in missing)
        raise Exit(f"Add the oldest protobuf to test against for each new Python to MINIMUM_PROTOBUF:\n{suggestions}")

    results = []
    for py in pythons:
        protobuf = Version(MINIMUM_PROTOBUF[py])
        if protobuf not in client_protobuf_spec:
            raise Exit(
                f"py{py}: MINIMUM_PROTOBUF's {protobuf} is outside pyproject.toml's range {client_protobuf_spec}"
            )
        if resolve(py, [f"protobuf=={protobuf}"]) is None:
            raise Exit(f"py{py}: MINIMUM_PROTOBUF's {protobuf} conflicts with PROTOS_REQUIREMENTS")
        print(f"py{py}: protobuf {protobuf}")

        # Step down from the newest grpcio-tools release that accepts `protobuf` until one generates code that
        # imports under it.
        found = None
        below = None
        while True:
            tools_req = f"grpcio-tools<{below}" if below else "grpcio-tools"
            resolved = resolve(py, [tools_req, f"protobuf=={protobuf}"])
            if resolved is None:
                break
            tools = resolved["grpcio-tools"]
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                if error := generate(py, tools, out):
                    print(f"  grpcio-tools {tools}: failed to generate protos: {error}")
                    below = tools
                    continue
                stamp = gencode_stamp(out)
                ok = imports(py, protobuf, out)
            print(f"  grpcio-tools {tools}: gencode {stamp or 'unstamped'} -> {'OK' if ok else 'does not import'}")
            if ok:
                found = (tools, stamp)
                break
            below = tools

        if found is None:
            raise Exit(f"py{py}: no grpcio-tools release generates code that imports under protobuf {protobuf}")
        results.append((py, protobuf, *found))

    print()
    print(f"{'Python':<8}{'protobuf':<12}{'grpcio-tools':<14}gencode")
    for py, protobuf, tools, stamp in results:
        print(f"{py:<8}{str(protobuf):<12}{tools:<14}{stamp or 'unstamped'}")

    pins = {
        "grpcio-tools": [tools for _, _, tools, _ in results],
        "protobuf": [str(protobuf) for _, protobuf, _, _ in results],
    }

    def marker(first: int, last: int) -> str:
        """Environment marker for the Python versions `pythons[first:last + 1]`.

        The newest Python version still gets an upper bound, so a Python version outside requires-python matches no
        pin rather than inheriting the newest one.
        """
        clauses = []
        if first > 0:
            clauses.append(f"python_version>='{pythons[first]}'")
        next_minor = int(pythons[last].split(".")[1]) + 1
        clauses.append(f"python_version<'3.{next_minor}'")
        return f";{' and '.join(clauses)}"

    def pin_lines(name: str) -> list[str]:
        by_python = pins[name]
        lines, first = [], 0
        for i in range(1, len(by_python) + 1):
            if i == len(by_python) or by_python[i] != by_python[first]:
                lines.append(f"{name}=={by_python[first]}{marker(first, i - 1)}")
                first = i
        return lines

    lines = [
        "# Generated by `inv update-proto-pins` from client/py/tasks.py. Do not edit by hand.",
        "# Pins grpcio-tools and protobuf per Python for generating the protos; see tasks.py for what uses this file.",
        "",
        *PROTOS_REQUIREMENTS,
        "",
        *pin_lines("grpcio-tools"),
        *pin_lines("protobuf"),
    ]
    requirements_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {requirements_path}")


@task
def check_minimum_installed_protobuf(ctx):
    """Check that the installed protobuf is the MINIMUM_PROTOBUF entry for the oldest supported Python."""
    from importlib.metadata import PackageNotFoundError, version

    py = _supported_pythons()[0]
    pin = MINIMUM_PROTOBUF.get(py)
    if pin is None:
        raise Exit(f"MINIMUM_PROTOBUF in tasks.py has no entry for Python {py}")
    try:
        installed = version("protobuf")
    except PackageNotFoundError:
        installed = "not installed"
    if installed != pin:
        raise Exit(f"protobuf is {installed}, but MINIMUM_PROTOBUF in tasks.py has {pin} for Python {py}")
    print(f"protobuf {installed} is the minimum for Python {py}")


@task(
    help={
        "fix": "Auto-fix issues if possible",
    },
)
def lint(ctx, fix=False):
    """Run linter on all files."""
    ctx.run(f"ruff check {'--fix' if fix else ''}", pty=True, echo=True)
    ctx.run(f"ruff format {'' if fix else '--diff'}", pty=True, echo=True)


@task
def type_stubs(ctx, verbose: bool = False):
    """Generate type stub files (.pyi) for synchronicity-wrapped Modal modules.

    We only generate type stubs for modules that contain synchronicity wrapped types.
    """
    from synchronicity.synchronizer import SYNCHRONIZER_ATTR

    print("Generating synchronicity type stubs...")

    stubs_to_remove = []
    for root, _, files in os.walk("modal"):
        for file in files:
            if file.endswith(".pyi"):
                stubs_to_remove.append(os.path.abspath(os.path.join(root, file)))
    for path in sorted(stubs_to_remove):
        os.remove(path)
        if verbose:
            print(f"Removed {path}")

    def find_modal_modules(root: str = "modal") -> list[str]:
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
            if not module_name.startswith("modal.cli.")  # CLI handlers are not called into directly
            and hasattr(obj, "__module__")
            and obj.__module__ == module_name
            and not name.startswith("_")  # Avoid deprecation of _App.__getattr__
            and hasattr(obj, SYNCHRONIZER_ATTR)
        ]

    modules = [m for m in find_modal_modules() if len(get_wrapped_types(m))]
    subprocess.check_call(
        ["python", "-m", "synchronicity.type_stubs", *modules],
        stdout=None if verbose else subprocess.DEVNULL,
        stderr=None if verbose else subprocess.DEVNULL,
    )
    ctx.run("ruff format modal/ --exclude=*.py --no-respect-gitignore", pty=True)


@task()
def type_check(ctx, stubs: bool = True, mypy: bool = True, pyright: bool = True):
    """Run static type checking.

    Uses mypy for most files, but since mypy will not check the *implementation* (.py) for files that also have .pyi
    type stubs, we use pyright for checking the implementation of those files.
    """
    if stubs:
        type_stubs(ctx)

    mypy_exclude_list = [
        "playground",
        "test/cls_test.py",  # blocked by mypy bug: https://github.com/python/mypy/issues/16527
        "test/supports/sibling_hydration_app.py",  # blocked by mypy bug: https://github.com/python/mypy/issues/16527
        "test/supports/type_assertions_negative.py",
        "modal_proto",
        "synchronicity2",  # standalone package with its own type-checking setup
    ]
    excludes = " ".join(f"--exclude {path}" for path in mypy_exclude_list)
    if mypy:
        print("Checking types with mypy...")
        ctx.run(f"mypy . {excludes}", pty=True)

    # Pyright checks are more limited; see pyproject.toml for configuration
    if pyright:
        print("Checking types with pyright...")
        ctx.run("pyright", pty=True)


@task(
    help={
        "pytest_args": "Arguments to pass to pytest",
    },
)
def test(ctx, pytest_args="-v"):
    """Run all tests."""
    ctx.run(
        f"pytest {pytest_args}",
        pty=sys.platform != "win32",
        in_stream=sys.stdin if "--pdb" in pytest_args else False,
    )  # win32 doesn't support the 'pty' module


@task()
def doctest(ctx):
    """Exercise tests in docstring examples."""
    ctx.run("pytest -v --markdown-docs -m markdown-docs modal", pty=sys.platform != "win32")


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
                # transient codegen output from the synchronicity2 test suite
                and not root.startswith("./synchronicity2/generated")
                # third-party code (i.e., in a local venv) has a different copyright
                and "/site-packages/" not in root
                and "/build/" not in root
                and "/.venv" not in root
                and not re.search(r"/venv[0-9]*/", root)
            )
        ]
        for fn in fns:
            head = open(fn).readlines()[:2]
            if not any(line.startswith(copyright_header_start) for line in head):
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


def _run_mount_publish_script(ctx, mount: str, args: str = ""):
    cmd = f"{sys.executable} {project_root}/modal_global_objects/mounts/{mount}.py{args}"
    return ctx.run(
        cmd,
        asynchronous=True,
        env={"PYTHONUNBUFFERED": "1"},
        out_stream=sys.stdout,
        err_stream=sys.stderr,
    )


def _python_standalone_args():
    from modal._image import LATEST_IMAGE_BUILDER_VERSION

    return f" --builder-version {LATEST_IMAGE_BUILDER_VERSION} --parallel"


@task(help={"python_standalone": "Also publish Python standalone mounts."})
def publish_base_mounts(ctx, no_confirm: bool = False, python_standalone: bool = True):
    """Publish the client mount and other mounts."""
    _check_prod(no_confirm)
    mount_scripts = [("modal_client_package", "")]
    if python_standalone:
        mount_scripts.append(("python_standalone", _python_standalone_args()))
    mount_scripts.append(("modal_client_dependencies", ""))
    promises = [(mount, _run_mount_publish_script(ctx, mount, args)) for mount, args in mount_scripts]

    failed = []
    for mount, promise in promises:
        try:
            promise.join()
        except UnexpectedExit:
            failed.append(mount)

    if failed:
        raise Exit(f"Failed to publish: {', '.join(failed)}", code=1)


@task
def publish_python_standalone_mounts(ctx, no_confirm: bool = False):
    """Publish the Python standalone mounts."""
    _check_prod(no_confirm)
    _run_mount_publish_script(ctx, "python_standalone", _python_standalone_args()).join()


@task(
    help={
        "name": "Image name (e.g. 'debian_slim')",
        "builder_version": "Image builder version (defaults to the latest)",
        "allow_global_deployment": "Required flag to confirm global deployment",
    },
)
def publish_base_images(
    ctx,
    name: str,
    builder_version: str | None = None,
    allow_global_deployment: bool = False,
    no_confirm: bool = False,
) -> None:
    """Publish base images. For example, `inv publish-base-images debian_slim`.

    These should be published as global deployments. However, publishing global
    deployments is *risky* because it would affect all workspaces. Pass the
    `--allow-global-deployment` flag to confirm this behavior."""
    if not allow_global_deployment:
        from rich.console import Console

        console = Console()
        console.print("This is a dry run. Rerun with `--allow-global-deployment` to publish.", style="yellow")

    if builder_version is None:
        from modal._image import LATEST_IMAGE_BUILDER_VERSION

        builder_version = LATEST_IMAGE_BUILDER_VERSION

    _check_prod(no_confirm)
    ctx.run(
        f"python -m modal_global_objects.images.base_images {name}",
        pty=True,
        env={
            "MODAL_IMAGE_ALLOW_GLOBAL_DEPLOYMENT": "1" if allow_global_deployment else "",
            "MODAL_IMAGE_BUILDER_VERSION": builder_version,
        },
    )


@task()
def prepare_with_version(ctx, version: str, dry_run: bool = False):
    """Overrides modal_version/__init__.py with version."""
    if not version:
        print("version must be specified")
        sys.exit(1)

    version_file = "modal_version/__init__.py"
    with open(version_file) as f:
        current_file_contents = f.read()

    version_pattern = r'__version__\s*=\s*["\']([^"\']+)["\']'

    if not re.search(version_pattern, current_file_contents):
        raise RuntimeError(f"Unable to find the line defining __version__ in {version_file}")

    updated_file_contents = re.sub(version_pattern, f'__version__ = "{version}"', current_file_contents)

    if dry_run:
        print(f"Would update {version_file} to the following:")
        print(updated_file_contents)
        return

    with open(version_file, "w") as f:
        f.write(updated_file_contents)


def get_next_dev_version(ctx) -> str:
    """Bumps the dev release based on the tags for `sdk-py/*`."""
    from packaging.version import Version

    from modal_version import __version__

    v = Version(__version__)

    if v.is_prerelease:
        if not v.is_devrelease:
            raise RuntimeError("We only know how to handle dev versions")
        # Already a dev release so we keep the version
        next_version = f"{v.major}.{v.minor}.{v.micro}"
    else:
        # Not a dev release so we bump the micro version
        next_version = f"{v.major}.{v.minor}.{v.micro + 1}"

    # Find any existing dev tags for next_version and bump to the next dev number.
    # ls-remote queries the remote tags directly
    result = ctx.run(f"git ls-remote --tags origin 'refs/tags/sdk-py/v{next_version}.dev*'", hide=True)

    max_dev = -1
    for line in result.stdout.splitlines():
        # Each line is "<sha>\trefs/tags/<tag>"; skip peeled tag entries (^{})
        ref = line.split("\t", 1)[-1]
        if ref.endswith("^{}"):
            continue
        tag_version = ref.removeprefix("refs/tags/sdk-py/v")
        with suppress(Exception):
            tag_v = Version(tag_version)
            if tag_v.dev is not None:
                max_dev = max(max_dev, tag_v.dev)

    return f"{next_version}.dev{max_dev + 1}"


@task()
def version(ctx, dev: bool = False):
    """Print version for current release or dev release."""
    if dev:
        print(get_next_dev_version(ctx))
    else:
        from modal_version import __version__

        print(__version__)


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

        def visit_ClassDef(self, node: ast.ClassDef):
            self.current_class = node.name
            self.generic_visit(node)
            self.current_class = None

        def get_message_string(self, message: ast.expr):
            # Handle a few different ways that the message can get passed to the deprecation helper
            # since it's not always a literal string (e.g. it's often a functions .__doc__ attribute)
            if isinstance(message, ast.Name):
                assert message  # mypy apparently thinks message can be None here
                message_str = str(self.assignments.get(message.id, ""))
            elif isinstance(message, ast.Attribute):
                assert message  # mypy apparently thinks message can be None here
                message_str = str(self.assignments.get(message.attr, ""))
            elif isinstance(message, ast.Constant):
                assert message  # mypy apparently thinks message can be None here
                message_str = str(message.s)
            elif isinstance(message, ast.JoinedStr):
                assert message  # mypy apparently thinks message can be None here
                message_str = "".join(str(v.s) for v in message.values if isinstance(v, ast.Constant))
            else:
                message_str = str(message)

            ret = message_str.replace("\n", " ")
            if len(ret) > (max_length := 80):
                ret = ret[:max_length] + "..."

            return ret

        def shared_visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
            self.current_function = node.name
            self.assignments["__doc__"] = ast.get_docstring(node)

            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue

                func = decorator.func
                if not isinstance(func, ast.Name):
                    continue

                if func.id != "with_deprecation_warning":
                    continue

                args = decorator.args
                if len(args) < 2:
                    continue

                date_element = args[0]
                if not isinstance(date_element, ast.Tuple):
                    continue

                depr_date = date(*(getattr(elt, "n") for elt in date_element.elts))
                function = (
                    f"{self.current_class}.{self.current_function}" if self.current_class else self.current_function
                )

                message_expr = args[1]
                message = self.get_message_string(message_expr)

                is_pending = any(
                    kw.arg == "pending" and isinstance(kw.value, ast.Constant) and kw.value.value is True
                    for kw in decorator.keywords
                )
                level = "[blue]pending[/blue]" if is_pending else "[yellow]warning[/yellow]"

                self.deprecations.append((str(depr_date), level, f"{self.fname}:{node.lineno}", function, message))

            self.generic_visit(node)
            self.current_function = None
            self.assignments.pop("__doc__", None)

        def visit_FunctionDef(self, node: ast.FunctionDef):
            return self.shared_visit_FunctionDef(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            return self.shared_visit_FunctionDef(node)

        def visit_Assign(self, node: ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.assignments[target.id] = node.value
            self.generic_visit(node)

        def visit_Attribute(self, node: ast.Attribute):
            self.assignments[node.attr] = node.value
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call):
            level: str | None = None
            if isinstance(node.func, ast.Name):
                if node.func.id == "deprecation_error":
                    level = "[red]error[/red]"
                # We may add a flag to make renamed_parameter error instead of warn
                # in which case this would get a little bit more complicated.
                elif node.func.id == "renamed_parameter":
                    level = "[yellow]warning[/yellow]"
                elif node.func.id == "deprecation_warning":
                    is_pending = any(
                        kw.arg == "pending" and isinstance(kw.value, ast.Constant) and kw.value.value is True
                        for kw in node.keywords
                    )
                    level = "[blue]pending[/blue]" if is_pending else "[yellow]warning[/yellow]"

            if level is not None and isinstance(node.func, ast.Name) and isinstance(node.args[0], ast.Tuple):
                depr_date = date(*(getattr(elt, "n") for elt in node.args[0].elts))
                function = (
                    f"{self.current_class}.{self.current_function}" if self.current_class else self.current_function
                )
                if node.func.id == "renamed_parameter":
                    old_name = getattr(node.args[1], "s")
                    new_name = getattr(node.args[2], "s")
                    message = f"Renamed parameter: {old_name} -> {new_name}"
                else:
                    message = self.get_message_string(node.args[1])

                self.deprecations.append((str(depr_date), level, f"{self.fname}:{node.lineno}", function, message))

    files = get_modal_source_files()
    deprecations = []
    for fname in files:
        with open(fname) as f:
            tree = ast.parse(f.read())
        visitor = FunctionCallVisitor(fname)
        visitor.visit(tree)
        deprecations.extend(visitor.deprecations)

    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table("Date", "Level", "Location", "Function", "Message")
    for row in sorted(deprecations, key=lambda r: r[0]):
        table.add_row(*row)
    console.print(table)
