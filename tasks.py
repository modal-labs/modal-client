import datetime
import json
import os
import re
import sys
from pathlib import Path
from typing import Literal

from invoke import context, task

project_root = Path(os.path.dirname(__file__))
os.chdir(project_root)


def get_current_js_version(package_path: Path):
    with package_path.open("r") as f:
        json_package = json.load(f)
        return json_package["version"]


# Checked-in version literals in the JS and Go SDKs. js/package.json is the
# source of truth the release tooling tags both SDKs from; these constants
# mirror it and are kept in sync by `update-version-go-js` and `lint-versions`.
JS_VERSION_PATH = Path("js/src/version.ts")
GO_VERSION_PATH = Path("go/version.go")
JS_VERSION_RE = re.compile(r'const SDK_VERSION = "([^"]+)"')
GO_VERSION_RE = re.compile(r'const sdkVersion = "([^"]+)"')


def get_checked_in_version(path: Path, pattern: re.Pattern) -> str:
    match = pattern.search(path.read_text())
    if match is None:
        raise RuntimeError(f"Could not find a version literal matching {pattern.pattern!r} in {path}")
    return match.group(1)


def set_checked_in_version(path: Path, pattern: re.Pattern, new_version: str):
    text = path.read_text()
    new_text, count = pattern.subn(lambda m: m.group(0).replace(m.group(1), new_version), text)
    if count != 1:
        raise RuntimeError(f"Expected exactly one version literal in {path}, found {count}")
    path.write_text(new_text)


def check_unreleased_has_items(changelog_content: str):
    """Check that there are items in the Unreleased section."""

    items_in_unreleased = []
    lines = changelog_content.splitlines()
    idx = 0
    while idx < len(lines):
        if lines[idx] != "## Unreleased":
            idx += 1
            continue
        # Find lines under unreleased
        idx += 1
        while idx < len(lines):
            if lines[idx].startswith("##"):
                break
            if lines[idx] and lines[idx].startswith("-"):
                items_in_unreleased.append(lines[idx])
            idx += 1

    for item in items_in_unreleased:
        if "No unreleased changes" in item:
            raise RuntimeError("Please update 'No unreleased changes' with changelog items.")

    if not items_in_unreleased:
        raise RuntimeError("Please add changelog items under the 'Unreleased' header.")


@task()
def update_version_go_js(
    ctx: context.Context,
    update: Literal["major", "minor", "patch"],
    dev: bool = False,
    dry_run: bool = False,
):
    modal_js_root = Path("js")
    package_json = modal_js_root / "package.json"
    current_version = get_current_js_version(package_json)

    with ctx.cd(modal_js_root):
        if dev:
            if "-dev." in current_version:
                ctx.run("npm version prerelease --no-git-tag-version", echo=True)
            else:
                ctx.run(f"npm version pre{update} --preid=dev --no-git-tag-version", echo=True)
        else:
            ctx.run(f"npm version {update} --no-git-tag-version", echo=True)

        new_version = get_current_js_version(package_json)

        # Keep the checked-in JS and Go version literals in sync with package.json.
        set_checked_in_version(JS_VERSION_PATH, JS_VERSION_RE, new_version)
        set_checked_in_version(GO_VERSION_PATH, GO_VERSION_RE, new_version)

        ctx.run("git diff", echo=True)

    if dry_run:
        ctx.run(
            f"git restore -- js/package.json js/package-lock.json {JS_VERSION_PATH} {GO_VERSION_PATH}",
            echo=True,
        )


def lint_protos_impl(ctx, proto_fname: str):
    with open(proto_fname) as f:
        proto_text = f.read()

    sections = ["import", "enum", "message", "service"]
    section_regex = "|".join(sections)
    matches = re.findall(rf"^((?:{section_regex})\s+(?:\w+))", proto_text, flags=re.MULTILINE)
    entities = [tuple(e.split()) for e in matches]

    from rich.console import Console

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
def lint_versions(ctx):
    """Ensure the JS and Go SDK versions are consistent.

    js/package.json is the source of truth the release tooling tags both SDKs
    from. This checks that the checked-in literals in js/src/version.ts and
    go/version.go match it, so the SDKs always report the same version.
    """
    from rich.console import Console

    console = Console()

    package_json = Path("js/package.json")
    versions = {
        str(package_json): get_current_js_version(package_json),
        str(JS_VERSION_PATH): get_checked_in_version(JS_VERSION_PATH, JS_VERSION_RE),
        str(GO_VERSION_PATH): get_checked_in_version(GO_VERSION_PATH, GO_VERSION_RE),
    }

    if len(set(versions.values())) > 1:
        console.print("[bold red]Version lint error:[/bold red] SDK versions are inconsistent:")
        for path, version in versions.items():
            console.print(f"  {path}: {version}")
        console.print(
            f"\nUpdate the checked-in versions to match {package_json} (the release source of "
            "truth). Running `inv update-version-go-js` keeps all three in sync automatically.",
            style="dim",
        )
        sys.exit(1)


CHANGELOG_PATHS = [Path("py/CHANGELOG.md"), Path("go/CHANGELOG.md"), Path("js/CHANGELOG.md")]


def lint_changelog_impl(changelog_path: Path) -> tuple[list[str], str | None]:
    """Validate the structure of one changelog.

    Returns any errors found along with the oldest version in the file, so callers can
    report the range that was checked.
    """
    errors: list[str] = []
    entry_re = re.compile(r"^### (\d+\.\d+\.\d+) \((\d{4}-\d{2}-\d{2})\)\s*$")
    section_re = re.compile(r"^## (.+)\s*$")

    current_section: str | None = None
    prev_version: tuple[int, ...] | None = None
    prev_version_str: str | None = None
    prev_date: str | None = None

    for lineno, line in enumerate(changelog_path.read_text().splitlines(), start=1):
        # Check section headers (## ...)
        section_match = section_re.match(line)
        if section_match:
            section_label = section_match.group(1).strip()
            if section_label == "Latest" or re.fullmatch(r"\d+\.\d+", section_label):
                current_section = section_label
            else:
                errors.append(
                    f"L{lineno}: unexpected section header '## {section_label}' (expected '## Latest' or '## X.Y')"
                )
            continue

        # Check entry headers (### ...)
        if line.startswith("### "):
            m = entry_re.match(line)
            if not m:
                errors.append(
                    f"L{lineno}: malformed entry header: {line.rstrip()!r} (expected '### X.Y.Z (YYYY-MM-DD)')"
                )
                continue

            version_str, date_str = m.group(1), m.group(2)
            # The entry regex guarantees three numeric components, so a tuple orders them correctly.
            version = tuple(int(part) for part in version_str.split("."))

            # Validate date format
            try:
                datetime.date.fromisoformat(date_str)
            except ValueError:
                errors.append(f"L{lineno}: invalid date {date_str!r} for version {version_str}")

            # Versions must be strictly decreasing
            if prev_version is not None and version >= prev_version:
                errors.append(
                    f"L{lineno}: version {version_str} is not strictly less than previous version {prev_version_str}"
                )

            # Dates must be non-increasing (same day is ok for multiple releases)
            if prev_date is not None and date_str > prev_date:
                errors.append(f"L{lineno}: date {date_str} for {version_str} is after previous entry date {prev_date}")

            # Check that entry belongs in the current section
            minor_prefix = f"{version[0]}.{version[1]}"
            if current_section == "Latest":
                pass  # Latest section holds the current minor series, no constraint needed
            elif current_section is not None and current_section != minor_prefix:
                errors.append(
                    f"L{lineno}: version {version_str} is under '## {current_section}' "
                    f"but belongs under '## {minor_prefix}'"
                )

            prev_version = version
            prev_version_str = version_str
            prev_date = date_str

    if prev_version_str is None:
        errors.append("no release entries found (expected '### X.Y.Z (YYYY-MM-DD)' headers)")

    return errors, prev_version_str


@task
def lint_changelogs(ctx):
    """Validate the structure of the Python, Go, and JS changelogs.

    Checks heading format, version ordering, date ordering, and section grouping.
    """
    from rich.console import Console

    console = Console()

    failed = False
    for changelog_path in CHANGELOG_PATHS:
        errors, oldest_version = lint_changelog_impl(changelog_path)
        if errors:
            failed = True
            console.print(f"[bold red]{changelog_path} has {len(errors)} error(s):[/bold red]")
            for error in errors:
                console.print(f"  {error}")
        else:
            print(f"{changelog_path} OK ({oldest_version} ... latest)")

    if failed:
        sys.exit(1)
