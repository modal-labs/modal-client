import json
import os
import re
import sys
from pathlib import Path
from textwrap import dedent
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


def update_changelog(changelog_path: Path, new_version: str):
    changelog_content = changelog_path.read_text()
    check_unreleased_has_items(changelog_content)

    version_header = f"js/v{new_version}, go/v{new_version}"
    new_header = dedent(f"""\
    ## Unreleased

    No unreleased changes.

    ## {version_header}""")

    new_changelog_content = changelog_content.replace("## Unreleased", new_header)
    changelog_path.write_text(new_changelog_content)


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

        if not dev:
            changelog_path = Path("CHANGELOG_GO_JS.md")
            update_changelog(changelog_path, new_version)

        ctx.run("git diff", echo=True)

    if dry_run:
        ctx.run(
            "git restore -- js/package.json js/package-lock.json"
            f" {JS_VERSION_PATH} {GO_VERSION_PATH} CHANGELOG_GO_JS.md",
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
