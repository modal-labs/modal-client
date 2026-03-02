import json
import os
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

            changelog_path = Path("CHANGELOG_GO_JS.md")
            update_changelog(changelog_path, new_version)

        ctx.run("git diff", echo=True)

    if dry_run:
        ctx.run("git restore -- js/package.json js/package-lock.json CHANGELOG_GO_JS.md", echo=True)
