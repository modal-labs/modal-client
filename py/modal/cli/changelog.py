# Copyright Modal Labs 2026
import json as json_mod
import re
import urllib.request
from dataclasses import asdict, dataclass
from typing import Optional

import typer

from modal._vendor.version import Version

CHANGELOG_URL = "https://modal.com/docs/reference/changelog.md"


@dataclass
class ChangelogEntry:
    version: str
    date: str
    body: str


def _fetch_changelog_markdown() -> str:
    req = urllib.request.Request(CHANGELOG_URL)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.read().decode("utf-8")


_HEADING_RE = re.compile(r"^### (\d+\.\d+\.\d+)\s+\((\d{4}-\d{2}-\d{2})\)\s*$", re.MULTILINE)
# Section headers like "## Latest", "## 1.2" that group entries in the raw markdown.
_SECTION_HEADER_RE = re.compile(r"^## .+$", re.MULTILINE)


def _preprocess(markdown: str) -> str:
    """Strip section headers (## ...) so they don't leak into entry bodies."""
    return _SECTION_HEADER_RE.sub("", markdown)


def _parse_entries(markdown: str) -> list[ChangelogEntry]:
    markdown = _preprocess(markdown)
    matches = list(_HEADING_RE.finditer(markdown))
    entries: list[ChangelogEntry] = []
    for i, m in enumerate(matches):
        version = m.group(1)
        date = m.group(2)
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[body_start:body_end].strip()
        entries.append(ChangelogEntry(version=version, date=date, body=body))
    return entries


def _filter_entries(
    entries: list[ChangelogEntry],
    *,
    last: Optional[int],
    since: Optional[str],
    for_version: Optional[str],
    newer: bool,
    all: bool,
    current_version: Version,
) -> list[ChangelogEntry]:
    if all:
        return entries
    if last is not None:
        # Find entries at or below the installed version, then take the first N
        at_or_below = [e for e in entries if Version(e.version) <= current_version]
        return at_or_below[:last]
    if since is not None:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", since):
            return [e for e in entries if e.date > since and Version(e.version) <= current_version]
        since_v = Version(since)
        return [e for e in entries if since_v < Version(e.version) <= current_version]
    if for_version is not None:
        n = len(for_version.split("."))
        if n not in (2, 3):
            raise typer.BadParameter(f"Invalid version format: {for_version!r}. Use X.Y or X.Y.Z.")
        fv = Version(for_version)
        return [e for e in entries if Version(e.version).release[:n] == fv.release[:n]]
    if newer:
        return [e for e in entries if Version(e.version) > current_version]

    # Default: current minor series
    current_series = current_version.release[:2]
    return [e for e in entries if Version(e.version).release[:2] == current_series]


def _format_entries(entries: list[ChangelogEntry], *, as_json: bool) -> str:
    if as_json:
        return json_mod.dumps([asdict(e) for e in entries], indent=2)
    parts: list[str] = []
    for entry in entries:
        parts.append(f"### {entry.version} ({entry.date})\n\n{entry.body}")
    return "\n\n".join(parts)


def changelog(
    last: Optional[int] = typer.Option(
        None, "--last", help="Show the N most recent entries before the installed version."
    ),
    since: Optional[str] = typer.Option(
        None, "--since", help="Show entries after a version (X.Y.Z) or date (YYYY-MM-DD), exclusive."
    ),
    for_version: Optional[str] = typer.Option(
        None, "--for", help="Show entries for a version (X.Y.Z) or series (X.Y)."
    ),
    newer: bool = typer.Option(False, "--newer", help="Show entries newer than the installed version."),
    all: bool = typer.Option(False, "--all", help="Show all entries."),
    json: bool = typer.Option(False, "--json", help="Output as JSON."),
):
    """Fetch release notes from the Modal changelog.

    This command prints changelog contents as markdown text and is useful for including
    information about recent updates in the context for agent development sessions.

    By default, the most recent updates in the current release series are shown. Other options
    allow for showing changes since a previous version, changes in a specific version, or changes
    that are newer than what's currently installed.

    Examples:

        modal changelog --since 1.2.0  # Show updates added after a specific version

        modal changelog --since 2026-01-01  # Show updates added after a specific date

        modal changelog --newer  # Show updates released after the currently installed version

        modal changelog --last 3  # Show updates included in the 3 most recent releases

        modal changelog --for 1.3.1  # Show the changelog for a specific release

    Note: when using `--since` or `--last`, only changes up to the currently installed version are shown.

    """
    from modal_version import __version__

    filter_flags = sum([last is not None, since is not None, for_version is not None, newer, all])
    if filter_flags > 1:
        raise typer.BadParameter("Only one of --last, --since, --for, --newer, or --all may be used at a time.")

    markdown = _fetch_changelog_markdown()
    entries = _parse_entries(markdown)
    filtered = _filter_entries(
        entries,
        last=last,
        since=since,
        for_version=for_version,
        newer=newer,
        all=all,
        current_version=Version(__version__),
    )

    if not filtered:
        typer.echo("No changelog entries found.")
        raise typer.Exit()

    output = _format_entries(filtered, as_json=json)
    typer.echo(output)
