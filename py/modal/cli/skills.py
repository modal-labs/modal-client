# Copyright Modal Labs 2026

import asyncio
import re
import shutil
import tempfile
from importlib import resources
from pathlib import Path
from urllib.parse import urlparse

import click

from modal.cli.utils import confirm_or_suggest_yes, yes_option
from modal.output import OutputManager
from modal_version import __version__

from ._help import ModalGroup
from .logo import GREEN, print_logo

skills_cli = ModalGroup(name="skills", help="Install and update Modal's agent skills.")

_SKILL_RESOURCE = "skills/modal/SKILL.md"
_SKILL_NAME = "modal"
_LLMS_TXT_URL = "https://modal.com/llms.txt"
_DOCS_DOWNLOAD_TIMEOUT = 5.0
_DOCS_DOWNLOAD_CONCURRENCY = 20
# The slug allows `/` so nested doc paths (e.g. guide/functions/dynamic-config.md) are
# matched; `_doc_resource_path` guards against `..`/absolute-path traversal.
_DOC_URL_RE = re.compile(r"https://modal\.com/docs/(examples|guide)/[A-Za-z0-9._~%+/-]+\.md")


def _target_dir(*, claude: bool, global_install: bool) -> Path:
    base = Path.home() if global_install else Path.cwd()
    agent_dir = ".claude" if claude else ".agents"
    return base / agent_dir / "skills" / _SKILL_NAME


def _skill_resource():
    return resources.files("modal").joinpath(_SKILL_RESOURCE)


def _skill_content_for_install() -> str:
    content = _skill_resource().read_text()
    lines = content.splitlines(keepends=True)

    closing_idx = next(idx for idx, line in enumerate(lines[1:], start=1) if line.strip() == "---")

    # Record the SDK version under the frontmatter `metadata` map, where the Agent
    # Skills spec expects non-standard fields (rather than as a top-level key).
    version_line = f'  version: "{__version__}"\n'
    metadata_idx = next((idx for idx in range(1, closing_idx) if lines[idx].rstrip() == "metadata:"), None)
    if metadata_idx is None:
        lines.insert(closing_idx, "metadata:\n")
        lines.insert(closing_idx + 1, version_line)
        return "".join(lines)

    for idx in range(metadata_idx + 1, closing_idx):
        if lines[idx].strip() and not lines[idx][:1].isspace():
            break  # reached the next top-level key; the metadata map has no version
        if lines[idx].strip().startswith("version:"):
            lines[idx] = version_line
            return "".join(lines)
    lines.insert(metadata_idx + 1, version_line)
    return "".join(lines)


def _target_has_modal_skill(target_dir: Path) -> bool:
    try:
        lines = (target_dir / "SKILL.md").read_text().splitlines()
    except OSError:
        return False
    if not lines or lines[0].strip() != "---":
        return False

    for line in lines[1:]:
        if line.strip() == "---":
            return False
        if line.strip() == f"name: {_SKILL_NAME}":
            return True
    return False


def _extract_doc_urls(llms_txt: str) -> list[str]:
    urls = []
    seen = set()
    for match in _DOC_URL_RE.finditer(llms_txt):
        url = match.group(0)
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def _doc_resource_path(target_dir: Path, url: str) -> Path:
    path = urlparse(url).path
    examples_prefix = "/docs/examples/"
    guide_prefix = "/docs/guide/"
    if path.startswith(examples_prefix) and path.endswith(".md"):
        rel_path = Path("examples") / path[len(examples_prefix) :]
    elif path.startswith(guide_prefix) and path.endswith(".md"):
        rel_path = Path("guide") / path[len(guide_prefix) :]
    else:
        raise ValueError(f"Unexpected Modal documentation URL: {url}")

    if rel_path.is_absolute() or ".." in rel_path.parts:
        raise ValueError(f"Unexpected Modal documentation URL: {url}")
    return target_dir / "references" / rel_path


def _llms_section_body(llms_txt: str, heading: str) -> str | None:
    """Return the body of a `## {heading}` section of llms.txt, or None if absent."""
    lines = llms_txt.splitlines(keepends=True)
    start = next((idx + 1 for idx, line in enumerate(lines) if line.strip() == f"## {heading}"), None)
    if start is None:
        return None
    end = next((idx for idx in range(start, len(lines)) if lines[idx].startswith("## ")), len(lines))
    return "".join(lines[start:end]).strip("\n")


def _write_section_index(target_dir: Path, subdir: str, llms_txt: str) -> None:
    """Write references/{subdir}/index.md from the matching llms.txt section.

    The section preserves llms.txt's category groupings and page titles, with links to
    the pages we mirror locally rewritten to relative paths. Links to pages we don't
    download (section roots, non-`.md` pages) are dropped.
    """
    heading = subdir.capitalize()
    body = _llms_section_body(llms_txt, heading)
    if not body:
        return

    def _localize(match: re.Match[str]) -> str:
        url, section = match.group(0), match.group(1)
        if section != subdir:
            return url  # link into another section; left absolute so it is dropped below
        # Keep the path under the section so nested pages stay nested (functions/foo.md).
        return "./" + url.split(f"/docs/{section}/", 1)[1]

    localized = _DOC_URL_RE.sub(_localize, body)
    # Drop entries still pointing at absolute URLs (i.e. pages we didn't mirror locally).
    kept = [line for line in localized.splitlines() if "](http" not in line]
    index_dir = target_dir / "references" / subdir
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "index.md").write_text(f"# {heading}\n\n" + "\n".join(kept).strip("\n") + "\n")


def _replace_existing_target(target_dir: Path) -> None:
    if target_dir.is_dir() and not target_dir.is_symlink():
        shutil.rmtree(target_dir)
    else:
        target_dir.unlink()


def _http_session():
    import aiohttp

    return aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=_DOCS_DOWNLOAD_TIMEOUT))


async def _fetch_text(session, url: str) -> str:
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.text()


async def _download_doc(session, target_dir: Path, url: str) -> Exception | None:
    try:
        content = await _fetch_text(session, url)
        output_path = _doc_resource_path(target_dir, url)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
    except Exception as exc:
        return exc
    return None


async def _download_docs(target_dir: Path) -> None:
    async with _http_session() as session:
        try:
            llms_txt = await _fetch_text(session, _LLMS_TXT_URL)
        except Exception as exc:
            suffix = f": {str(exc)}" if str(exc) else ""
            click.echo(f"Warning: could not download Modal documentation{suffix}", err=True)
            return

        urls = _extract_doc_urls(llms_txt)
        semaphore = asyncio.Semaphore(_DOCS_DOWNLOAD_CONCURRENCY)

        async def _download_with_limit(url: str) -> Exception | None:
            async with semaphore:
                return await _download_doc(session, target_dir, url)

        failures = await asyncio.gather(*(_download_with_limit(url) for url in urls))

    failed = [(url, failure) for url, failure in zip(urls, failures) if failure is not None]
    failed_count = len(failed)
    if failed_count:
        failed_url, failure = failed[0]
        click.echo(
            f"Warning: failed to download {failed_count} of {len(urls)} Modal documentation resources. "
            f"First failure: {failed_url}: {failure}",
            err=True,
        )

    for subdir in ("guide", "examples"):
        _write_section_index(target_dir, subdir, llms_txt)


def _generate_api_reference_docs(target_dir: Path) -> None:
    """Generate the API reference locally from the installed client into references/api/.

    Introspecting the installed `modal` package keeps the reference in sync with the
    user's SDK version and removes the network dependency for this section.
    """
    import contextlib
    import io
    import warnings

    try:
        from modal_docs.gen_reference_docs import run
    except Exception as exc:
        click.echo(f"Warning: could not generate Modal API reference: {exc}", err=True)
        return

    api_dir = target_dir / "references" / "api"
    with tempfile.TemporaryDirectory() as raw_dir:
        # TODO(michael): doc generation is very noisy by default; make it configurable
        with (
            warnings.catch_warnings(),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            warnings.simplefilter("ignore")
            run(raw_dir)

        api_dir.mkdir(parents=True, exist_ok=True)
        for raw_file in Path(raw_dir).glob("*.md"):
            content = raw_file.read_text()
            # Strip the leading mdsvex `<script>` block; it is website-only boilerplate.
            if content.startswith("<script>"):
                content = content.partition("</script>\n")[2].lstrip("\n")
            (api_dir / raw_file.name).write_text(content)


def _install_or_update(*, claude: bool, global_install: bool, no_docs: bool, yes: bool, update: bool) -> None:
    target_dir = _target_dir(claude=claude, global_install=global_install)
    action = "Updating" if update else "Installing"
    completed_action = "Updated" if update else "Installed"
    output_manager = OutputManager.get()

    target_exists = target_dir.exists() or target_dir.is_symlink()
    if target_exists:
        target_has_modal_skill = _target_has_modal_skill(target_dir)
        if not yes and not (update and target_has_modal_skill):
            confirm_or_suggest_yes(f"Replace existing content at {target_dir}?")
    elif update and not yes:
        confirm_or_suggest_yes(f"No existing Modal skill found at {target_dir}. Install it there?")

    print_logo()

    # Build the output in a tmp directory so final installation is atomic.
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(dir=target_dir.parent, prefix=".modal-skill-"))
    try:
        with output_manager.status(f"{action} Modal skill..."):
            (staging_dir / "SKILL.md").write_text(_skill_content_for_install())
        if not no_docs:
            with output_manager.status("Downloading Modal documentation..."):
                asyncio.run(_download_docs(staging_dir))
                _generate_api_reference_docs(staging_dir)

        if target_exists:
            _replace_existing_target(target_dir)
        staging_dir.rename(target_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

    output_manager.print(f"[{GREEN}]✓[/{GREEN}] {completed_action} Modal skill to {target_dir}", highlight=False)


def _install_options(func):
    func = click.option("--claude", is_flag=True, default=False, help="Install to .claude/ rather than .agents/.")(func)
    func = click.option(
        "-g", "--global", "global_install", is_flag=True, default=False, help="Install in the user home directory."
    )(func)
    func = click.option(
        "--no-docs",
        is_flag=True,
        default=False,
        help="Skip downloading Modal documentation resources.",
    )(func)
    return yes_option(func)


@skills_cli.command("install", help="Install Modal skills.")
@_install_options
def install(*, claude: bool, global_install: bool, no_docs: bool, yes: bool):
    _install_or_update(claude=claude, global_install=global_install, no_docs=no_docs, yes=yes, update=False)


@skills_cli.command("update", help="Update installed Modal skills.")
@_install_options
def update(*, claude: bool, global_install: bool, no_docs: bool, yes: bool):
    _install_or_update(claude=claude, global_install=global_install, no_docs=no_docs, yes=yes, update=True)


@skills_cli.command("show", help="Print Modal skill content to the terminal.")
def show():
    click.echo(_skill_content_for_install(), nl=False)
