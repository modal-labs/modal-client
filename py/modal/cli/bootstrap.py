# Copyright Modal Labs 2026
import io
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import click
import typer

from modal._utils.async_utils import synchronizer
from modal._utils.http_utils import _http_client_with_tls
from modal.client import _Client
from modal.config import logger
from modal.output import OutputManager
from modal_proto import api_pb2

from .logo import GREEN, print_logo
from .selector import Selector


@synchronizer.create_blocking
async def bootstrap(
    name: Optional[str] = typer.Argument(None, help="The name of the template to load."),
    output: str = typer.Option(".", "-o", "--output", help="Location for storing the template."),
    force: bool = typer.Option(False, "--force", help="Overwrite the output directory if it already exists."),
) -> None:
    """Initialize a sample Modal App."""
    client = await _Client.from_env()
    resp = await client.stub.TemplateList(api_pb2.TemplateListRequest())
    names = sorted([item.name for item in resp.items])

    output_manager = OutputManager.get()
    print_logo()

    if name is None:
        if not names:
            output_manager.print("No templates available. Try updating modal.")
            raise typer.Exit(1)
        try:
            selector = Selector(names, title="Select a template", highlight_style=f"bold {GREEN}")
            name = selector.run()
        except Exception:
            output_manager.print("Available templates:")
            for n in names:
                output_manager.print(f"  - {n}")
            output_manager.print("\nRun `modal bootstrap <name>` to select a template.")
            raise typer.Exit(0)

    item = next((item for item in resp.items if item.name == name), None)
    if item is None:
        output_manager.print(f"Unknown template: {name}")
        output_manager.print(f"Available templates: {', '.join(names)}")
        raise typer.Exit(1)

    template_root = Path(output)
    template_root.mkdir(parents=True, exist_ok=True)
    template_written_to_cwd = template_root.resolve() == Path.cwd().resolve()

    dest = template_root / name
    if dest.exists() and not force:
        raise click.UsageError(f"Output path '{dest}' already exists. Use --force to overwrite.")

    # Download the repo archive
    ref = item.ref or "main"
    archive_url = item.repo.rstrip("/") + f"/archive/refs/heads/{ref}.zip"

    with output_manager.status(f"Downloading template '{name}'..."):
        try:
            session = _http_client_with_tls(timeout=30)
            try:
                async with session.get(archive_url) as resp_http:
                    resp_http.raise_for_status()
                    data = await resp_http.read()
            finally:
                await session.close()
        except Exception as e:
            output_manager.print(f"Failed to download template from {archive_url}: {e}")
            raise typer.Exit(1)

    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
        all_names = zf.namelist()
        root_prefix = all_names[0].split("/")[0] + "/"
    except Exception as e:
        logger.debug(f"Failed to parse template archive: {e}")
        output_manager.print(f"Unable to fetch template '{name}'.")
        raise typer.Exit(1)

    with zf:
        template_prefix = root_prefix + name + "/"
        logger.debug(f"Zip root: {root_prefix!r}, template: {template_prefix!r}, entries: {len(all_names)}")
        for entry in all_names[:20]:
            logger.debug(f"  {entry}")

        members = [m for m in all_names if m.startswith(template_prefix) and m != template_prefix]
        logger.debug(f"Matched {len(members)} members under {template_prefix!r}")

        if not members:
            output_manager.print(f"Template '{name}' not found in repository.")
            raise typer.Exit(1)

        if dest.exists() and force:
            shutil.rmtree(dest)

        dest.mkdir(parents=True, exist_ok=True)

        resolved_dest = dest.resolve()
        for member in members:
            rel_path = member[len(template_prefix) :]
            out_path = (dest / rel_path).resolve()
            if not out_path.is_relative_to(resolved_dest):
                logger.debug(f"Zip entry escapes destination: {member!r}")
                output_manager.print(f"Unable to fetch template '{name}'.")
                raise typer.Exit(1)
            if member.endswith("/"):
                out_path.mkdir(parents=True, exist_ok=True)
            else:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(zf.read(member))

    output_manager.print(f"[{GREEN}]✓[/{GREEN}] Template '{name}' written to {dest}", highlight=False)
    output_manager.print(f"[{GREEN}]→[/{GREEN}] To see it in action, run the following command:")
    amp = " [dim white]&&[/dim white] "
    if template_written_to_cwd:
        steps = []
    else:
        steps = [f"[#ff8de6]cd {template_root}[/#ff8de6]"]
    steps.extend(
        [
            f"[{GREEN}]modal deploy -m {name}.app[/{GREEN}]",
            f"[#91c8ef]python -m {name}.try[/#91c8ef]",
        ]
    )
    command = amp.join(steps)
    output_manager.print(command)
