# Copyright Modal Labs 2023
import inspect
import json
import re
import sys
from pathlib import Path
from typing import cast

from click import Command, Context, Group

from modal.cli._help import group_commands_by_panel
from modal.cli.entry_point import entrypoint_cli


def _escape_svelte(text: str) -> str:
    """Wrap {placeholder} patterns in backticks so mdsvex/Svelte won't interpret them as expressions."""
    return re.sub(r"(?<!`)\{([a-z_]+)\}(?!`)", r"`{\1}`", text)


# Editorial copy that isn't derivable from the CLI source. The command groupings and
# one-line summaries below are generated from the live panel structure so they stay in
# sync with `modal --help`.
_INTRO_FRONTMATTER = (
    "---\n"
    "description: Complete reference for the Modal command-line interface. "
    "Documentation for run, deploy, serve, shell, and all modal CLI commands.\n"
    "---\n"
)

_INTRO_PREAMBLE = """\
# CLI Reference

This is the reference for the `modal` command-line interface, installed
alongside the [`modal`](https://pypi.org/project/modal/) Python package.

"""


def _escape_table_cell(text: str) -> str:
    """Escape text destined for a Markdown table cell."""
    return _escape_svelte(text).replace("|", "\\|")


def get_intro_docs() -> str:
    """Render the CLI reference index from the entrypoint's command panels.

    Each panel becomes a section whose table links to the per-command pages and
    shows the same short help that `modal --help` displays. Hidden commands are
    omitted, matching the terminal output.
    """
    entrypoint: Group = cast(Group, entrypoint_cli)
    sections = [f"{_INTRO_FRONTMATTER}\n{_INTRO_PREAMBLE}"]
    for panel_name, items in group_commands_by_panel(entrypoint).items():
        rows = ["|  |  |", "| --- | --- |"]
        for name, command in items:
            link = f"[`modal {name}`](/docs/cli/latest/{name})"
            short_help = _escape_table_cell(command.get_short_help_str(limit=250))
            rows.append(f"| {link} | {short_help} |")
        sections.append(f"## {panel_name}\n\n" + "\n".join(rows) + "\n")
    return "\n".join(sections)


# Adapted from typer_cli for generating CLI docs from click commands
# (see https://github.com/tiangolo/typer-cli/issues/50)
def get_docs_for_click(
    obj: Command,
    ctx: Context,
    *,
    indent: int = 0,
    name: str = "",
    call_prefix: str = "",
) -> str:
    docs = "#" * (1 + indent)
    command_name = name or obj.name
    if call_prefix:
        command_name = f"{call_prefix} {command_name}"
    title = f"`{command_name}`" if command_name else "CLI"
    docs += f" {title}\n\n"
    if obj.help:
        docs += f"{_escape_svelte(inspect.cleandoc(obj.help))}\n\n"
    usage_pieces = obj.collect_usage_pieces(ctx)
    if usage_pieces:
        docs += "**Usage**:\n\n"
        docs += "```shell\n"
        if command_name:
            docs += f"{command_name} "
        docs += f"{' '.join(usage_pieces)}\n"
        docs += "```\n\n"
    args = []
    opts = []
    for param in obj.get_params(ctx):
        rv = param.get_help_record(ctx)
        if rv is not None:
            if getattr(param, "hidden", False):
                continue
            if param.param_type_name == "argument":
                args.append(rv)
            elif param.param_type_name == "option":
                opts.append(rv)
    if args:
        docs += "**Arguments**:\n\n"
        for arg_name, arg_help in args:
            docs += f"* `{arg_name}`"
            if arg_help:
                docs += f": {_escape_svelte(arg_help)}"
            docs += "\n"
        docs += "\n"
    if opts:
        docs += "**Options**:\n\n"
        for opt_name, opt_help in opts:
            docs += f"* `{opt_name}`"
            if opt_help:
                docs += f": {_escape_svelte(opt_help)}"
            docs += "\n"
        docs += "\n"
    if obj.epilog:
        docs += f"{_escape_svelte(obj.epilog)}\n\n"
    if isinstance(obj, Group):
        group: Group = cast(Group, obj)
        commands = group.list_commands(ctx)
        if commands:
            docs += "**Commands**:\n\n"
            for command in commands:
                command_obj = group.get_command(ctx, command)
                assert command_obj
                if command_obj.hidden:
                    continue
                docs += f"* `{command_obj.name}`"
                command_help = command_obj.get_short_help_str(limit=250)
                if command_help:
                    docs += f": {_escape_svelte(command_help)}"
                docs += "\n"
            docs += "\n"
        for command in commands:
            command_obj = group.get_command(ctx, command)
            if command_obj.hidden:
                continue
            assert command_obj
            use_prefix = ""
            if command_name:
                use_prefix += f"{command_name}"
            docs += get_docs_for_click(obj=command_obj, ctx=ctx, indent=indent + 1, call_prefix=use_prefix)
    return docs


def run(output_dirname: str | None) -> None:
    entrypoint: Group = cast(Group, entrypoint_cli)
    ctx = Context(entrypoint)
    commands = entrypoint.list_commands(ctx)

    pages = {"intro": get_intro_docs()}
    # Top-level command names that get their own sidebar entry, in display order.
    sidebar_commands: list[str] = []
    for command in commands:
        command_obj = entrypoint.get_command(ctx, command)
        if command_obj.hidden:
            continue
        pages[command] = get_docs_for_click(obj=command_obj, ctx=ctx, call_prefix="modal")
        sidebar_commands.append(command)

    # The CLI sidebar is a flat, alphabetical list of commands. `list_commands`
    # already returns them sorted, but sort defensively in case that changes.
    sidebar_data = {"items": [{"label": name} for name in sorted(sidebar_commands)]}

    def _write_file(rel_path: str, data: str) -> None:
        if output_dirname is None:
            print(f"<<< {rel_path}")
            print(data)
            print(f">>> {rel_path}")
            return

        output_dir = Path(output_dirname)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / rel_path).write_text(data)

    for name, docs in pages.items():
        _write_file(f"{name}.md", docs)

    _write_file("sidebar.json", json.dumps(sidebar_data))


if __name__ == "__main__":
    run(None if len(sys.argv) <= 1 else sys.argv[1])
