# Copyright Modal Labs 2026
"""Custom help formatting for the Modal CLI."""

from __future__ import annotations

import inspect
import os
import sys
from typing import Any, Optional

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

_HEADING_STYLE = "bold bright_green"
_COMMAND_NAME_STYLE = ""
_COMMAND_DESC_STYLE = "dim"
_OPTION_FLAG_STYLE = "green"
_OPTION_METAVAR_STYLE = "dim"
_ERROR_STYLE = "red"

_MAX_HELP_WIDTH = 80


def use_rich_style() -> bool:
    """Whether help output should be rendered in the rich style."""
    env = os.environ.get("MODAL_RICH_CLI")  # TODO move to config
    if env in ("0", "1"):
        return env == "1"
    return sys.stdout.isatty()


def _make_console(file: Optional[Any] = None) -> Console:
    return Console(file=file, highlight=False)


def _render_usage(cmd: click.Command, ctx: click.Context, console: Console) -> None:
    pieces = " ".join(cmd.collect_usage_pieces(ctx))
    usage = Text()
    usage.append("Usage: ", style=_HEADING_STYLE)
    usage.append(f"{ctx.command_path} {pieces}".rstrip())
    console.print(usage)
    console.print()


def _render_help_text(cmd: click.Command, console: Console) -> None:
    text = cmd.help or cmd.short_help or ""
    if not text:
        return
    console.print(Padding(Markdown(inspect.cleandoc(text)), (0, 2)))
    console.print()


def _option_label(param: click.Parameter, ctx: click.Context) -> Text:
    text = Text()
    for i, opt in enumerate(param.opts):
        if i > 0:
            text.append(", ")
        text.append(opt, style=_OPTION_FLAG_STYLE)
    if param.secondary_opts:
        text.append(" / ")
        for i, opt in enumerate(param.secondary_opts):
            if i > 0:
                text.append(", ")
            text.append(opt, style=_OPTION_FLAG_STYLE)
    if not getattr(param, "is_flag", False) and not getattr(param, "count", False):
        text.append(" ")
        text.append(param.make_metavar(ctx), style=_OPTION_METAVAR_STYLE)
    return text


def _render_options(cmd: click.Command, ctx: click.Context, console: Console) -> None:
    rows: list[tuple[Text, str]] = []
    for param in cmd.get_params(ctx):
        rec = param.get_help_record(ctx)
        if rec is None:  # skips arguments and hidden options
            continue
        rows.append((_option_label(param, ctx), rec[1] or ""))
    if not rows:
        return

    console.print(Text("Options", style=_HEADING_STYLE))
    table = Table(box=None, show_header=False, pad_edge=False, padding=(0, 2))
    table.add_column(no_wrap=True)  # styling lives in the Text cells
    table.add_column(overflow="fold")
    for label, help_str in rows:
        table.add_row(label, help_str)
    console.print(table)
    console.print()


def _render_epilog(cmd: click.Command, console: Console) -> None:
    if cmd.epilog:
        console.print(cmd.epilog)
        console.print()


def _group_commands_by_panel(group: click.Group) -> dict[str, list[tuple[str, click.Command]]]:
    """Bucket visible subcommands, preserving registration order."""
    panels: dict[str, list[tuple[str, click.Command]]] = {}
    for name, sub in group.commands.items():
        if sub.hidden:
            continue
        panels.setdefault(getattr(sub, "panel", None) or "Commands", []).append((name, sub))
    return panels


def _render_commands(group: click.Group, console: Console) -> None:
    panels = _group_commands_by_panel(group)
    if not panels:
        return

    # We want the name / description columns to be the same widths across groups
    name_width = max(len(name) for items in panels.values() for name, _ in items)
    table_width = min(_MAX_HELP_WIDTH, console.width)

    for panel_name, items in panels.items():
        console.print(Text(panel_name.ljust(table_width), style=f"{_HEADING_STYLE} underline"))
        _render_command_table(console, items, name_width, table_width)
        console.print()


def build_command_table(name_width: int, table_width: Optional[int] = None) -> Table:
    kwargs: dict[str, Any] = {"box": None, "show_header": False, "pad_edge": False, "padding": (0, 2)}
    if table_width is not None:
        kwargs["width"] = table_width
    table = Table(**kwargs)
    table.add_column(style=_COMMAND_NAME_STYLE, no_wrap=True, width=name_width)
    # ratio=1 claims any extra width for the description column so col1 stays
    # fixed at name_width across panels — otherwise Rich distributes slack into
    # both columns and the name column grows in panels with shorter content.
    table.add_column(style=_COMMAND_DESC_STYLE, overflow="fold", ratio=1)
    return table


def _render_command_table(
    console: Console,
    items: list[tuple[str, click.Command]],
    name_width: int,
    table_width: int,
) -> None:
    table = build_command_table(name_width, table_width)
    for name, sub in items:
        table.add_row(name, sub.get_short_help_str(limit=80))
    console.print(table)


# -- Public Command / Group subclasses ---------------------------------------


class ModalCommand(click.Command):
    """click.Command that renders --help with custom rich output."""

    def __init__(self, *args: Any, panel: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.panel = panel

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        if not use_rich_style():
            return super().format_help(ctx, formatter)
        console = _make_console()
        with console.capture() as capture:
            _render_usage(self, ctx, console)
            _render_help_text(self, console)
            _render_options(self, ctx, console)
            _render_epilog(self, console)
        formatter.write(capture.get())


class ModalGroup(click.Group):
    """click.Group whose subcommands and subgroups inherit `ModalCommand`."""

    command_class = ModalCommand
    group_class = type  # nested @group.group() reuses the enclosing class

    def __init__(self, *args: Any, panel: Optional[str] = None, **kwargs: Any) -> None:
        # Default to showing help when a group is invoked with no subcommand.
        # Callers can opt out by passing no_args_is_help=False explicitly.
        kwargs.setdefault("no_args_is_help", True)
        super().__init__(*args, **kwargs)
        self.panel = panel

    def add_command(
        self,
        cmd: click.Command,
        name: Optional[str] = None,
        *,
        panel: Optional[str] = None,
        hidden: Optional[bool] = None,
    ) -> None:
        super().add_command(cmd, name)
        if panel is not None:
            cmd.panel = panel  # type: ignore[attr-defined]
        if hidden is not None:
            cmd.hidden = hidden

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        # Replaces click's single flat "Commands:" section with one section per
        # panel so the simple-style help output still preserves grouping.
        for panel_name, items in _group_commands_by_panel(self).items():
            rows = [(name, sub.get_short_help_str(limit=80)) for name, sub in items]
            with formatter.section(panel_name):
                formatter.write_dl(rows)

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        if not use_rich_style():
            return super().format_help(ctx, formatter)
        console = _make_console()
        with console.capture() as capture:
            _render_usage(self, ctx, console)
            _render_help_text(self, console)
            _render_options(self, ctx, console)
            _render_commands(self, console)
            _render_epilog(self, console)
        formatter.write(capture.get())


# -- Error rendering ---------------------------------------------------------


def _render_click_exception(exc: click.ClickException, file: Any) -> None:
    console = _make_console(file if file is not None else sys.stderr)
    title = "Error"

    if isinstance(exc, click.UsageError) and exc.ctx is not None:
        ctx = exc.ctx
        console.print(ctx.get_usage())
        if ctx.command.get_help_option(ctx) is not None:
            option = ctx.help_option_names[0] if ctx.help_option_names else "--help"
            console.print(f"Try [bold]'{ctx.command_path} {option}'[/bold] for help.")

    console.print(
        Panel(
            Text(exc.format_message()),
            title=title,
            title_align="left",
            border_style=_ERROR_STYLE,
            expand=True,
        )
    )


_orig_click_show = click.ClickException.show
_orig_usage_show = click.UsageError.show


def _click_show(self: click.ClickException, file: Any = None) -> None:
    if not use_rich_style():
        return _orig_click_show(self, file)
    _render_click_exception(self, file)


def _usage_show(self: click.UsageError, file: Any = None) -> None:
    if not use_rich_style():
        return _orig_usage_show(self, file)
    _render_click_exception(self, file)


click.ClickException.show = _click_show  # type: ignore[method-assign]
click.UsageError.show = _usage_show  # type: ignore[method-assign]
