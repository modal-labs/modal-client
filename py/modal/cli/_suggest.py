# Copyright Modal Labs 2026
"""Helpers for "Did you mean ...?" hints in CLI error messages."""

import difflib
import shlex
from collections.abc import Iterable, Sequence
from contextvars import ContextVar

import click

MAX_SUGGESTIONS = 3
MAX_NESTED_SUGGESTIONS = 5
CUTOFF = 0.6
MIN_PREFIX_LEN = 3

# Async handlers run on the synchronizer's thread, where Click's thread-local context isn't
# visible; this ContextVar reaches them via the context copied by run_coroutine_threadsafe.
_invocation_args: ContextVar[tuple[str, ...] | None] = ContextVar("modal_cli_invocation_args", default=None)


def _normalize(s: str) -> str:
    return s.lower().replace("_", "-")


def close_matches(
    word: str, candidates: Iterable[str], *, n: int = MAX_SUGGESTIONS, cutoff: float = CUTOFF
) -> list[str]:
    """Return up to `n` candidates that `word` was most likely meant to be, best first.

    Order of preference:
    1. A normalized exact match (case-insensitive, `_` == `-`).
    2. difflib similarity above `cutoff`.
    3. Candidates that `word` is a prefix of (only for words of 3+ characters).
    """
    unique = list(dict.fromkeys(c for c in candidates if c and c != word))
    if not word or not unique:
        return []

    by_norm: dict[str, str] = {}
    for c in unique:
        by_norm.setdefault(_normalize(c), c)

    norm_word = _normalize(word)
    if norm_word in by_norm:
        return [by_norm[norm_word]]

    matches = difflib.get_close_matches(norm_word, list(by_norm), n=n, cutoff=cutoff)
    if matches:
        return [by_norm[m] for m in matches]

    if len(word) >= MIN_PREFIX_LEN:
        prefixed = [c for norm, c in by_norm.items() if norm.startswith(norm_word)]
        return prefixed if len(prefixed) == 1 else []
    return []


def slash_ref_candidates(identifier: str) -> list[str]:
    """Possible `APP_NAME/OBJECT_NAME` rewrites of a dot-separated ref.

    Handles `app.func` and `app.Cls.method`. Returns an empty list for colon-separated refs
    or when the ref already contains `/`, since `/` is the only separator and
    `app.name/func.name` is already valid.
    """
    if not identifier or "/" in identifier or ":" in identifier:
        return []

    candidates: list[str] = []
    # App and object names may both contain dots, so every split is a candidate. Callers must
    # disambiguate (e.g. against deployed App names) rather than assume any one split.
    parts = identifier.split(".")
    for i in range(1, len(parts)):
        app, rest = ".".join(parts[:i]), ".".join(parts[i:])
        if app and rest:
            candidates.append(f"{app}/{rest}")

    return list(dict.fromkeys(candidates))


def corrected_commands(reference: str, suggestions: Sequence[str], *, allow_trailing_args: bool = True) -> list[str]:
    """The current invocation with `reference` replaced by each suggestion, as retry commands.

    Returns an empty list unless `reference` is exactly one argument of the invocation, since
    otherwise the argument that Click parsed as the reference is ambiguous. With
    `allow_trailing_args=False`, also returns an empty list when arguments follow `reference`.
    """
    args = _invocation_args.get()
    if args is None or args.count(reference) != 1:
        return []
    if not allow_trailing_args and args[-1] != reference:
        return []
    return [shlex.join(["modal", *(new if arg == reference else arg for arg in args)]) for new in suggestions]


def format_hint(suggestions: Sequence[str], commands: Sequence[str] = ()) -> str:
    """Render a hint block. `suggestions` are values; `commands` are full command lines.

    Returns an empty string when there is nothing to suggest.
    """
    items = commands or suggestions
    if not items:
        return ""
    return "\n".join(["Did you mean:", *(f"    {item}" for item in items)])


def _visible_commands(group: click.Group, ctx: click.Context) -> dict[str, click.Command]:
    out: dict[str, click.Command] = {}
    for name in group.list_commands(ctx):
        cmd = group.get_command(ctx, name)
        if cmd is not None and not cmd.hidden:
            out[name] = cmd
    return out


def unknown_command_hint(group: click.Group, ctx: click.Context, name: str) -> str:
    """Hint for `modal <name>` when `<name>` isn't a subcommand of `group`.

    Combines fuzzy matches at this level with exact matches one level down, so
    `modal logs` suggests `modal app logs`, `modal function logs`, ...
    """
    visible = _visible_commands(group, ctx)
    prefix = ctx.command_path
    commands = close_matches(name, visible)

    nested: list[str] = []
    for child_name, child in visible.items():
        if isinstance(child, click.Group) and not getattr(child, "dynamic_subcommands", False):
            sub_ctx = click.Context(child, info_name=child_name, parent=ctx, resilient_parsing=True)
            sub_visible = _visible_commands(child, sub_ctx)
            norm = _normalize(name)
            for sub_name in sub_visible:
                if _normalize(sub_name) == norm:
                    nested.append(f"{prefix} {child_name} {sub_name}")
    if len(nested) > MAX_NESTED_SUGGESTIONS:
        nested = nested[:MAX_NESTED_SUGGESTIONS] + [f"... (see '{prefix} --help')"]

    suggestions = list(dict.fromkeys(commands + nested))
    if len(suggestions) == 1 and not nested:
        return f"Did you mean '{suggestions[0]}'?"
    return format_hint(suggestions)
