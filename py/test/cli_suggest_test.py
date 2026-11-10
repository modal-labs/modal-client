# Copyright Modal Labs 2026
import pytest

import click
from click.testing import CliRunner

from modal.cli._help import ModalGroup
from modal.cli._suggest import (
    close_matches,
    format_hint,
    slash_ref_candidates,
    unknown_command_hint,
)
from modal_proto import api_pb2


@pytest.fixture(autouse=True)
def no_command_suggestion_network(monkeypatch):
    monkeypatch.setattr("modal.cli._help._fetch_command_suggestion", lambda *args: api_pb2.CliCommandSuggestResponse())


@pytest.mark.parametrize(
    "word,candidates,expected",
    [
        ("fucntion", ["function", "app", "volume"], ["function"]),  # transposition
        ("My_App", ["my-app", "other"], ["my-app"]),  # case + separator normalization
        ("func", ["function", "app"], ["function"]),  # unique prefix
        ("zzz", ["function", "app"], []),  # nothing close
        ("app", ["app", "apps"], ["apps"]),  # never suggests the word itself
        ("", ["app"], []),
        ("trian", ["train", "train_lora", "serve"], ["train"]),  # far matches dropped
    ],
)
def test_close_matches(word, candidates, expected):
    assert close_matches(word, candidates) == expected


def test_close_matches_limits_results():
    candidates = [f"train{i}" for i in range(10)]
    assert len(close_matches("train", candidates)) == 3


def test_prefix_fallback_requires_unique_match():
    assert close_matches("abc", ["abcdefghijk"], cutoff=0.9) == ["abcdefghijk"]
    assert close_matches("abc", ["abcdefghijk", "abcdefghijkl"], cutoff=0.9) == []


@pytest.mark.parametrize(
    "identifier,expected",
    [
        ("my-app.train", ["my-app/train"]),
        ("my-app.Model.generate", ["my-app/Model.generate", "my-app.Model/generate"]),
        ("my-app::train", []),
        ("my-app:train", []),
        ("my-app.name::Model.train", []),
        ("my-app.name:Model.train", []),
        ("my-app/train", []),  # already correct shape
        ("my-app/Model.generate", []),  # dots after `/` are legal
        ("train", []),  # no separator at all: nothing to rewrite
        ("my-app.", []),
        (".train", []),
    ],
)
def test_slash_ref_candidates(identifier, expected):
    assert slash_ref_candidates(identifier) == expected


def test_format_hint():
    assert format_hint([]) == ""
    assert format_hint(["a/b"]) == "Did you mean:\n    a/b"
    assert format_hint(["a/b"], ["modal function logs a/b"]) == ("Did you mean:\n    modal function logs a/b")
    assert format_hint(["a", "b"]) == "Did you mean:\n    a\n    b"
    assert format_hint([], ["modal app logs"]) == "Did you mean:\n    modal app logs"
    assert format_hint(["a", "b"], ["modal app logs a", "modal app logs b"]) == (
        "Did you mean:\n    modal app logs a\n    modal app logs b"
    )


def _toy_cli() -> click.Group:
    @click.group(cls=ModalGroup, name="modal")
    def root():
        pass

    @root.group(name="app")
    def app():
        pass

    @root.group(name="function")
    def function():
        pass

    @root.group(name="secret-stuff", hidden=True)
    def hidden():
        pass

    for grp in (app, function):

        @grp.command(name="logs")
        def logs():
            pass

        @grp.command(name="list")
        def list_():
            pass

    return root


def test_unknown_command_hint_fuzzy_and_nested():
    root = _toy_cli()
    ctx = click.Context(root, info_name="modal")
    assert unknown_command_hint(root, ctx, "fucntion") == "Did you mean 'function'?"
    assert unknown_command_hint(root, ctx, "logs") == ("Did you mean:\n    modal app logs\n    modal function logs")
    # Hidden commands are never suggested.
    assert unknown_command_hint(root, ctx, "secret-stuf") == ""


def test_modal_group_reports_suggestion_on_unknown_command():
    result = CliRunner().invoke(_toy_cli(), ["fucntion", "logs", "my app/x", "--since", "1h"])
    assert result.exit_code == 2
    assert "No such command 'fucntion'." in result.output
    assert "Did you mean 'function'?" in result.output
    assert "my app/x" not in result.output
    assert "--since" not in result.output
    assert "modal function logs" not in result.output
    # Exactly one hint block, whatever click version is installed.
    assert result.output.count("Did you mean") == 1


def test_modal_group_nested_typo():
    result = CliRunner().invoke(_toy_cli(), ["app", "lgos"])
    assert result.exit_code == 2
    assert "Did you mean 'logs'?" in result.output


def test_modal_group_unknown_option_unchanged():
    # Option typos are click's job (NoSuchOption); we must not rewrite them as command errors.
    result = CliRunner().invoke(_toy_cli(), ["--nope"])
    assert result.exit_code == 2
    assert "No such option" in result.output
