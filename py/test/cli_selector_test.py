# Copyright Modal Labs 2026
import contextlib
import pytest
import sys
from unittest.mock import Mock, patch

if sys.platform == "win32":
    pytest.skip("Selector requires termios (Unix only)", allow_module_level=True)

from modal.cli.selector import Selector, _has_pending_input

# -- Helpers ------------------------------------------------------------------

PATCH_STDIN = "sys.stdin"
PATCH_STDOUT = "sys.stdout"


@contextlib.contextmanager
def _noop_cbreak():
    yield


@contextlib.contextmanager
def _noop_live(*args, **kwargs):
    live = Mock()
    yield live


def _make_selector(options=None, title="test"):
    """Create a Selector with TTY guards satisfied (safe for CI)."""
    if options is None:
        options = list(OPTIONS)
    with patch(PATCH_STDIN) as mock_stdin, patch(PATCH_STDOUT) as mock_stdout:
        mock_stdin.isatty.return_value = True
        mock_stdout.isatty.return_value = True
        return Selector(options, title)


OPTIONS = ["alpha", "beta", "gamma"]


# -- Constructor guards -------------------------------------------------------


def test_constructor_empty_options():
    with pytest.raises(ValueError, match="empty"):
        _make_selector(options=[])


def test_constructor_non_tty_stdin():
    with patch(PATCH_STDIN) as mock_stdin, patch(PATCH_STDOUT) as mock_stdout:
        mock_stdin.isatty.return_value = False
        mock_stdout.isatty.return_value = True
        with pytest.raises(RuntimeError, match="TTY"):
            Selector(["a", "b"])


def test_constructor_non_tty_stdout():
    with patch(PATCH_STDIN) as mock_stdin, patch(PATCH_STDOUT) as mock_stdout:
        mock_stdin.isatty.return_value = True
        mock_stdout.isatty.return_value = False
        with pytest.raises(RuntimeError, match="TTY"):
            Selector(["a", "b"])


# -- Selector state -----------------------------------------------------------


def test_selector_initial_state():
    sel = _make_selector(OPTIONS, "Pick one")
    assert sel.selected == 0
    assert sel.value == "alpha"


def test_selector_move_down():
    sel = _make_selector()
    sel.move_down()
    assert sel.value == "beta"
    sel.move_down()
    assert sel.value == "gamma"


def test_selector_move_down_wraps():
    sel = _make_selector()
    sel.move_down()
    sel.move_down()
    sel.move_down()
    assert sel.value == "alpha"


def test_selector_move_up_wraps():
    sel = _make_selector()
    sel.move_up()
    assert sel.value == "gamma"


def test_selector_move_up():
    sel = _make_selector()
    sel.move_down()
    sel.move_down()
    sel.move_up()
    assert sel.value == "beta"


# -- Selector.__rich__ (rendering) --------------------------------------------


def test_rich_selected_item_highlighted():
    sel = _make_selector(OPTIONS, "Pick one")
    result = sel.__rich__()
    plain = result.plain
    assert "Pick one" in plain
    assert "> alpha" in plain


def test_rich_non_selected_no_arrow():
    sel = _make_selector(OPTIONS, "Pick one")
    result = sel.__rich__()
    plain = result.plain
    for line in plain.splitlines():
        if "beta" in line or "gamma" in line:
            assert ">" not in line


def test_rich_title_present():
    sel = _make_selector(OPTIONS, "My Title")
    assert "My Title" in sel.__rich__().plain


def test_rich_second_option_selected():
    sel = _make_selector(OPTIONS, "Pick one")
    sel.move_down()
    result = sel.__rich__()
    plain = result.plain
    assert "> beta" in plain
    for line in plain.splitlines():
        if "alpha" in line:
            assert ">" not in line


# -- Selector.run keystroke handling ------------------------------------------

PATCH_CBREAK = "modal.cli.selector._cbreak_terminal"
PATCH_LIVE = "modal.cli.selector.Live"
PATCH_OS_READ = "os.read"
PATCH_HAS_PENDING = "modal.cli.selector._has_pending_input"

# Escape sequence bytes
ESC = b"\x1b"
BRACKET = b"["
UP = b"A"
DOWN = b"B"
ENTER = b"\r"


def _arrow_down():
    """Return the os.read side_effect entries for a down-arrow key."""
    return [ESC, BRACKET, DOWN]


def _arrow_up():
    """Return the os.read side_effect entries for an up-arrow key."""
    return [ESC, BRACKET, UP]


def _pending_for_arrow():
    """Return _has_pending_input side_effect entries for one arrow key.

    First call after ESC: True (there's a bracket coming).
    Second call after bracket: True (there's the direction byte).
    """
    return [True, True]


def _run_selector(read_side_effects, pending_side_effects, options=None):
    """Helper to run Selector.run() with mocked I/O."""
    if options is None:
        options = list(OPTIONS)
    with (
        patch(PATCH_CBREAK, _noop_cbreak),
        patch(PATCH_LIVE, _noop_live),
        patch(PATCH_OS_READ, side_effect=read_side_effects),
        patch(PATCH_HAS_PENDING, side_effect=pending_side_effects),
        patch(PATCH_STDIN) as mock_stdin,
        patch(PATCH_STDOUT) as mock_stdout,
    ):
        mock_stdin.fileno.return_value = 0
        mock_stdin.isatty.return_value = True
        mock_stdout.isatty.return_value = True
        return Selector(options, "test").run()


def test_run_enter_immediately():
    result = _run_selector([ENTER], [])
    assert result == "alpha"


def test_run_down_then_enter():
    reads = [*_arrow_down(), ENTER]
    pending = [*_pending_for_arrow()]
    result = _run_selector(reads, pending)
    assert result == "beta"


def test_run_up_wraps_to_last():
    reads = [*_arrow_up(), ENTER]
    pending = [*_pending_for_arrow()]
    result = _run_selector(reads, pending)
    assert result == "gamma"


def test_run_multiple_arrows():
    # Down, Down, Up -> index goes 0->1->2->1
    reads = [*_arrow_down(), *_arrow_down(), *_arrow_up(), ENTER]
    pending = [*_pending_for_arrow(), *_pending_for_arrow(), *_pending_for_arrow()]
    result = _run_selector(reads, pending)
    assert result == "beta"


def test_run_ctrl_c():
    with pytest.raises(KeyboardInterrupt):
        _run_selector([b"\x03"], [])


def test_run_eof():
    result = _run_selector([b""], [])
    assert result == "alpha"


def test_run_bare_escape_ignored():
    # Bare ESC (no pending input) followed by Enter -> still first option
    reads = [ESC, ENTER]
    pending = [False]  # After ESC, no pending input -> bare escape
    result = _run_selector(reads, pending)
    assert result == "alpha"


# -- _has_pending_input -------------------------------------------------------


def test_has_pending_input_data_ready():
    with patch("modal.cli.selector.select.select", return_value=([0], [], [])):
        assert _has_pending_input(0) is True


def test_has_pending_input_timeout():
    with patch("modal.cli.selector.select.select", return_value=([], [], [])):
        assert _has_pending_input(0) is False


# -- Interactive demo (python -m test.cli_selector_test) ----------------------

if __name__ == "__main__":
    import time

    from rich.console import Console, Group
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.text import Text

    options = ["gpu-a10g", "gpu-h100", "gpu-t4", "cpu-only"]
    selector = Selector(options, title="Select a hardware option")
    spinner = Spinner("dots", text=Text("Selecting a GPU!\n", style="bold cyan"))

    layout = Group(spinner, selector)
    console = Console()

    with Live(layout, console=console, refresh_per_second=12) as live:
        result = selector.run(live)
        # Swap out the layout to remove the selector
        spinner.update(text=Text(f"Provisioning {result}...", style="bold cyan"))
        live.update(spinner)
        time.sleep(2)

    console.print(f"[bold green]Done![/bold green] Provisioned {result}.")
