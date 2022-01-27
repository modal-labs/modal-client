import io

from modal._progress import ProgressSpinner, Symbols

FRAMES = range(1, 10)


def get_test_spinner():
    buf = io.StringIO()
    return buf, ProgressSpinner(buf, frames=FRAMES, use_color=False)


def text_between(source, substr1, substr2):
    return source[source.index(substr1) + len(substr1) : source.index(substr2)]


CLEAR = "\033[K"
CLEAR_TWO_LINES = "\r\033[1A\033[J"


def test_tick():
    buf, p = get_test_spinner()
    p._tick()
    assert f"{FRAMES[0]} \r" == buf.getvalue()
    p._tick()
    assert f"{FRAMES[0]} \r{CLEAR}{FRAMES[1]} \r" == buf.getvalue()


def test_state_subtext():
    buf, p = get_test_spinner()
    p.step("foo", "done")
    p.set_substep_text("sub")
    p._tick()
    assert f"{Symbols.ONGOING} foo\n{FRAMES[0]} sub\r" == buf.getvalue()
    p._tick()
    assert (
        f"{Symbols.ONGOING} foo\n{FRAMES[0]} sub\r{CLEAR_TWO_LINES}{Symbols.ONGOING} foo\n{FRAMES[1]} sub\r"
        == buf.getvalue()
    )


def test_overwrite():
    buf, p = get_test_spinner()
    p.set_substep_text("something_long")
    p._tick()
    p.set_substep_text("short")
    p._tick()
    v = buf.getvalue()
    between = text_between(v, "something_long", "short")
    assert "\n" not in between, "no newlines between status messages"
    assert CLEAR in between, "clear line before next status"
