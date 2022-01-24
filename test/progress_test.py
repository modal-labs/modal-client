import io

from modal._progress import ProgressSpinner, Symbols


def get_test_spinner():
    buf = io.StringIO()
    return buf, ProgressSpinner(buf, frames="#", use_color=False)


def text_between(source, substr1, substr2):
    return source[source.index(substr1) + len(substr1) : source.index(substr2)]


CLEAR = "\033[K"


def test_single_tick():
    buf, p = get_test_spinner()
    p._tick()
    assert f"{CLEAR}# \r" == buf.getvalue()


def test_state_subtext():
    buf, p = get_test_spinner()
    p.step("foo", "done")
    p.set_substep_text("sub")
    p._tick()
    assert f"{CLEAR}{Symbols.ONGOING} foo\n{CLEAR}# sub\r" == buf.getvalue()


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
