import io
import pytest

from modal._progress import ProgressSpinner
from modal._terminfo import term_seq_str

FRAMES = range(1, 10)


def get_test_spinner():
    buf = io.StringIO()
    return buf, ProgressSpinner(buf, frames=FRAMES, use_color=False)


def text_between(source, substr1, substr2):
    return source[source.index(substr1) + len(substr1) : source.index(substr2)]


CLEAR = "\033[K"
LINE_END = "\n" + term_seq_str("cuu", 1)
CLEAR_TWO_LINES = term_seq_str("cr") + term_seq_str("cuu", 1) + term_seq_str("ed") + term_seq_str("el")
MESSAGE = "foo"
DONE_MESSAGE = "bar"


@pytest.fixture
def mocked_random(monkeypatch):
    return monkeypatch.setattr("random.randint", lambda a, b: 0)


def test_tick(mocked_random):
    buf, p = get_test_spinner()
    p.step(MESSAGE, DONE_MESSAGE)
    p._tick()
    assert f"{FRAMES[0]} {MESSAGE}{LINE_END}" == buf.getvalue()
    p._tick()
    print(repr(buf.getvalue()))
    assert f"{FRAMES[0]} {MESSAGE}{LINE_END}{CLEAR}{FRAMES[1]} {MESSAGE}{LINE_END}" == buf.getvalue()


def test_state_subtext(mocked_random):
    buf, p = get_test_spinner()
    p.step("foo", "done")
    p.substep("sub")
    p._tick()
    assert f"- foo\n{FRAMES[0]} sub{LINE_END}" == buf.getvalue().replace("\r\n", "\n")
    p._tick()
    assert (
        f"- foo\n{FRAMES[0]} sub{LINE_END}{CLEAR_TWO_LINES}- foo\n{FRAMES[1]} sub{LINE_END}"
        == buf.getvalue().replace("\r\n", "\n")
    )
