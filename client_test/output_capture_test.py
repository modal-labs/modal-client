import os
import pty
import pytest
import subprocess
import sys
import time
from contextlib import contextmanager

from modal._output_capture import thread_capture
from modal_utils.async_utils import synchronizer


@pytest.fixture(scope="class")
def suspend_capture(pytestconfig):
    # Pytest itself does global capture on a file descriptor level which
    # for some reason interacts with our capture, so we need to suspend
    # pytest capture when running the subprocess test below

    # Code from:
    # https://github.com/pytest-dev/pytest/issues/1599

    @contextmanager
    def suspend_block():
        capman = pytestconfig.pluginmanager.getplugin("capturemanager")
        capman.suspend_global_capture(in_=True)
        try:
            yield
        finally:
            capman.suspend_global_capture(in_=True)

    return suspend_block()


@pytest.fixture
def capture_stdout_as_list(suspend_capture):
    @synchronizer.asynccontextmanager
    async def ctx():
        caps = []

        def callback(line, _):
            # When using a pty \n get replaced with \r\n.
            # However, a pty is only used when stdout buffer is a tty, which is the case in local
            # testing, but not in the Github Actions runner.
            caps.append(line.replace("\r\n", "\n"))

        with suspend_capture:
            async with thread_capture(sys.stdout, callback):
                yield caps

    return ctx


@pytest.mark.asyncio
async def test_capture_prints(capture_stdout_as_list):
    async with capture_stdout_as_list() as caps:
        # in case line buffering is turned off for sys.stdout in the test env
        print("foo", flush=True)
        time.sleep(0.1)  # TODO: can we remove this
        # test that we actually capture continually and not just at end of block
        assert caps == ["foo\n"]
        print("bar")

    # intentionally no wait here - all results should be captured when capture block ends
    assert caps == ["foo\n", "bar\n"]


@pytest.mark.asyncio
async def test_capture_partial_line(capture_stdout_as_list):
    async with capture_stdout_as_list() as caps:
        # in case line buffering is turned off for sys.stdout in the test env
        sys.stdout.write("no newline")

    # intentionally no wait here - all results should be captured when capture block ends
    assert caps == ["no newline"]


@pytest.mark.asyncio
async def test_capture_empty_line(capture_stdout_as_list):
    async with capture_stdout_as_list() as caps:
        # in case line buffering is turned off for sys.stdout in the test env
        sys.stdout.write("\n")

    # intentionally no wait here - all results should be captured when capture block ends
    assert caps == ["\n"]


@pytest.mark.asyncio
async def test_error_bubbles_through_and_shuts_down_thread(suspend_capture, capsys):
    caps = []

    def callback(line, original_stream):
        caps.append(line)

    class DummyException(Exception):
        pass

    with capsys.disabled():
        with suspend_capture:
            with pytest.raises(DummyException):
                async with thread_capture(sys.stdout, callback):
                    # in case line buffering is turned off for sys.stdout in the test env
                    print("foo")
                    raise DummyException()

    print("bar")
    out, err = capsys.readouterr()
    assert out == "bar\n"


@pytest.mark.asyncio
async def test_capture_subprocess(capture_stdout_as_list):
    async with capture_stdout_as_list() as caps:
        subprocess.call(["echo", "foo"])

    assert caps == ["foo\n"]


@pytest.mark.skip("Fails in Github Actions runner; TODO: investigate")
@pytest.mark.asyncio
async def test_capture_tty(suspend_capture):
    _, s = pty.openpty()
    reader = os.fdopen(s, "r")

    def callback(line, _):
        pass

    with suspend_capture:
        async with thread_capture(reader, callback):
            assert reader.isatty()


@pytest.mark.asyncio
async def test_capture_line_boundaries(capture_stdout_as_list):
    async with capture_stdout_as_list() as caps:
        sys.stdout.write("abc")
        sys.stdout.flush()
        time.sleep(0.01)
        assert caps == []
        sys.stdout.write("d\nx")
        sys.stdout.flush()
        time.sleep(0.01)
        assert caps == ["abcd\n"]
        sys.stdout.write("\ryz\r")
        sys.stdout.flush()
        time.sleep(0.01)
        assert caps == ["abcd\n", "x\r", "yz\r"]
