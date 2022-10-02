import functools
import traceback
from typing import Any, Dict, Optional, Tuple

from rich.console import RenderResult, group
from rich.syntax import Syntax
from rich.text import Text
from rich.traceback import PathHighlighter, Stack, Traceback, install
from tblib import Traceback as TBLibTraceback

TBDictType = Dict[str, Any]
LineCacheType = Dict[Tuple[str, str], str]


def extract_traceback(exc: BaseException, task_id: str) -> Tuple[TBDictType, LineCacheType]:
    """Given an exception, extract a serializable traceback (with task ID markers included),
    and a line cache that maps (filename, lineno) to line contents. The latter is used to show
    a helpful traceback to the user, even if they don't have packages installed locally that
    are referenced in the traceback."""

    tb = TBLibTraceback(exc.__traceback__)
    # Prefix traceback file paths with <task_id>. This lets us attribute which parts of
    # the traceback came from specific remote containers, while still fitting in the TracebackType
    # spec. Real paths can never start with <; we can use this to extract task_ids from filenames
    # at the client.
    cur = tb
    while cur is not None:
        file = cur.tb_frame.f_code.co_filename

        # Paths starting with < indicate that they're from a traceback from a remote
        # container. This means we've reached the end of the local traceback.
        if file.startswith("<"):
            break
        cur.tb_frame.f_code.co_filename = f"<{task_id}>:{file}"
        cur = cur.tb_next

    tb_dict = tb.to_dict()

    line_cache = getattr(exc, "__line_cache__", {})

    for frame in traceback.extract_tb(exc.__traceback__):
        line_cache[(frame.filename, frame.lineno)] = frame.line

    return tb_dict, line_cache


def append_modal_tb(exc: BaseException, tb_dict: TBDictType, line_cache: LineCacheType) -> None:
    tb = TBLibTraceback.from_dict(tb_dict).as_traceback()

    # Filter out the prefix corresponding to internal Modal frames, and then make
    # the remote traceback from a Modal function the starting point of the current
    # exception's traceback.

    while tb is not None:
        if "/pkg/modal/" not in tb.tb_frame.f_code.co_filename:
            break
        tb = tb.tb_next

    exc.__traceback__ = tb

    setattr(exc, "__line_cache__", line_cache)


@group()
def _render_stack(self, stack: Stack) -> RenderResult:
    """Patched variant of rich.Traceback._render_stack that uses the line from the modal StackSummary,
    when the file isn't available to be read locally."""

    path_highlighter = PathHighlighter()
    theme = self.theme
    code_cache: Dict[str, str] = {}
    line_cache = getattr(stack, "line_cache", {})
    task_id = None

    def read_code(filename: str) -> str:
        code = code_cache.get(filename)
        if code is None:
            with open(filename, "rt", encoding="utf-8", errors="replace") as code_file:
                code = code_file.read()
            code_cache[filename] = code
        return code

    exclude_frames: Optional[range] = None
    if self.max_frames != 0:
        exclude_frames = range(
            self.max_frames // 2,
            len(stack.frames) - self.max_frames // 2,
        )

    excluded = False
    for frame_index, frame in enumerate(stack.frames):
        if exclude_frames and frame_index in exclude_frames:
            excluded = True
            continue

        if excluded:
            assert exclude_frames is not None
            yield Text(
                f"\n... {len(exclude_frames)} frames hidden ...",
                justify="center",
                style="traceback.error",
            )
            excluded = False

        first = frame_index == 0
        # Patched Modal-specific code.
        if frame.filename.startswith("<"):
            next_task_id, frame_filename = frame.filename.split(":", 2)
            next_task_id = next_task_id.strip("<>")
        else:
            frame_filename = frame.filename
            next_task_id = None
        suppressed = any(frame_filename.startswith(path) for path in self.suppress)

        if next_task_id != task_id:
            task_id = next_task_id
            yield ""
            yield Text(
                f"...Remote call to Modal Function ({task_id})...",
                justify="center",
                style="green",
            )

        text = Text.assemble(
            path_highlighter(Text(frame_filename, style="pygments.string")),
            (":", "pygments.text"),
            (str(frame.lineno), "pygments.number"),
            " in ",
            (frame.name, "pygments.function"),
            style="pygments.text",
        )
        if not frame_filename.startswith("<") and not first:
            yield ""

        yield text
        if not suppressed:
            try:
                code = read_code(frame_filename)
                lexer_name = self._guess_lexer(frame_filename, code)
                syntax = Syntax(
                    code,
                    lexer_name,
                    theme=theme,
                    line_numbers=True,
                    line_range=(
                        frame.lineno - self.extra_lines,
                        frame.lineno + self.extra_lines,
                    ),
                    highlight_lines={frame.lineno},
                    word_wrap=self.word_wrap,
                    code_width=88,
                    indent_guides=self.indent_guides,
                    dedent=False,
                )
                yield ""
            except Exception as error:
                # Patched Modal-specific code.
                line = line_cache.get((frame_filename, frame.lineno))
                if line:
                    try:
                        lexer_name = self._guess_lexer(frame_filename, line)
                        yield ""
                        yield Syntax(
                            line,
                            lexer_name,
                            theme=theme,
                            line_numbers=True,
                            line_range=(0, 1),
                            highlight_lines={frame.lineno},
                            word_wrap=self.word_wrap,
                            code_width=88,
                            indent_guides=self.indent_guides,
                            dedent=False,
                            start_line=frame.lineno,
                        )
                    except Exception:
                        yield Text.assemble(
                            (f"\n{error}", "traceback.error"),
                        )
                yield ""
            else:
                yield syntax


def setup_rich_traceback() -> None:
    from_exception = Traceback.from_exception

    @functools.wraps(Traceback.from_exception)
    def _from_exception(exc_type, exc_value, *args, **kwargs):
        """Patch from_exception to grab the Modal line_cache and store it with the
        Stack object, so it's available to render_stack at display time."""

        line_cache = getattr(exc_value, "__line_cache__", {})
        tb = from_exception(exc_type, exc_value, *args, **kwargs)
        for stack in tb.trace.stacks:
            stack.line_cache = line_cache  # type: ignore
        return tb

    Traceback._render_stack = _render_stack  # type: ignore
    Traceback.from_exception = _from_exception  # type: ignore

    import grpclib
    import synchronicity

    import modal
    import modal_utils

    install(suppress=[modal, synchronicity, modal_utils, grpclib], extra_lines=1)
