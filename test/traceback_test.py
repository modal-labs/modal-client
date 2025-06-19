# Copyright Modal Labs 2024
import pytest
import types
import typing
from pathlib import Path
from traceback import extract_tb

from grpclib import GRPCError, Status

import modal
from modal._traceback import (
    append_modal_tb,
    extract_traceback,
    reduce_traceback_to_user_code,
    traceback_contains_remote_call,
)
from modal._vendor import tblib
from modal.exception import NotFoundError

from .supports.raise_error import raise_error

SUPPORT_MODULE = "supports.raise_error"


def call_raise_error():
    raise_error()


def test_extract_traceback():
    task_id = "ta-123"
    try:
        call_raise_error()
    except Exception as exc:
        tb_dict, line_cache = extract_traceback(exc, task_id)

    test_path = Path(__file__)
    support_path = test_path.parent / (SUPPORT_MODULE.replace(".", "/") + ".py")

    frame = tb_dict["tb_frame"]
    assert tb_dict["tb_lineno"] == frame["f_lineno"] - 2
    assert frame["f_code"]["co_filename"] == f"<{task_id}>:{test_path}"
    assert frame["f_code"]["co_name"] == "test_extract_traceback"
    assert frame["f_globals"]["__file__"] == str(test_path)
    assert frame["f_globals"]["__name__"] == f"test.{test_path.name[:-3]}"
    assert frame["f_locals"] == {}

    frame = tb_dict["tb_next"]["tb_frame"]
    assert frame["f_code"]["co_filename"] == f"<{task_id}>:{test_path}"
    assert frame["f_code"]["co_name"] == "call_raise_error"
    assert frame["f_globals"]["__file__"] == str(test_path)
    assert frame["f_globals"]["__name__"] == f"test.{test_path.name[:-3]}"
    assert frame["f_locals"] == {}

    frame = tb_dict["tb_next"]["tb_next"]["tb_frame"]
    assert frame["f_code"]["co_filename"] == f"<{task_id}>:{support_path}"
    assert frame["f_code"]["co_name"] == "raise_error"
    assert frame["f_globals"]["__file__"] == str(support_path)
    assert frame["f_globals"]["__name__"] == f"test.{SUPPORT_MODULE}"
    assert frame["f_locals"] == {}

    assert tb_dict["tb_next"]["tb_next"]["tb_next"] is None

    line_cache_list = list(line_cache.items())
    assert line_cache_list[0][0][0] == str(test_path)
    assert line_cache_list[0][1] == "call_raise_error()"
    assert line_cache_list[1][0][0] == str(test_path)
    assert line_cache_list[1][1] == "raise_error()"
    assert line_cache_list[2][0][0] == str(support_path)
    assert line_cache_list[2][1] == 'raise RuntimeError("Boo!")'


def test_append_modal_tb():
    task_id = "ta-123"
    try:
        call_raise_error()
    except Exception as exc:
        tb_dict, line_cache = extract_traceback(exc, task_id)

    try:
        raise RuntimeError("Remote error")
    except Exception as exc:
        remote_exc = exc
        append_modal_tb(exc, tb_dict, line_cache)

    assert remote_exc.__line_cache__ == line_cache  # type: ignore
    frames = [f.name for f in extract_tb(remote_exc.__traceback__)]
    assert frames == ["test_append_modal_tb", "call_raise_error", "raise_error"]


def make_tb_stack(frames: list[tuple[str, str]]) -> list[dict]:
    """Given a minimal specification of (code filename, code name), return dict formatted for tblib."""
    tb_frames = []
    for lineno, (filename, name) in enumerate(frames):
        tb_frames.append(
            {
                "tb_lineno": lineno,
                "tb_frame": {
                    "f_lineno": lineno,
                    "f_globals": {},
                    "f_locals": {},
                    "f_code": {"co_filename": filename, "co_name": name},
                },
            }
        )
    return tb_frames


def tb_dict_from_stack_dicts(stack: list[dict]) -> dict:
    tb_root = tb = stack.pop(0)
    while stack:
        tb["tb_next"] = stack.pop(0)
        tb = tb["tb_next"]
    tb["tb_next"] = None
    return tb_root


@pytest.mark.parametrize("user_mode", ["script", "module"])
def test_reduce_traceback_to_user_code(user_mode):
    if user_mode == "script":
        user_source, user_filename, user_name = ("/root/user/ai.py", "/root/user/ai.py", "train")
    elif user_mode == "module":
        user_source, user_filename, user_name = ("ai.training", "/root/user/ai/training.py", "<module>")

    stack = [
        ("/modal/__main__.py", "main"),
        ("/modal/entrypoint.py", "run"),
        ("/site-packages/synchronicity/wizard.py", "magic"),
        (user_filename, user_name),
        ("/modal/function.py", "execute"),
        ("/site-packages/synchronicity/devil.py", "pitchfork"),
    ]

    tb_dict = tb_dict_from_stack_dicts(make_tb_stack(stack))
    tb = tblib.Traceback.from_dict(tb_dict)
    tb_out = reduce_traceback_to_user_code(tb, user_source)

    f = tb_out.tb_frame
    assert f.f_code.co_filename == user_filename
    assert f.f_code.co_name == user_name

    f = tb_out.tb_next.tb_frame
    assert f.f_code.co_filename == "/modal/function.py"
    assert f.f_code.co_name == "execute"

    assert tb_out.tb_next.tb_next is None


def test_traceback_contains_remote_call():
    stack = [
        ("/home/foobar/code/script.py", "f"),
        ("/usr/local/venv/modal.py", "local"),
    ]

    tb = tblib.Traceback.from_dict(tb_dict_from_stack_dicts(make_tb_stack(stack)))
    assert not traceback_contains_remote_call(tb)

    task_id = "ta-0123456789ABCDEFGHILJKMNOP"
    stack.extend(
        [
            (f"<{task_id}>:/usr/local/lib/python3.11/importlib/__init__.py", ""),
            ("/root/script.py", ""),
        ]
    )

    tb = tblib.Traceback.from_dict(tb_dict_from_stack_dicts(make_tb_stack(stack)))
    assert traceback_contains_remote_call(tb)


ModuleOrFilename = typing.Union[types.ModuleType, str]


def to_path(mof: ModuleOrFilename) -> Path:
    if isinstance(mof, str):
        return Path(mof)
    module_file = Path(mof.__file__)
    if module_file.name == "__init__.py":
        return module_file.parent
    return module_file


def assert_expected_traceback(traceback, expected_module_frames: list[tuple[ModuleOrFilename, str]]):
    failure = False
    for i, frame in enumerate(traceback):
        if i >= len(expected_module_frames):
            failure = "(past end of expected traceback)"
        else:
            expected_path = to_path(expected_module_frames[i][0])
            expected_name = expected_module_frames[i][1]
            if expected_path != Path(frame.path) or expected_name != frame.name:
                failure = f"Expected: {str(expected_path)}, {expected_name}"

        if failure:
            full_tb = "\n".join(f"{'>>>' if i == j else ''}{frame}" for j, frame in enumerate(traceback))
            raise AssertionError(f"Unexpected traceback frame:\n{full_tb}\n{failure}")


def test_internal_frame_suppression_graceful_error(set_env_client, servicer):
    # when converting a grpc error into a modal error, like modal.exceptions.NotFoundError
    with pytest.raises(NotFoundError):
        modal.Queue.from_name("asdlfjkjalsdkf").get()

    with servicer.intercept() as ctx:

        async def QueueGetOrCreate(self, stream):
            raise GRPCError(Status.NOT_FOUND)

        ctx.set_responder("QueueGetOrCreate", QueueGetOrCreate)

        with pytest.raises(NotFoundError) as exc_info:
            modal.Queue.from_name("asdlfjkjalsdkf").get()

        assert_expected_traceback(
            exc_info.traceback,
            [
                (__file__, "test_internal_frame_suppression_graceful_error"),  # this frame
                (modal._object, "wrapped"),  # from @live_method calling .hydrate()
                (modal.queue, "_load"),
            ],
        )


def test_internal_frame_suppression_internal_error(set_env_client, servicer):
    with pytest.raises(NotFoundError):
        modal.Queue.from_name("asdlfjkjalsdkf").get()

    with servicer.intercept() as ctx:

        async def QueueGetOrCreate(self, stream):
            raise GRPCError(status=Status.INTERNAL, message="kaboom")

        ctx.set_responder("QueueGetOrCreate", QueueGetOrCreate)
        with pytest.raises(GRPCError, match="kaboom") as exc_info:
            modal.Queue.from_name("asdlfjkjalsdkf").get()

        assert_expected_traceback(
            exc_info.traceback,
            [
                (__file__, "test_internal_frame_suppression_internal_error"),  # this frame
                (modal._object, "wrapped"),  # from @live_method calling .hydrate()
                (modal.queue, "_load"),
            ],
        )


def test_internal_frame_suppression_full_trace(set_env_client, servicer, monkeypatch):
    monkeypatch.setenv("MODAL_TRACEBACK", "1")

    with pytest.raises(NotFoundError):
        modal.Queue.from_name("asdlfjkjalsdkf").get()

    with servicer.intercept() as ctx:

        async def QueueGetOrCreate(self, stream):
            await stream.recv_message()
            raise GRPCError(status=Status.INTERNAL, message="kaboom")

        ctx.set_responder("QueueGetOrCreate", QueueGetOrCreate)
        with pytest.raises(GRPCError, match="kaboom") as exc:
            modal.Queue.from_name("asdlfjkjalsdkf").get()

        print(len(exc.traceback))

        for frame in exc.traceback:
            print(frame.name, frame.statement, frame.path)
