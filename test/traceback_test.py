# Copyright Modal Labs 2024
import pytest
from pathlib import Path
from traceback import extract_tb
from typing import Dict, List, Tuple

from modal._traceback import append_modal_tb, extract_traceback, reduce_traceback_to_user_code
from modal._vendor import tblib

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


def make_tb_stack(frames: List[Tuple[str, str]]) -> List[Dict]:
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


def tb_dict_from_stack_dicts(stack: List[Dict]) -> Dict:
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
