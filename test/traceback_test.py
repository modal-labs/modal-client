# Copyright Modal Labs 2024
from pathlib import Path
from traceback import extract_tb

from modal._traceback import append_modal_tb, extract_traceback

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
