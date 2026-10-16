# Copyright Modal Labs 2026
"""Consumer typing checks for multi_sync wrappers."""

import multi_sync.a
import multi_sync.b
from typing_extensions import assert_type


def _usage() -> None:
    ta = multi_sync.a.thread_and_loop_a()
    tb = multi_sync.b.thread_and_loop_b()
    assert_type(ta, tuple[int, int])
    assert_type(tb, tuple[int, int])
