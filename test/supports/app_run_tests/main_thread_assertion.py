# Copyright Modal Labs 2022
import pytest
import threading

import modal

assert threading.current_thread() == threading.main_thread()

# can be checked to ensure module is loaded at all
pytest._did_load_main_thread_assertion = True  # type: ignore

stub = modal.Stub()


@stub.function()
def dummy():
    pass
