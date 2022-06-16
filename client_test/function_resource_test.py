import pytest

import modal
from modal.exception import InvalidError
from modal.functions import MAX_MEMORY_MB, MIN_MEMORY_MB


def test_function_memory_request(client):
    stub = modal.Stub()

    with pytest.raises(InvalidError):

        @stub.function(memory=MIN_MEMORY_MB - 1)
        def f():
            pass

    with pytest.raises(InvalidError):

        @stub.function(memory=MAX_MEMORY_MB)
        def f():
            pass

    @stub.function(memory=MAX_MEMORY_MB - 1)
    def f():
        pass

    @stub.function(memory=MIN_MEMORY_MB)
    def f():
        pass
