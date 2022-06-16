import pytest

import modal
from modal.exception import InvalidError
from modal.functions import MAX_MEMORY_MB, MIN_MEMORY_MB


def test_function_memory_request(client):
    stub = modal.Stub()

    with pytest.raises(InvalidError):

        @stub.function(memory=MIN_MEMORY_MB - 1)
        def f1():
            pass

    with pytest.raises(InvalidError):

        @stub.function(memory=MAX_MEMORY_MB)
        def f2():
            pass

    @stub.function(memory=MAX_MEMORY_MB - 1)
    def f3():
        pass

    @stub.function(memory=MIN_MEMORY_MB)
    def f4():
        pass
