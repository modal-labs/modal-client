# Copyright Modal Labs 2025
from dataclasses import dataclass

from modal._utils.blob_utils import MAX_ASYNC_OBJECT_SIZE_BYTES, MAX_OBJECT_SIZE_BYTES
from modal._utils.function_utils import should_upload
from modal_proto import api_pb2


@dataclass
class Input:
    size: int
    is_async: bool
    should_use_blob: bool


def test_should_upload():
    SMALL = MAX_ASYNC_OBJECT_SIZE_BYTES // 2
    LARGE = MAX_ASYNC_OBJECT_SIZE_BYTES * 2
    HUGE = MAX_OBJECT_SIZE_BYTES * 2
    test_cases = [
        Input(size=SMALL, is_async=True, should_use_blob=False),
        Input(size=SMALL, is_async=False, should_use_blob=False),
        Input(size=LARGE, is_async=True, should_use_blob=True),
        Input(size=LARGE, is_async=False, should_use_blob=False),
        Input(size=HUGE, is_async=True, should_use_blob=True),
        Input(size=HUGE, is_async=False, should_use_blob=True),
    ]

    for case in test_cases:
        used_blob = should_upload(
            case.size,
            max_object_size_bytes=MAX_OBJECT_SIZE_BYTES,
            function_call_invocation_type=(
                api_pb2.FUNCTION_CALL_INVOCATION_TYPE_ASYNC
                if case.is_async
                else api_pb2.FUNCTION_CALL_INVOCATION_TYPE_SYNC
            ),
        )

        assert used_blob == case.should_use_blob


def test_should_upload_with_max_object_size_bytes():
    # below threshold should not upload
    assert not should_upload(
        999,
        max_object_size_bytes=1000,
        function_call_invocation_type=api_pb2.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
    )
    # equal to threshold should not upload
    assert not should_upload(
        1000,
        max_object_size_bytes=1000,
        function_call_invocation_type=api_pb2.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
    )
    # above threshold should upload
    assert should_upload(
        1001,
        max_object_size_bytes=1000,
        function_call_invocation_type=api_pb2.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
    )


def test_should_upload_with_max_async_object_size_bytes():
    # A custom async threshold governs async invocations.
    assert not should_upload(
        500,
        max_object_size_bytes=MAX_OBJECT_SIZE_BYTES,
        function_call_invocation_type=api_pb2.FUNCTION_CALL_INVOCATION_TYPE_ASYNC,
        max_async_object_size_bytes=500,
    )
    assert should_upload(
        501,
        max_object_size_bytes=MAX_OBJECT_SIZE_BYTES,
        function_call_invocation_type=api_pb2.FUNCTION_CALL_INVOCATION_TYPE_ASYNC,
        max_async_object_size_bytes=500,
    )
    # The async threshold does not affect sync invocations (governed by max_object_size_bytes).
    assert not should_upload(
        501,
        max_object_size_bytes=MAX_OBJECT_SIZE_BYTES,
        function_call_invocation_type=api_pb2.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        max_async_object_size_bytes=500,
    )
