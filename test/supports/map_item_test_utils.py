# Copyright Modal Labs 2025

# This file contains helpers for map_item_context_test.py and map_item_manager_test.py

from dataclasses import dataclass
from typing import Union

from modal._utils.jwt_utils import DecodedJwt
from modal.parallel_map import _MapItemContext, _MapItemState
from modal_proto import api_pb2

result_success = api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS)
result_failure = api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_FAILURE)
result_internal_failure = api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_INTERNAL_FAILURE)


@dataclass
class InputJwtData:
    """
    A helper class that represents a decoded input jwt that contains a map idx and retry count.
    """

    idx: int
    retry_count: int

    @staticmethod
    def of(idx: int, retry_count: int) -> "InputJwtData":
        return InputJwtData(idx, retry_count)

    @staticmethod
    def from_jwt(jwt: str) -> "InputJwtData":
        decoded = DecodedJwt.decode_without_verification(jwt)
        return InputJwtData(decoded.payload["idx"], decoded.payload["retry_count"])

    def to_jwt(self) -> str:
        return DecodedJwt.encode_without_signature({"idx": self.idx, "retry_count": self.retry_count})


def assert_context_is(
    ctx: _MapItemContext,
    state: _MapItemState,
    retry_count: int,
    input_id: Union[str, None],
    input_jwt: Union[InputJwtData, None],
    input_args: bytes,
):
    assert ctx
    assert ctx.state == state
    assert ctx.input == api_pb2.FunctionInput(args=input_args)
    assert ctx.retry_manager.retry_count == retry_count
    if input_id:
        # We call result rather than await because we want to test that the result has been set already.
        assert ctx.input_id.result() == input_id
    else:
        assert not ctx.input_id.done()
    if input_jwt:
        assert InputJwtData.from_jwt(ctx.input_jwt.result()) == input_jwt
    else:
        assert not ctx.input_jwt.done()


def assert_retry_item_is(
    retry_item: api_pb2.FunctionRetryInputsItem, input_jwt: InputJwtData, retry_count: int, input_args: bytes
):
    assert InputJwtData.from_jwt(retry_item.input_jwt) == input_jwt
    assert retry_item.retry_count == retry_count
    assert retry_item.input == api_pb2.FunctionInput(args=input_args)
