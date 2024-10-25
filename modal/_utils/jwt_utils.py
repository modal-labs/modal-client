# TODO(ryan): Use acutal JWT rather than regular string
from typing import Tuple


def encode_input_jwt(idx: int, input_id: str) -> str:
    return f"{idx}:{input_id}"


def decode_input_jwt(input_jwt: str) -> Tuple[int, str]:
    """
    Returns idx, input_id
    """
    parts = input_jwt.split(":")
    assert len(parts) == 2
    return int(parts[0]), parts[1]


def encode_function_call_jwt(function_id: str, function_call_id: str) -> str:
    return f"{function_id}:{function_call_id}"


def decode_function_call_jwt(function_call_jwt: str) -> Tuple[str, str]:
    """
    Returns function_id, function_call_id
    """
    parts = function_call_jwt.split(":")
    assert len(parts) == 2
    return parts[0], parts[1]
