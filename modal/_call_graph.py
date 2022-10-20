# Copyright Modal Labs 2022
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List

from modal_proto import api_pb2


class InputStatus(IntEnum):
    PENDING = 0
    SUCCESS = api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    FAILURE = api_pb2.GenericResult.GENERIC_STATUS_FAILURE
    TIMEOUT = api_pb2.GenericResult.GENERIC_STATUS_TIMEOUT

    def _missing_(cls, value):
        return cls.PENDING


@dataclass
class InputInfo:
    input_id: str
    task_id: str
    status: InputStatus
    function_name: str
    module_name: str
    children: List["InputInfo"]


def reconstruct_call_graph(ser_graph: api_pb2.FunctionGetCallGraphResponse) -> List[InputInfo]:
    function_calls_by_id: Dict[str, api_pb2.FunctionCallCallGraphInfo] = {}
    inputs_by_id: Dict[str, api_pb2.InputCallGraphInfo] = {}

    for function_call in ser_graph.function_calls:
        function_calls_by_id[function_call.function_call_id] = function_call

    for input in ser_graph.inputs:
        inputs_by_id[input.input_id] = input

    input_info_by_id: Dict[str, InputInfo] = {}
    result = []

    def _reconstruct(input_id: str) -> InputInfo:
        if input_id in input_info_by_id:
            return input_info_by_id[input_id]

        input = inputs_by_id[input_id]
        function_call = function_calls_by_id[input.function_call_id]
        input_info_by_id[input_id] = InputInfo(
            input_id,
            input.task_id,
            InputStatus(input.status),
            function_call.function_name,
            function_call.module_name,
            [],
        )

        if function_call.parent_input_id:
            # Find parent and append to list of children.
            parent = _reconstruct(function_call.parent_input_id)
            parent.children.append(input_info_by_id[input_id])
        else:
            # Top-level input.
            result.append(input_info_by_id[input_id])

        return input_info_by_id[input_id]

    for input_id in inputs_by_id.keys():
        _reconstruct(input_id)

    return result
