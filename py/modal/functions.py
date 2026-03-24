# Copyright Modal Labs 2025
from ._functions import _Function, _FunctionCall
from ._utils.async_utils import synchronize_api

Function = synchronize_api(_Function, target_module=__name__)
FunctionCall = synchronize_api(_FunctionCall, target_module=__name__)
