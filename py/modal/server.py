# Copyright Modal Labs 2026
from ._server import _Server, _ServerAutoscalerManager
from ._utils.async_utils import synchronize_api

Server = synchronize_api(_Server, target_module=__name__)
ServerAutoscalerManager = synchronize_api(_ServerAutoscalerManager, target_module=__name__)
