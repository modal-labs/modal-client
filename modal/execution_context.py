# Copyright Modal Labs 2024
from contextvars import ContextVar
from typing import Callable, List, Optional

from modal._container_io_manager import _ContainerIOManager
from modal._utils.async_utils import synchronize_api
from modal.exception import InvalidError


def is_local() -> bool:
    """Returns if we are currently on the machine launching/deploying a Modal app

    Returns `True` when executed locally on the user's machine.
    Returns `False` when executed from a Modal container in the cloud.
    """
    return not _ContainerIOManager._singleton


async def _interact() -> None:
    """Enable interactivity with user input inside a Modal container.

    See the [interactivity guide](https://modal.com/docs/guide/developing-debugging#interactivity)
    for more information on how to use this function.
    """
    container_io_manager = _ContainerIOManager._singleton
    if not container_io_manager:
        raise InvalidError("Interactivity only works inside a Modal container.")
    else:
        await container_io_manager.interact()


interact = synchronize_api(_interact)


def current_input_id() -> Optional[str]:
    """Returns the input ID for the current input.

    Can only be called from Modal function (i.e. in a container context).

    ```python
    from modal import current_input_id

    @app.function()
    def process_stuff():
        print(f"Starting to process {current_input_id()}")
    ```
    """
    try:
        return _current_input_id.get()
    except LookupError:
        return None


def current_function_call_id() -> Optional[str]:
    """Returns the function call ID for the current input.

    Can only be called from Modal function (i.e. in a container context).

    ```python
    from modal import current_function_call_id

    @app.function()
    def process_stuff():
        print(f"Starting to process input from {current_function_call_id()}")
    ```
    """
    try:
        return _current_function_call_id.get()
    except LookupError:
        return None


def _set_current_context_ids(input_ids: List[str], function_call_ids: List[str]) -> Callable[[], None]:
    assert len(input_ids) == len(function_call_ids) and len(input_ids) > 0
    input_id = input_ids[0]
    function_call_id = function_call_ids[0]
    input_token = _current_input_id.set(input_id)
    function_call_token = _current_function_call_id.set(function_call_id)

    def _reset_current_context_ids():
        _current_input_id.reset(input_token)
        _current_function_call_id.reset(function_call_token)

    return _reset_current_context_ids


_current_input_id: ContextVar = ContextVar("_current_input_id")
_current_function_call_id: ContextVar = ContextVar("_current_function_call_id")
