# Copyright Modal Labs 2025
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modal_proto import api_pb2


class StatusRow:
    """A row in the object creation status tree.

    This class is designed to avoid importing rich when progress is None,
    which is important for environments where rich is not installed or
    when output is disabled.
    """

    def __init__(self, progress: Any):
        # progress is a rich.tree.Tree or None
        self._spinner: Any = None
        self._step_node: Any = None
        if progress is not None:
            # Only import rich-dependent code when we actually have a progress tree
            from ._output import OutputManager

            self._spinner = OutputManager.step_progress()
            self._step_node = progress.add(self._spinner)

    def message(self, message: str) -> None:
        if self._spinner is not None:
            self._spinner.update(text=message)

    def warning(self, warning: "api_pb2.Warning") -> None:
        if self._step_node is not None:
            self._step_node.add(f"[yellow]:warning:[/yellow] {warning.message}")

    def finish(self, message: str) -> None:
        if self._step_node is not None and self._spinner is not None:
            from ._output import OutputManager

            self._spinner.update(text=message)
            self._step_node.label = OutputManager.substep_completed(message)
