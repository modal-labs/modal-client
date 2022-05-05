import platform

from rich.console import RenderableType
from rich.spinner import Spinner
from rich.text import Text

if platform.system() == "Windows":
    default_spinner = "line"
else:
    default_spinner = "dots"


def step_progress(message: str) -> RenderableType:
    """Returns the element to be rendered when a step is in progress."""
    return Spinner(default_spinner, Text(message, style="blue"), style="default")


def step_completed(message: str, is_substep: bool = False) -> RenderableType:
    """Returns the element to be rendered when a step is completed."""

    STEP_COMPLETED = "✓"
    SUBSTEP_COMPLETED = "🔨"

    symbol = SUBSTEP_COMPLETED if is_substep else STEP_COMPLETED
    return f"[green]{symbol}[/green] " + message
