# Copyright Modal Labs 2022
import sys

from ._traceback import highlight_modal_deprecation_warnings, setup_rich_traceback
from .cli.entry_point import entrypoint_cli
from .cli.import_refs import CliUserExecutionError
from .config import config


def main():
    # Setup rich tracebacks, but only on user's end, when using the Modal CLI.
    setup_rich_traceback()
    highlight_modal_deprecation_warnings()

    try:
        entrypoint_cli()
    except CliUserExecutionError as exc:
        raise exc.__cause__ from None
    except Exception as exc:
        if config.get("traceback"):
            raise
        else:
            from rich.console import Console
            from rich.panel import Panel

            console = Console(stderr=True)
            panel = Panel(str(exc), border_style="red", title="Error", title_align="left")
            console.print(panel, highlight=False)
            sys.exit(1)


if __name__ == "__main__":
    main()
