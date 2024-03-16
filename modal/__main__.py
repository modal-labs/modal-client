# Copyright Modal Labs 2022
import sys

from ._traceback import highlight_modal_deprecation_warnings, reduce_traceback_to_user_code, setup_rich_traceback
from .cli.entry_point import entrypoint_cli
from .cli.import_refs import _CliUserExecutionError
from .config import config


def main():
    # Setup rich tracebacks, but only on user's end, when using the Modal CLI.
    setup_rich_traceback()
    highlight_modal_deprecation_warnings()

    try:
        entrypoint_cli()

    except _CliUserExecutionError as exc:
        if config.get("traceback"):
            raise

        tb = reduce_traceback_to_user_code(exc.__cause__.__traceback__, exc.user_source)
        sys.excepthook(type(exc.__cause__), exc.__cause__, tb)
        sys.exit(1)

    except Exception as exc:
        if config.get("traceback"):
            raise

        from rich.console import Console
        from rich.panel import Panel

        console = Console(stderr=True)
        panel = Panel(str(exc), border_style="red", title="Error", title_align="left")
        console.print(panel, highlight=False)
        sys.exit(1)


if __name__ == "__main__":
    main()
