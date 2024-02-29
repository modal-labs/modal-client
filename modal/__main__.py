# Copyright Modal Labs 2022
import sys

from ._traceback import highlight_modal_deprecation_warnings, setup_rich_traceback
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

        # Try to step forward in the traceback until we get to the user code that failed to import
        tb = orig_tb = exc.__cause__.__traceback__
        if exc.user_source.endswith(".py"):
            while tb is not None and tb.tb_frame.f_code.co_filename != exc.user_source:
                tb = tb.tb_next
        else:
            while tb is not None and tb.tb_frame.f_code.co_name != "<module>":
                tb = tb.tb_next
        if tb is None:
            # In case we didn't find a frame that matched the user source, revert to the original traceback
            tb = orig_tb
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
