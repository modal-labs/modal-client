# Copyright Modal Labs 2022
import sys

from ._traceback import reduce_traceback_to_user_code
from .cli._traceback import highlight_modal_warnings, setup_rich_traceback
from .cli.entry_point import entrypoint_cli
from .cli.import_refs import _CliUserExecutionError
from .config import config
from .output import OutputManager, enable_output


def main():
    # Setup rich tracebacks, but only on user's end, when using the Modal CLI.
    setup_rich_traceback()
    highlight_modal_warnings()

    # Enable rich output for the entire CLI session
    with enable_output():
        try:
            entrypoint_cli()

        except _CliUserExecutionError as exc:
            if config.get("traceback"):
                raise

            assert exc.__cause__  # We should always raise this class from another error
            tb = reduce_traceback_to_user_code(exc.__cause__.__traceback__, exc.user_source)
            sys.excepthook(type(exc.__cause__), exc.__cause__, tb)
            sys.exit(1)

        except Exception as exc:
            if (
                # User has asked to alway see full tracebacks
                config.get("traceback")
                # The exception message is empty, so we need to provide _some_ actionable information
                or not str(exc)
            ):
                raise

            content = str(exc)
            if notes := getattr(exc, "__notes__", []):
                content = f"{content}\n\nNote: {' '.join(notes)}"

            OutputManager.get().print_error(content)
            sys.exit(1)


if __name__ == "__main__":
    main()
