# Copyright Modal Labs 2022
from ._traceback import setup_rich_traceback
from .cli.entry_point import entrypoint_cli


def main():
    # Setup rich tracebacks, but only on user's end, when using the Modal CLI.
    setup_rich_traceback()
    entrypoint_cli()


if __name__ == "__main__":
    main()
