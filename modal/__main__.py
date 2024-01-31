# Copyright Modal Labs 2022
import os
import sys
from textwrap import dedent

from rich.console import Console
from rich.panel import Panel

from ._traceback import setup_rich_traceback
from .cli.entry_point import entrypoint_cli
from .config import _profile, _user_config
from .exception import InvalidError


def check_profile():
    if ("MODAL_PROFILE" in os.environ) or (len(sys.argv) == 1) or (sys.argv[1] in {"setup", "profile"}):
        return

    num_profiles = len(_user_config)
    num_active = sum(v.get("active", False) for v in _user_config.values())
    if num_active > 1:
        raise InvalidError(
            "More than one Modal profile is active. "
            "Please fix with `modal profile activate` or by editing your Modal config file."
        )
    elif num_profiles > 1 and num_active == 0 and _profile == "default":
        # Eventually we plan to have num_profiles > 1 with num_active = 0 be an error
        # But we want to give users time to activate one of their profiles without disruption
        console = Console()
        message = dedent(
            """\
            Support for using an implicit 'default' profile is deprecated.
            Please use `modal profile activate` to activate one of your profiles.
            (Use `modal profile list` to see the options.)

            This will become an error in a future update."""
        )
        console.print(Panel(message, style="yellow", title="Warning", title_align="left"))


def main():
    # Setup rich tracebacks, but only on user's end, when using the Modal CLI.
    setup_rich_traceback()
    check_profile()
    entrypoint_cli()


if __name__ == "__main__":
    main()
