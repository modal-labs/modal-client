from typing import Callable, Dict

try:
    import curses

    curses.setupterm()
    _using_terminfo = True
except Exception:
    _using_terminfo = False

FALLBACK_SEQUENCES: Dict[str, Callable[..., bytes]] = {
    "cuu": lambda i: f"\r\033[{i}A".encode("utf-8"),
    "cr": lambda: b"\r",
    "ed": lambda: b"\033[J",
    "el": lambda: b"\033[K",
    "civis": lambda: b"\033[?25l",
    "cvvis": lambda: b"\033[?25h",
}


def fallback_term_seq(capname, *args):
    if capname not in FALLBACK_SEQUENCES:
        raise Exception(f"Could not find capability {capname}")

    return FALLBACK_SEQUENCES[capname](*args)


def term_seq(capname, *args):
    """See manpage for terminfo or tput for a list of capability names.

    https://www.gnu.org/software/termutils/manual/termutils-2.0/html_chapter/tput_1.html
    """

    if _using_terminfo:
        capstr = curses.tigetstr(capname)
        if capstr is None:
            return fallback_term_seq(capname, *args)
        else:
            return curses.tparm(capstr, *args)
    else:
        return fallback_term_seq(capname, *args)


def term_seq_str(capname, *args):
    return term_seq(capname, *args).decode("utf-8")
