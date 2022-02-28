import curses

_initialized_terminfo = False


def term_seq(capname, *args):
    """See manpage for terminfo or tput for a list of capability names.

    https://www.gnu.org/software/termutils/manual/termutils-2.0/html_chapter/tput_1.html
    """
    global _initialized_terminfo
    if not _initialized_terminfo:
        curses.setupterm()
        _initialized_terminfo = True

    capstr = curses.tigetstr(capname)
    if capstr is None:
        raise Exception(f"Could not find capability {capname}")
    return curses.tparm(capstr, *args)


def term_seq_str(capname, *args):
    return term_seq(capname, *args).decode("ascii")
