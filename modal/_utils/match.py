# Copyright Modal Labs 2024
"""Pattern matching library ported from https://github.com/golang/go/blob/go1.23.4/src/path/match.go

This is the same pattern-matching logic used by Docker for Dockerfiles (not dockerignore),
except it is written in Python rather than Go.
"""


class PatternError(Exception):
    """Indicates a pattern was malformed."""

    pass


def match(pattern: str, name: str) -> bool:
    while len(pattern) > 0:
        continue_outer_loop = False
        star, chunk, pattern = scan_chunk(pattern)

        if star and chunk == "":
            # Trailing * matches rest of string unless it has a /.
            # return bytealg.IndexByteString(name, '/') < 0, nil
            return name.find("/") < 0

        # Look for match at current position.
        t, ok = match_chunk(chunk, name)
        # if we're the last chunk, make sure we've exhausted the name
        # otherwise we'll give a false result even if we could still match
        # using the star
        if ok and (len(t) == 0 or len(pattern) > 0):
            name = t
            continue

        if star:
            i = 0
            while i < len(name) and name[i] != "/":
                t, ok = match_chunk(chunk, name[i + 1 :])
                if ok:
                    if len(pattern) == 0 and len(t) > 0:
                        i += 1
                        continue
                    name = t
                    continue_outer_loop = True
                    break

                i += 1
            if continue_outer_loop:
                continue

        while len(pattern) > 0:
            _, chunk, pattern = scan_chunk(pattern)
            match_chunk(chunk, "")

        return False

    return len(name) == 0


def scan_chunk(pattern: str) -> tuple[bool, str, str]:
    star = False
    while len(pattern) > 0 and pattern[0] == "*":
        pattern = pattern[1:]
        star = True

    inrange = False
    i = 0
    while i < len(pattern):
        if pattern[i] == "\\":
            if i + 1 < len(pattern):
                i += 1
        elif pattern[i] == "[":
            inrange = True
        elif pattern[i] == "]":
            inrange = False
        elif pattern[i] == "*":
            if not inrange:
                break
        i += 1

    return star, pattern[0:i], pattern[i:]


def match_chunk(chunk: str, s: str) -> tuple[str, bool]:
    failed = False

    while len(chunk) > 0:
        if not failed and len(s) == 0:
            failed = True

        if chunk[0] == "[":
            r = ""
            if not failed:
                r = s[0]
                s = s[1:]
            chunk = chunk[1:]
            negated = False
            if len(chunk) > 0 and chunk[0] == "^":
                negated = True
                chunk = chunk[1:]

            match = False
            nrange = 0

            while True:
                if len(chunk) > 0 and chunk[0] == "]" and nrange > 0:
                    chunk = chunk[1:]
                    break
                lo, chunk = get_esc(chunk)
                hi = lo

                if chunk[0] == "-":
                    hi, chunk = get_esc(chunk[1:])
                if lo <= r and r <= hi:
                    match = True
                nrange += 1

            if match == negated:
                failed = True
        elif chunk[0] == "?":
            if not failed:
                if s[0] == "/":
                    failed = True
                s = s[1:]
            chunk = chunk[1:]
        elif chunk[0] == "\\":
            chunk = chunk[1:]
            if len(chunk) == 0:
                raise PatternError("Bad pattern")

            if not failed:
                if chunk[0] != s[0]:
                    failed = True
                s = s[1:]
            chunk = chunk[1:]
        else:
            if not failed:
                if chunk[0] != s[0]:
                    failed = True
                s = s[1:]
            chunk = chunk[1:]

    if failed:
        return "", False
    return s, True


def get_esc(chunk: str) -> tuple[str, str]:
    if len(chunk) == 0 or chunk[0] == "-" or chunk[0] == "]":
        raise PatternError("Bad pattern")
    if chunk[0] == "\\":
        chunk = chunk[1:]
        if len(chunk) == 0:
            raise PatternError("Bad pattern")
    try:
        r = chunk[0]
        nchunk = chunk[1:]
    except IndexError:
        raise PatternError("Invalid pattern")
    if len(nchunk) == 0:
        raise PatternError("Bad pattern")
    return r, nchunk
