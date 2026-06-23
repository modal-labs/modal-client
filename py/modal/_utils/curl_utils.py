# Copyright Modal Labs 2026
from urllib.parse import urlparse

# Short curl flags (single dash, single letter) that consume the following argument.
_CURL_SHORT_VALUE_FLAGS = frozenset("AbCcDdEeFHhKmoQrTtUuwXxYyz")

# Long curl flags that consume the following argument (when given as `--flag value`).
# Flags given as `--flag=value` are self-contained and handled generically.
_CURL_LONG_VALUE_FLAGS = frozenset(
    {
        "--cacert",
        "--capath",
        "--cert",
        "--cert-type",
        "--config",
        "--connect-timeout",
        "--connect-to",
        "--continue-at",
        "--cookie",
        "--cookie-jar",
        "--data",
        "--data-ascii",
        "--data-binary",
        "--data-raw",
        "--data-urlencode",
        "--dump-header",
        "--form",
        "--form-string",
        "--header",
        "--interface",
        "--key",
        "--key-type",
        "--limit-rate",
        "--max-filesize",
        "--max-redirs",
        "--max-time",
        "--oauth2-bearer",
        "--output",
        "--pass",
        "--proxy",
        "--proxy-user",
        "--range",
        "--referer",
        "--request",
        "--resolve",
        "--retry",
        "--retry-delay",
        "--retry-max-time",
        "--unix-socket",
        "--upload-file",
        "--url",
        "--user",
        "--user-agent",
        "--write-out",
    }
)


def endpoint_cache_host(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"https://{url}")
    return (parsed.hostname or url).lower().rstrip(".")


def is_modal_direct_url(url: str) -> bool:
    # Only fetch token for app.server() functions
    host = endpoint_cache_host(url)
    return host.endswith("modal.direct") or host.endswith("modal-dev.direct")


def _is_modal_direct_candidate(arg: str) -> bool:
    try:
        return is_modal_direct_url(arg)
    except ValueError:
        return False


def find_url(curl_args: tuple[str, ...]) -> str | None:
    """Find the Modal direct URL operand within a curl argument list.

    Parses the args the way curl does well enough to skip over flag values, so that
    a `modal.direct` host appearing as e.g. a header or data value is not mistaken
    for the request URL.
    """
    i = 0
    n = len(curl_args)
    while i < n:
        arg = curl_args[i]

        if arg.startswith("--"):
            flag, sep, value = arg.partition("=")
            if sep:
                # `--flag=value`: the value is self-contained.
                if flag == "--url" and _is_modal_direct_candidate(value):
                    return value
            elif flag in _CURL_LONG_VALUE_FLAGS:
                # `--flag value`: the next argument is this flag's value.
                value = curl_args[i + 1] if i + 1 < n else ""
                if flag == "--url" and _is_modal_direct_candidate(value):
                    return value
                i += 2
                continue
        elif arg.startswith("-") and len(arg) >= 2:
            last = arg[-1]
            if len(arg) == 2 and last in _CURL_SHORT_VALUE_FLAGS:
                # `-X value`: the next argument is this flag's value.
                i += 2
                continue
            # `-Xvalue` (attached value) or a cluster of boolean flags: nothing to skip.
        elif _is_modal_direct_candidate(arg):
            # A bare (positional) URL operand.
            return arg

        i += 1

    return None
