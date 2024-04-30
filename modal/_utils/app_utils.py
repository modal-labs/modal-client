# Copyright Modal Labs 2022
import re

from ..exception import InvalidError

# https://www.rfc-editor.org/rfc/rfc1035
subdomain_regex = re.compile("^(?![0-9]+$)(?!-)[a-z0-9-]{,63}(?<!-)$")


def is_valid_subdomain_label(label: str) -> bool:
    return subdomain_regex.match(label) is not None


def replace_invalid_subdomain_chars(label: str) -> str:
    return re.sub("[^a-z0-9-]", "-", label.lower())


def is_valid_app_name(name: str) -> bool:
    # Note: uses archaic "app name" terminology as it is also used in the server
    return len(name) <= 64 and re.match("^[a-zA-Z0-9-_.]+$", name) is not None


def check_object_name(name: str) -> None:
    message = (
        f"Invalid object name: '{name}'."
        " Object names may contain only alphanumeric characters, dashes, periods, and underscores,"
        " and must be shorter than 64 characters."
    )
    if not is_valid_app_name(name):
        raise InvalidError(message)
