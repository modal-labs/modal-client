# Copyright Modal Labs 2022
import re

from ..exception import InvalidError, deprecation_warning

# https://www.rfc-editor.org/rfc/rfc1035
subdomain_regex = re.compile("^(?![0-9]+$)(?!-)[a-z0-9-]{,63}(?<!-)$")


def is_valid_subdomain_label(label: str) -> bool:
    return subdomain_regex.match(label) is not None


def replace_invalid_subdomain_chars(label: str) -> str:
    return re.sub("[^a-z0-9-]", "-", label.lower())


def is_valid_object_name(name: str) -> bool:
    return len(name) <= 64 and re.match("^[a-zA-Z0-9-_.]+$", name) is not None


def is_valid_environment_name(name: str) -> bool:
    # first char is alnum, the rest allows other chars
    return len(name) <= 64 and re.match(r"^[a-zA-Z0-9][a-zA-Z0-9-_.]+$", name) is not None


def is_valid_tag(tag: str) -> bool:
    """Tags are alphanumeric, dashes, periods, and underscores, and must be 50 characters or less"""
    pattern = r"^[a-zA-Z0-9._-]{1,50}$"
    return bool(re.match(pattern, tag))


def check_object_name(name: str, object_type: str, warn: bool = False) -> None:
    message = (
        f"Invalid {object_type} name: '{name}'."
        "\n\nNames may contain only alphanumeric characters, dashes, periods, and underscores,"
        " and must be shorter than 64 characters."
    )
    if warn:
        message += "\n\nThis will become an error in the future. Please rename your object to preserve access to it."
    if not is_valid_object_name(name):
        if warn:
            deprecation_warning((2024, 4, 30), message, show_source=False)
        else:
            raise InvalidError(message)


def check_environment_name(name: str, warn: bool = False) -> None:
    message = (
        f"Invalid environment name: '{name}'."
        "\n\nEnvironment names can only start with alphanumeric characters,"
        " may contain only alphanumeric characters, dashes, periods, and underscores,"
        " and must be shorter than 64 characters."
    )
    if warn:
        message += "\n\nThis will become an error in the future. Please rename your object to preserve access to it."
    if not is_valid_environment_name(name):
        if warn:
            deprecation_warning((2024, 4, 30), message, show_source=False)
        else:
            raise InvalidError(message)


is_valid_app_name = is_valid_object_name  # TODO becaue we use the former in the server
