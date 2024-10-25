# Copyright Modal Labs 2022
import re

from ..exception import InvalidError

# https://www.rfc-editor.org/rfc/rfc1035
subdomain_regex = re.compile("^(?![0-9]+$)(?!-)[a-z0-9-]{,63}(?<!-)$")


def is_valid_subdomain_label(label: str) -> bool:
    return subdomain_regex.match(label) is not None


def replace_invalid_subdomain_chars(label: str) -> str:
    return re.sub("[^a-z0-9-]", "-", label.lower())


def is_valid_object_name(name: str) -> bool:
    return (
        # Limit object name length
        len(name) <= 64
        # Limit character set
        and re.match("^[a-zA-Z0-9-_.]+$", name) is not None
        # Avoid collisions with App IDs
        and re.match("^ap-[a-zA-Z0-9]{22}$", name) is None
    )


def is_valid_environment_name(name: str) -> bool:
    # first char is alnum, the rest allows other chars
    return len(name) <= 64 and re.match(r"^[a-zA-Z0-9][a-zA-Z0-9-_.]+$", name) is not None


def is_valid_tag(tag: str) -> bool:
    """Tags are alphanumeric, dashes, periods, and underscores, and must be 50 characters or less"""
    pattern = r"^[a-zA-Z0-9._-]{1,50}$"
    return bool(re.match(pattern, tag))


def check_object_name(name: str, object_type: str) -> None:
    message = (
        f"Invalid {object_type} name: '{name}'."
        "\n\nNames may contain only alphanumeric characters, dashes, periods, and underscores,"
        " must be shorter than 64 characters, and cannot conflict with App ID strings."
    )
    if not is_valid_object_name(name):
        raise InvalidError(message)


def check_environment_name(name: str) -> None:
    message = (
        f"Invalid environment name: '{name}'."
        "\n\nEnvironment names can only start with alphanumeric characters,"
        " may contain only alphanumeric characters, dashes, periods, and underscores,"
        " and must be shorter than 64 characters."
    )
    if not is_valid_environment_name(name):
        raise InvalidError(message)
