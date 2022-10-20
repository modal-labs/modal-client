# Copyright Modal Labs 2022
import re

# https://www.rfc-editor.org/rfc/rfc1035
subdomain_regex = re.compile("^(?![0-9]+$)(?!-)[a-z0-9-]{,63}(?<!-)$")


def is_valid_subdomain_label(label: str):
    return subdomain_regex.match(label) is not None


def replace_invalid_subdomain_chars(label: str):
    return re.sub("[^a-z0-9-]", "-", label.lower())


def is_valid_app_name(name: str):
    return len(name) <= 64 and re.match("^[a-zA-Z0-9-_.]+$", name) is not None
