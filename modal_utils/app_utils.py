import re

# https://www.rfc-editor.org/rfc/rfc1035
subdomain_regex = re.compile("^(?![0-9]+$)(?!-)[a-z0-9-]{,63}(?<!-)$")


def is_valid_subdomain_label(label: str):
    return subdomain_regex.match(label) is not None


def replace_invalid_subdomain_chars(label: str):
    return re.sub("[^a-z0-9-]", "-", label.lower())
