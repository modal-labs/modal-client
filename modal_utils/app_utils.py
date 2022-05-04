import re
import urllib.parse

# https://www.rfc-editor.org/rfc/rfc1035
subdomain_regex = re.compile("^(?![0-9]+$)(?!-)[a-z0-9-]{,63}(?<!-)$")


def is_valid_subdomain_label(label: str):
    return subdomain_regex.match(label) is not None


def is_valid_deployment_name(name):
    # Basically make sure we can put them in an URL
    return name and "/" not in name and name == urllib.parse.quote(name)
