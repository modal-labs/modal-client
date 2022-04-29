import urllib.parse


def is_valid_deployment_name(name):
    # Basically make sure we can put them in an URL
    return "/" not in name and name == urllib.parse.quote(name)
