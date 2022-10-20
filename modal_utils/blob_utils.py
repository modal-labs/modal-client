# Copyright Modal Labs 2022
from urllib.parse import urlparse


def use_md5(url: str) -> bool:
    """This takes an upload URL in S3 and returns whether we should attach a checksum.

    It's only a workaround for missing functionality in moto.
    https://github.com/spulec/moto/issues/816
    """
    host = urlparse(url).netloc.split(":")[0]
    if host.endswith(".amazonaws.com"):
        return True
    elif host in ["127.0.0.1", "localhost", "172.19.0.1"]:
        return False
    else:
        raise Exception(f"Unknown S3 host: {host}")
