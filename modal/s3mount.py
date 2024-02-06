# Copyright Modal Labs 2022
from dataclasses import dataclass
from typing import List, Tuple

from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_api

from .secret import _Secret


@dataclass
class _S3Mount:
    bucket_name: str

    # Credentials used to access the S3 bucket.
    # The given secret should contain AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION.
    # AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY can be omitted if the bucket is publicly accessible.
    credentials: _Secret


def s3_mounts_to_proto(mounts: List[Tuple[str, _S3Mount]]) -> List[api_pb2.S3Mount]:
    """
    Helper function to convert S3 mounts to a list of protobufs that can be passed to the server.
    """
    return [
        api_pb2.S3Mount(
            bucket_name=mount.bucket_name,
            mount_path=path,
            credentials_secret_id=mount.credentials.object_id,
        )
        for path, mount in mounts
    ]


S3Mount = synchronize_api(_S3Mount)
