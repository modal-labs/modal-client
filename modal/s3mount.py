# Copyright Modal Labs 2022
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_api

from .secret import _Secret


@dataclass
class _S3Mount:
    bucket_name: str

    # Credentials used to access the S3 bucket.
    # The given secret should contain AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.
    # This does not need to be given if the S3 bucket is publicly accessible.
    credentials: Optional[_Secret] = None


def s3mounts_to_proto(mounts: Dict[Union[str, os.PathLike], _S3Mount]) -> List[api_pb2.S3Mount]:
    """
    Helper function to convert S3 mounts passed in as a Volume, {"/mount": modal.S3Mount(...)},
    to a list of protobufs that can be passed to the server.
    """
    return [
        api_pb2.S3Mount(
            bucket_name=mount.bucket_name,
            mount_path=path,
            credentials_secret_id=mount.credentials.object_id if mount.credentials else "",
        )
        for path, mount in mounts.items()
    ]


S3Mount = synchronize_api(_S3Mount)
