# Copyright Modal Labs 2022
from dataclasses import dataclass
from typing import List, Dict, Union
import os

from modal_proto import api_pb2
from .secret import _Secret
from modal_utils.async_utils import synchronize_api

@dataclass
class _S3Mount:
    bucket_name: str
    credentials: _Secret

def s3mounts_to_proto(mounts: Dict[Union[str, os.PathLike], _S3Mount]) -> List[api_pb2.S3Mount]:
    """
    Helper function to convert S3 mounts passed in as a Volume, {"/mount": modal.S3Mount(...)},
    to a list of protobufs that can be passed to the server.
    """
    print(mounts["/mountpoint"].credentials.object_id)
    return [
        api_pb2.S3Mount(
            bucket_name=mount.bucket_name,
            mount_path=path,
            credentials_secret_id=mount.credentials.object_id
        ) for path, mount in mounts.items()
    ]

S3Mount = synchronize_api(_S3Mount)
