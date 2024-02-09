# Copyright Modal Labs 2022
from dataclasses import dataclass
from typing import List, Optional, Tuple

from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_api

from .secret import _Secret


@dataclass
class _S3Mount:
    """Mounts an S3 bucket to your container using AWS S3 Mountpoint.

    S3 mounts are optimized for reading large files sequentially. It does not support every file operation; consult
    [the AWS S3 Mountpoint documentation](https://github.com/awslabs/mountpoint-s3/blob/main/doc/SEMANTICS.md) for more information.

    **Usage**

    ```python
    import modal
    import os

    stub = modal.Stub()

    @stub.function(
        volumes={
            "/my-mount": modal.S3Mount("s3-bucket-name", secret=modal.Secret.from_dict({
                "AWS_ACCESS_KEY_ID": "...",
                "AWS_SECRET_ACCESS_KEY": "...",
            }), read_only=True)
        }
    )
    def f():
        os.system("ls /my-mount")
    ```
    """

    bucket_name: str

    # Credentials used to access the S3 bucket.
    # The given secret can contain AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.
    # AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY can be omitted if the bucket is publicly accessible.
    secret: Optional[_Secret] = None

    read_only: bool = False


def s3_mounts_to_proto(mounts: List[Tuple[str, _S3Mount]]) -> List[api_pb2.S3Mount]:
    """
    Helper function to convert S3 mounts to a list of protobufs that can be passed to the server.
    """
    return [
        api_pb2.S3Mount(
            bucket_name=mount.bucket_name,
            mount_path=path,
            credentials_secret_id=mount.secret.object_id if mount.secret else "",
            read_only=mount.read_only,
        )
        for path, mount in mounts
    ]


S3Mount = synchronize_api(_S3Mount)
