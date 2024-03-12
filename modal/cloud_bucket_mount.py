# Copyright Modal Labs 2022
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union

from modal_proto import api_pb2

from ._utils.async_utils import synchronize_api
from .secret import _Secret


class BucketType(Enum):
    S3 = "s3"

    @property
    def proto(self):
        if self.value == "s3":
            return api_pb2.CloudBucketMount.BucketType.S3


@dataclass
class _CloudBucketMount:
    """Mounts a cloud bucket to your container. Currently supports AWS S3 buckets.

    S3 buckets are mounted using [AWS' S3 Mountpoint](https://github.com/awslabs/mountpoint-s3).
    S3 mounts are optimized for reading large files sequentially. It does not support every file operation; consult
    [the AWS S3 Mountpoin documentation](https://github.com/awslabs/mountpoint-s3/blob/main/doc/SEMANTICS.md) for more information.

    **Usage**

    ```python
    import modal
    import subprocess

    stub = modal.Stub()

    @stub.function(
        volumes={
            "/my-mount": modal.CloudBucketMount("s3-bucket-name", secret=modal.Secret.from_dict({
                "AWS_ACCESS_KEY_ID": "...",
                "AWS_SECRET_ACCESS_KEY": "...",
            }), read_only=True)
        }
    )
    def f():
        subprocess.run(["ls", "/my-mount"])
    ```
    """

    bucket_name: str

    # Credentials used to access a cloud bucket. When
    # The given secret can contain AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.
    # AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY can be omitted if the bucket is publicly accessible.
    secret: Optional[_Secret] = None

    read_only: bool = False
    bucket_type: Union[
        BucketType, str
    ] = BucketType.S3.value  # S3 is the default bucket type until other cloud buckets are supported


def cloud_bucket_mounts_to_proto(mounts: List[Tuple[str, _CloudBucketMount]]) -> List[api_pb2.CloudBucketMount]:
    """Helper function to convert `CloudBucketMount` to a list of protobufs that can be passed to the server."""
    cloud_bucket_mounts: List[api_pb2.CloudBucketMount] = []

    for path, mount in mounts:
        if isinstance(mount.bucket_type, str):
            bucket_type = BucketType(mount.bucket_type)
        else:
            bucket_type = mount.bucket_type

        cloud_bucket_mount = api_pb2.CloudBucketMount(
            bucket_name=mount.bucket_name,
            mount_path=path,
            credentials_secret_id=mount.secret.object_id if mount.secret else "",
            read_only=mount.read_only,
            bucket_type=bucket_type.proto,
        )
        cloud_bucket_mounts.append(cloud_bucket_mount)

    return cloud_bucket_mounts


CloudBucketMount = synchronize_api(_CloudBucketMount)
