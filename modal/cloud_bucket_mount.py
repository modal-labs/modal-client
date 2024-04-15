# Copyright Modal Labs 2022
from dataclasses import dataclass
from typing import List, Optional, Tuple

from modal_proto import api_pb2

from ._utils.async_utils import synchronize_api
from .secret import _Secret


@dataclass
class _CloudBucketMount:
    """Mounts a cloud bucket to your container. Currently supports AWS S3 buckets.

    S3 buckets are mounted using [AWS S3 Mountpoint](https://github.com/awslabs/mountpoint-s3).
    S3 mounts are optimized for reading large files sequentially. It does not support every file operation; consult
    [the AWS S3 Mountpoint documentation](https://github.com/awslabs/mountpoint-s3/blob/main/doc/SEMANTICS.md) for more information.

    **AWS S3 Usage**

    ```python
    import subprocess

    stub = modal.Stub()
    secret = modal.Secret.from_dict({
        "AWS_ACCESS_KEY_ID": "...",
        "AWS_SECRET_ACCESS_KEY": "...",
    })
    @stub.function(
        volumes={
            "/my-mount": modal.CloudBucketMount(
                bucket_name="s3-bucket-name",
                secret=secret,
                read_only=True
            )
        }
    )
    def f():
        subprocess.run(["ls", "/my-mount"], check=True)
    ```

    **Cloudflare R2 Usage**

    Cloudflare R2 is [S3-compatible](https://developers.cloudflare.com/r2/api/s3/api/) so its setup looks very similar to S3.
    But additionally the `bucket_endpoint_url` argument must be passed.

    ```python
    import subprocess

    stub = modal.Stub()
    secret = modal.Secret.from_dict({
        "AWS_ACCESS_KEY_ID": "...",
        "AWS_SECRET_ACCESS_KEY": "...",
    })
    @stub.function(
        volumes={
            "/my-mount": modal.CloudBucketMount(
                bucket_name="my-r2-bucket",
                bucket_endpoint_url="https://<ACCOUNT ID>.r2.cloudflarestorage.com",
                secret=secret,
                read_only=True
            )
        }
    )
    def f():
        subprocess.run(["ls", "/my-mount"], check=True)
    ```
    """

    bucket_name: str
    # Endpoint URL is used to support Cloudflare R2.
    bucket_endpoint_url: Optional[str] = None

    # Credentials used to access a cloud bucket.
    # If the bucket is private, the secret **must** contain AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.
    # If the bucket is publicly accessible, the secret is unnecessary and can be omitted.
    secret: Optional[_Secret] = None

    read_only: bool = False
    requester_pays: bool = False


def cloud_bucket_mounts_to_proto(mounts: List[Tuple[str, _CloudBucketMount]]) -> List[api_pb2.CloudBucketMount]:
    """Helper function to convert `CloudBucketMount` to a list of protobufs that can be passed to the server."""
    cloud_bucket_mounts: List[api_pb2.CloudBucketMount] = []

    for path, mount in mounts:
        # TODO: in future this relationship between endpoint URL and type will not hold true.
        if mount.bucket_endpoint_url:
            bucket_type = api_pb2.CloudBucketMount.BucketType.R2
        else:
            bucket_type = api_pb2.CloudBucketMount.BucketType.S3

        if mount.requester_pays and not mount.secret:
            raise ValueError("Credentials required in order to use Requester Pays.")

        cloud_bucket_mount = api_pb2.CloudBucketMount(
            bucket_name=mount.bucket_name,
            bucket_endpoint_url=mount.bucket_endpoint_url,
            mount_path=path,
            credentials_secret_id=mount.secret.object_id if mount.secret else "",
            read_only=mount.read_only,
            bucket_type=bucket_type,
            requester_pays=mount.requester_pays,
        )
        cloud_bucket_mounts.append(cloud_bucket_mount)

    return cloud_bucket_mounts


CloudBucketMount = synchronize_api(_CloudBucketMount)
