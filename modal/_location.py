# Copyright Modal Labs 2022
from enum import Enum

import modal_proto.api_pb2

from .exception import InvalidError


class CloudProvider(Enum):
    AWS = modal_proto.api_pb2.CLOUD_PROVIDER_AWS
    GCP = modal_proto.api_pb2.CLOUD_PROVIDER_GCP
    AUTO = modal_proto.api_pb2.CLOUD_PROVIDER_AUTO
    OCI = modal_proto.api_pb2.CLOUD_PROVIDER_OCI


def parse_cloud_provider(value: str) -> "modal_proto.api_pb2.CloudProvider.V":
    try:
        cloud_provider = CloudProvider[value.upper()]
    except KeyError:
        # provider's int identifier may be directly specified
        try:
            return int(value)  # type: ignore
        except ValueError:
            pass

        raise InvalidError(
            f"Invalid cloud provider: {value}. "
            f"Value must be one of {[x.name.lower() for x in CloudProvider]} (case-insensitive)."
        )

    return cloud_provider.value


def display_location(cloud_provider: "modal_proto.api_pb2.CloudProvider.V") -> str:
    if cloud_provider == modal_proto.api_pb2.CLOUD_PROVIDER_GCP:
        return "GCP (us-east1)"
    elif cloud_provider == modal_proto.api_pb2.CLOUD_PROVIDER_AWS:
        return "AWS (us-east-1)"
    else:
        return ""
