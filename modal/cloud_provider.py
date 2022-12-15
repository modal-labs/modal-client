# Copyright Modal Labs 2022
from enum import Enum

from modal.exception import InvalidError
from modal_proto import api_pb2


class CloudProvider(Enum):
    AWS = api_pb2.CLOUD_PROVIDER_AWS
    GCP = api_pb2.CLOUD_PROVIDER_GCP
    AUTO = api_pb2.CLOUD_PROVIDER_AUTO

    def to_proto(self) -> "api_pb2.CloudProvider.V":
        return self.value


def parse_cloud_provider(value: str) -> CloudProvider:
    try:
        return CloudProvider[value.upper()]
    except KeyError:
        raise InvalidError(
            f"Invalid cloud provider: {value}. Value must be one of {[x.name.lower() for x in CloudProvider]} (case-insensitive)."
        )
