# Copyright Modal Labs 2024
from typing import Optional

from modal_proto import api_pb2

from ._resolver import Resolver
from ._utils.async_utils import synchronize_api
from .object import _get_environment_name, _Object


class _VproxLink(_Object, type_prefix="pr"):
    """
    `modal.VproxLink` creates a secure connection mechanism between Modal
    containers and exit nodes with static IPs. This allows for you to allow list
    static IPs in your firewall. This is useful if you need Modal Functions to
    access a database, for example.

    `modal.VproxLink` are experimental.

    Currently `modal.VproxLink` objects must be setup with the assistance of
    Modal staff. If you require a vprox link please contact us.
    """

    @staticmethod
    def from_name(
        label: str,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        environment_name: Optional[str] = None,
    ) -> "_VproxLink":
        async def _load(self: _VproxLink, resolver: Resolver, existing_object_id: Optional[str]):
            req = api_pb2.VproxLinkGetOrCreateRequest(
                deployment_name=label,
                namespace=namespace,
                environment_name=_get_environment_name(environment_name, resolver),
            )
            response = await resolver.client.stub.VproxLinkGetOrCreateRequest(req)
            self._hydrate(response.proxy_id, resolver.client, None)

        return _VproxLink._from_loader(_load, "VproxLink()", is_another_app=True)


VproxLink = synchronize_api(_VproxLink, target_module=__name__)
