# Copyright Modal Labs 2024
from typing import Optional

from modal_proto import api_pb2

from ._resolver import Resolver
from ._utils.async_utils import synchronize_api
from .object import _get_environment_name, _Object


class _Proxy(_Object, type_prefix="pr"):
    """Proxy objects give your Modal containers a static outbound IP address.

    This can be used for connecting to a remote address with network whitelist, for example
    a database. See [the guide](/docs/guide/proxy-ips) for more information.
    """

    @staticmethod
    def from_name(
        label: str,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        environment_name: Optional[str] = None,
    ) -> "_Proxy":
        async def _load(self: _Proxy, resolver: Resolver, existing_object_id: Optional[str]):
            req = api_pb2.ProxyGetOrCreateRequest(
                deployment_name=label,
                namespace=namespace,
                environment_name=_get_environment_name(environment_name, resolver),
            )
            response = await resolver.client.stub.ProxyGetOrCreate(req)
            self._hydrate(response.proxy_id, resolver.client, None)

        return _Proxy._from_loader(_load, "Proxy()", is_another_app=True)


Proxy = synchronize_api(_Proxy, target_module=__name__)
