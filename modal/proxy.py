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
        name: str,
        environment_name: Optional[str] = None,
    ) -> "_Proxy":
        """Reference a Proxy by its name.

        In contrast to most other Modal objects, new Proxy objects must be
        provisioned via the Dashboard and cannot be created on the fly from code.

        """

        async def _load(self: _Proxy, resolver: Resolver, existing_object_id: Optional[str]):
            req = api_pb2.ProxyGetRequest(
                name=name,
                environment_name=_get_environment_name(environment_name, resolver),
            )
            response: api_pb2.ProxyGetResponse = await resolver.client.stub.ProxyGet(req)
            self._hydrate(response.proxy.proxy_id, resolver.client, None)

        return _Proxy._from_loader(_load, "Proxy()", is_another_app=True)


Proxy = synchronize_api(_Proxy, target_module=__name__)
