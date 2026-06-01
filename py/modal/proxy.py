# Copyright Modal Labs 2024

from modal_proto import api_pb2

from ._load_context import LoadContext
from ._object import _Object
from ._resolver import Resolver
from ._utils.async_utils import synchronize_api
from .client import _Client


class _Proxy(_Object, type_prefix="pr"):
    """Proxy objects give your Modal containers a static outbound IP address.

    This can be used for connecting to a remote address with network whitelist, for example
    a database. See [the guide](https://modal.com/docs/guide/proxy-ips) for more information.
    """

    @staticmethod
    def from_name(
        name: str,
        *,
        environment_name: str | None = None,
        client: _Client | None = None,
    ) -> "_Proxy":
        """Reference a Proxy by its name.

        In contrast to most other Modal objects, new Proxy objects must be
        provisioned via the Dashboard and cannot be created on the fly from code.

        Args:
            name: Name of the Proxy in the target environment.
            environment_name: Environment to resolve the name in; defaults to the active environment.
            client: Modal client to use for loading; defaults to `Client.from_env()` when omitted.

        Returns:
            A lazy `Proxy` handle.
        """

        async def _load(self: _Proxy, resolver: Resolver, load_context: LoadContext, existing_object_id: str | None):
            req = api_pb2.ProxyGetRequest(
                name=name,
                environment_name=load_context.environment_name,
            )
            response: api_pb2.ProxyGetResponse = await load_context.client.stub.ProxyGet(req)
            self._hydrate(response.proxy.proxy_id, load_context.client, None)

        rep = _Proxy._repr(name, environment_name)
        return _Proxy._from_loader(
            _load,
            rep,
            skip_reload=True,
            load_context_overrides=LoadContext(client=client, environment_name=environment_name),
        )


Proxy = synchronize_api(_Proxy, target_module=__name__)
