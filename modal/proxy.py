# Copyright Modal Labs 2022
from modal_utils.async_utils import synchronize_apis

from .object import Handle, Provider


class _ProxyHandle(Handle, type_prefix="pr"):
    pass


class _Proxy(Provider[_ProxyHandle]):
    """
    Proxy objects are used to setup secure tunnel connections to a private remote address, for example
    a database.

    Currently `modal.Proxy` objects must be setup with the assistance of Modal staff. If you require a proxy
    please contact us.
    """

    pass


ProxyHandle, AioProxyHandle = synchronize_apis(_ProxyHandle)
Proxy, AioProxy = synchronize_apis(_Proxy)
