# Copyright Modal Labs 2022
from modal_utils.async_utils import synchronize_apis

from .object import Handle, Provider


class _ProxyHandle(Handle, type_prefix="pr"):
    pass


class _Proxy(Provider[_ProxyHandle]):
    pass


ProxyHandle, AioProxyHandle = synchronize_apis(_ProxyHandle)
Proxy, AioProxy = synchronize_apis(_Proxy)
