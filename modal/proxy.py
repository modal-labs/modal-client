# Copyright Modal Labs 2022
from modal_utils.async_utils import synchronize_apis

from .object import _Handle, _Provider


class _ProxyHandle(_Handle, type_prefix="pr"):
    pass


class _Proxy(_Provider[_ProxyHandle]):
    """
    Proxy objects are used to setup secure tunnel connections to a private remote address, for example
    a database.

    Currently `modal.Proxy` objects must be setup with the assistance of Modal staff. If you require a proxy
    please contact us.
    """

    pass


ProxyHandle, AioProxyHandle = synchronize_apis(_ProxyHandle, target_module=__name__)
Proxy, AioProxy = synchronize_apis(_Proxy, target_module=__name__)
