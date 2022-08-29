from modal_utils.async_utils import synchronize_apis

from .object import Object


class _Proxy(Object, type_prefix="pr"):
    pass


Proxy, AioProxy = synchronize_apis(_Proxy)
