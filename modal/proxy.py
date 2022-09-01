from .object import Handle


class _ProxyHandle(Handle, type_prefix="pr"):
    pass
