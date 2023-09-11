# Copyright Modal Labs 2022
from modal_utils.async_utils import synchronize_api

from .object import _Object


class _Proxy(_Object, type_prefix="pr"):
    """
    Proxy objects are used to setup secure tunnel connections to a private remote address, for example
    a database.

    Currently `modal.Proxy` objects must be setup with the assistance of Modal staff. If you require a proxy
    please contact us.
    """

    pass


Proxy = synchronize_api(_Proxy, target_module=__name__)
