# Copyright Modal Labs 2023
from datetime import date

from modal_utils.async_utils import synchronize_api

from .exception import deprecation_warning
from .network_file_system import _NetworkFileSystem


class _SharedVolume(_NetworkFileSystem):
    def __init__(self, *args, **kwargs) -> None:
        """`SharedVolume(...)` is deprecated. Please use `NetworkFileSystem.new(...)` instead."""
        deprecation_warning(date(2023, 7, 5), _SharedVolume.__init__.__doc__)
        obj = _NetworkFileSystem.new(*args, **kwargs)
        self._init_from_other(obj)

    @staticmethod
    def new(*args, **kwargs) -> "_NetworkFileSystem":
        """`SharedVolume.new(...)` is deprecated. Please use `NetworkFileSystem.new(...)` instead."""
        deprecation_warning(date(2023, 7, 5), _SharedVolume.new.__doc__)
        return _NetworkFileSystem.new(*args, **kwargs)

    @staticmethod
    def persisted(*args, **kwargs) -> _NetworkFileSystem:
        """`SharedVolume.persisted(...)` is deprecated. Please use `NetworkFileSystem.persisted(...)` instead."""
        deprecation_warning(date(2023, 7, 5), _SharedVolume.persisted.__doc__)
        return _NetworkFileSystem.persisted(*args, **kwargs)


SharedVolume = synchronize_api(_SharedVolume)
