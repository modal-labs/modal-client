# Copyright Modal Labs 2023

from ._utils.async_utils import synchronize_api
from .exception import deprecation_error


class _SharedVolume:
    def __init__(self, *args, **kwargs):
        """`SharedVolume` is deprecated. We recommend `Volume` (https://modal.com/docs/guide/volumes) instead."""
        deprecation_error((2023, 7, 5), _SharedVolume.__init__.__doc__)

    @staticmethod
    def new(*args, **kwargs):
        """`SharedVolume` is deprecated. We recommend `Volume` (https://modal.com/docs/guide/volumes) instead."""
        deprecation_error((2023, 7, 5), _SharedVolume.new.__doc__)

    @staticmethod
    def persisted(*args, **kwargs):
        """`SharedVolume` is deprecated. We recommend `Volume` (https://modal.com/docs/guide/volumes) instead."""
        deprecation_error((2023, 7, 5), _SharedVolume.persisted.__doc__)


SharedVolume = synchronize_api(_SharedVolume)
