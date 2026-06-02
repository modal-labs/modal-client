# Copyright Modal Labs 2026
from ._image import _Image
from ._utils.async_utils import synchronize_api

Image = synchronize_api(_Image, target_module=__name__)
