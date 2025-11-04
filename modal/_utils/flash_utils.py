# Copyright Modal Labs 2025
from typing import Any, Literal, Optional, Union

from .._partial_function import _find_partial_methods_for_user_cls, _FlashConfig, _PartialFunctionFlags
from ..exception import (
    InvalidError,
)


def is_flash_object(experimental_options: Optional[dict[str, Any]]) -> bool:
    return experimental_options.get("flash", False) if experimental_options else False


def validate_flash_configs(flash_configs: list[_FlashConfig]) -> None:
    # TODO(claudia): Refactor once multiple flash servers are supported.
    if len(flash_configs) > 1:
        raise InvalidError("Multiple flash objects are not yet supported, please only specify a single flash object.")


def get_flash_configs(user_cls: type[Any]) -> list[_FlashConfig]:
    flash_configs = [
        partial_method.params.flash_config
        for partial_method in _find_partial_methods_for_user_cls(
            user_cls, _PartialFunctionFlags.FLASH_WEB_INTERFACE
        ).values()
        if partial_method.params.flash_config
    ]
    validate_flash_configs(flash_configs)
    return flash_configs


def get_region_from_flash_configs(flash_configs: list[_FlashConfig]) -> Optional[Union[str, Literal[True]]]:
    regions = set()
    for flash_config in flash_configs:
        if flash_config.region:
            regions.add(flash_config.region)

    if len(regions) > 1:
        raise InvalidError(
            "Multiple regions specified for flash objects, "
            "please specify a single region or use `True` to create endpoints in all available regions."
        )

    return regions.pop() if regions else None
