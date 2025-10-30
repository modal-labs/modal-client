# Copyright Modal Labs 2025
from typing import Any, Optional

from modal_proto import api_pb2
from .._partial_function import _find_partial_methods_for_user_cls, _PartialFunctionFlags

def is_flash_object(experimental_options: Optional[dict[str, Any]]) -> bool:
    return experimental_options.get("flash", False) if experimental_options else False


def get_flash_configs(user_cls: type[Any]) -> Optional[api_pb2.FlashConfig]:
    flash_configs = [
        partial_method.params.flash_config
        for partial_method in _find_partial_methods_for_user_cls(
            user_cls, _PartialFunctionFlags.FLASH_WEB_INTERFACE
        ).values()
        if partial_method.params.flash_config
    ]

    flash_configs = [
                partial_method.params.flash_config
                for partial_method in _find_partial_methods_for_user_cls(
                    type(service.user_cls_instance), _PartialFunctionFlags.FLASH_WEB_INTERFACE
                ).values()
                if partial_method.params.flash_config
            ]

    flash_configs = [
                partial_method.params.flash_config
                for partial_method in _find_partial_methods_for_user_cls(
                    user_cls, _PartialFunctionFlags.FLASH_WEB_INTERFACE
                ).values()
                if partial_method.params.flash_config
            ]
            # TODO: Get the super set of regions
            assert len(flash_configs) == 1
            flash_config = flash_configs[0]

    return flash_config

def get_region_from_flash_configs(flash_configs: Optional[api_pb2.FlashConfig]) -> Optional[str]:
    return list(set([flash_config.region for flash_config in flash_configs]))[0]