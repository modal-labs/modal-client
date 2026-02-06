# Copyright Modal Labs 2022
"""Function creation status tracking for Modal CLI.

This module contains the FunctionCreationStatus context manager used to
track and display function creation progress.
"""

from __future__ import annotations

from typing import Optional

from modal_proto import api_pb2

from .manager import OutputManager


def _get_suffix_from_web_url_info(url_info: api_pb2.WebUrlInfo) -> str:
    if url_info.truncated:
        suffix = " [grey70](label truncated)[/grey70]"
    elif url_info.label_stolen:
        suffix = " [grey70](label stolen)[/grey70]"
    else:
        suffix = ""
    return suffix


class FunctionCreationStatus:
    """Context manager for tracking and displaying function creation progress."""

    tag: str
    response: Optional[api_pb2.FunctionCreateResponse] = None

    def __init__(self, tag: str):
        self.tag = tag
        self._output_mgr = OutputManager.get()

    def __enter__(self):
        self.status_row = self._output_mgr.add_status_row()
        self.status_row.message(f"Creating function {self.tag}...")
        return self

    def set_response(self, resp: api_pb2.FunctionCreateResponse):
        self.response = resp

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            raise exc_val

        if not self.response:
            self.status_row.finish(f"Unknown error when creating function {self.tag}")

        elif web_url := self.response.handle_metadata.web_url:
            url_info = self.response.function.web_url_info
            requires_proxy_auth = self.response.function.webhook_config.requires_proxy_auth
            proxy_auth_suffix = " 🔑" if requires_proxy_auth else ""
            # Ensure terms used here match terms used in modal.com/docs/guide/webhook-urls doc.
            suffix = _get_suffix_from_web_url_info(url_info)
            # TODO: this is only printed when we're showing progress. Maybe move this somewhere else.
            for warning in self.response.server_warnings:
                self.status_row.warn(warning)
            self.status_row.finish(
                f"Created web function {self.tag} => [magenta underline]{web_url}[/magenta underline]"
                f"{proxy_auth_suffix}{suffix}"
            )

            # Print custom domain in terminal
            for custom_domain in self.response.function.custom_domain_info:
                custom_domain_status_row = self._output_mgr.add_status_row()
                custom_domain_status_row.finish(
                    f"Custom domain for {self.tag} => [magenta underline]{custom_domain.url}[/magenta underline]"
                )

        elif self.response.function.flash_service_urls:
            self.status_row.finish(f"Created function {self.tag}.")
            for flash_service_url in self.response.function.flash_service_urls:
                flash_service_url_status_row = self._output_mgr.add_status_row()
                flash_service_url_status_row.finish(
                    f"Created flash service endpoint for {self.tag} => "
                    f"[magenta underline]{flash_service_url}[/magenta underline]"
                )

        else:
            for warning in self.response.server_warnings:
                self.status_row.warn(warning)
            self.status_row.finish(f"Created function {self.tag}.")
            if self.response.function.method_definitions_set:
                for method_definition in self.response.function.method_definitions.values():
                    if method_definition.web_url:
                        url_info = method_definition.web_url_info
                        suffix = _get_suffix_from_web_url_info(url_info)
                        class_web_endpoint_method_status_row = self._output_mgr.add_status_row()
                        class_web_endpoint_method_status_row.finish(
                            f"Created web endpoint for {method_definition.function_name} => [magenta underline]"
                            f"{method_definition.web_url}[/magenta underline]{suffix}"
                        )
                        for custom_domain in method_definition.custom_domain_info:
                            custom_domain_status_row = self._output_mgr.add_status_row()
                            custom_domain_status_row.finish(
                                f"Custom domain for {method_definition.function_name} => [magenta underline]"
                                f"{custom_domain.url}[/magenta underline]"
                            )
