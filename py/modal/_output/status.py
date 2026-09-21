# Copyright Modal Labs 2022
"""Function creation status tracking for Modal CLI.

This module contains the FunctionCreationStatus context manager used to
track and display function creation progress.
"""

from __future__ import annotations

from modal_proto import api_pb2

from .manager import OutputManager


def _get_annotation_for_web_url(url_info: api_pb2.WebUrlInfo) -> str:
    if url_info.truncated:
        suffix = " [grey70](label truncated)[/grey70]"
    elif url_info.label_stolen:
        suffix = " [grey70](label stolen)[/grey70]"
    else:
        suffix = ""
    return suffix


class FunctionCreationStatus:
    """Context manager for tracking and displaying function creation progress."""

    type: str
    name: str
    response: api_pb2.FunctionCreateResponse | None = None

    def __init__(self, type: str, name: str):
        self.type = type
        self.name = name
        self._output_mgr = OutputManager.get()

    def __enter__(self):
        self.status_row = self._output_mgr.add_status_row()
        self.status_row.message(f"Creating {self.type} {self.name}...")
        return self

    def set_response(self, resp: api_pb2.FunctionCreateResponse):
        self.response = resp

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            raise exc_val

        if not self.response:
            self.status_row.finish(f"Unknown error when creating {self.type} {self.name}")
            return

        for warning in self.response.server_warnings:
            self.status_row.warn(warning)

        if web_url := self.response.handle_metadata.web_url:
            url_info = self.response.function.web_url_info
            authenticated = self.response.function.webhook_config.requires_proxy_auth
            auth_suffix = " 🔑" if authenticated else " [yellow](unauthenticated)[/yellow]"
            suffix = _get_annotation_for_web_url(url_info)

            self.status_row.finish(f"Created {self.type} [green]{self.name}[/green]")
            self.status_row.details(f"[dim underline]{web_url}[/dim underline]{auth_suffix}{suffix}")
            for custom_domain in self.response.function.custom_domain_info:
                self.status_row.details(f"Custom domain: [dim underline]{custom_domain.url}[/dim underline]")

        elif self.response.function.flash_service_urls:
            # Despite the gRPC API types, Servers only have one URL
            authenticated = self.response.function.is_server and not self.response.function.http_config.unauthenticated
            auth_suffix = " 🔑" if authenticated else " [yellow](unauthenticated)[/yellow]"
            self.status_row.finish(f"Created {self.type} [green]{self.name}[/green]")
            url = self.response.function.flash_service_urls[0]
            self.status_row.details(f"[dim underline]{url}[/dim underline]{auth_suffix}")

        else:
            self.status_row.finish(f"Created {self.type} [green]{self.name}[/green].")
            if self.response.function.method_definitions_set:
                for method_definition in self.response.function.method_definitions.values():
                    if method_definition.web_url:
                        url_info = method_definition.web_url_info
                        suffix = _get_annotation_for_web_url(url_info)
                        authenticated = method_definition.webhook_config.requires_proxy_auth
                        auth_suffix = " 🔑" if authenticated else " [yellow](unauthenticated)[/yellow]"
                        self.status_row.details(
                            f"{method_definition.function_name} -> "
                            f"[dim underline]{method_definition.web_url}[/dim underline]{auth_suffix}{suffix}"
                        )
                        for custom_domain in method_definition.custom_domain_info:
                            indent = len(method_definition.function_name) * " "
                            self.status_row.details(
                                f"{indent} -> [dim underline]{custom_domain.url}[/dim underline]{auth_suffix}"
                            )
