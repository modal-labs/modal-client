# Copyright Modal Labs 2025
from dataclasses import dataclass


@dataclass
class SandboxCommandRouterAccess:
    url: str
    jwt: str
