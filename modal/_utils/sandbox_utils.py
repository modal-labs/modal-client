# Copyright Modal Labs 2025
from dataclasses import dataclass


@dataclass
class DirectAccessMetadata:
    jwt: str
    url: str
