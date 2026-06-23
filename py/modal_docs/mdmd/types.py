# Copyright Modal Labs 2023
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedParam:
    name: str
    type: str
    default: Optional[str]
    description: Optional[str]


@dataclass
class ParsedRaise:
    type: str
    description: str


@dataclass
class ParsedDoc:
    """Structured docstring sections from Google/NumPy-style docs.

    Used for functions (with signature-backed parameters), classes, modules, and
    other objects that share the same section conventions.
    """

    name: str
    description: Optional[str]
    params: list[ParsedParam]
    returns: Optional[str]
    raises: list[ParsedRaise]
    examples: Optional[str]
    see_also: Optional[str]
