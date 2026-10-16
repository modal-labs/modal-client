# Copyright Modal Labs 2026
"""Qualified references stored in code-generation IR."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class ObjectReferenceIR:
    """An importable object identified by module and qualified name."""

    module: str
    qualname: str
