# Copyright Modal Labs 2026
"""Protocol for generated-module emission backends."""

from __future__ import annotations

from typing import Protocol

from ..ir.declarations import ModuleIR


class ModuleEmitter(Protocol):
    """Emit wrapper module source from a :class:`~synchronicity2.codegen.ir.ModuleIR`."""

    def emit_module(
        self,
        ir: ModuleIR,
    ) -> str:
        """Return full source for ``ir.target_module`` using only the IR."""
        ...
