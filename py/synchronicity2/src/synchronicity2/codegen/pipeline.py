# Copyright Modal Labs 2026
"""Orchestrate parsing registered modules and emitting generated source.

Pipeline:
1. :func:`~synchronicity2.codegen.parsing.declarations.build_module_ir` parses declarations and annotations.
2. :class:`~synchronicity2.codegen.emission.module_codegen.SyncAsyncModuleEmitter` emits source.

Function- and method-level parsing lives in :mod:`synchronicity2.codegen.parsing`; source generation
lives in :mod:`synchronicity2.codegen.emission`.
"""

from __future__ import annotations

import importlib
import typing

from synchronicity2.module import Module

from .emission.module_codegen import SyncAsyncModuleEmitter
from .emission.protocol import ModuleEmitter
from .ir.references import ObjectReferenceIR
from .parsing.declarations import build_module_ir

if typing.TYPE_CHECKING:
    from synchronicity import Synchronizer as Synchronicity1Synchronizer

__all__ = [
    "compile_module",
    "compile_modules",
]


def compile_module(
    module: Module,
    *,
    synchronizer_module: str,
    runtime_package: str = "synchronicity2",
    emitter: ModuleEmitter | None = None,
    forbidden_wrapper_modules: frozenset[str] | None = None,
    synchronicity1_synchronizer_path: str | None = None,
) -> str:
    """
    Compile wrapped items for a single target module.

    Args:
        module: The Module instance with registered items
        synchronizer_module: Generated module containing the shared runtime synchronizer.
        runtime_package: Dotted import path for runtime submodules in generated imports
        emitter: Optional emitter (defaults to :class:`SyncAsyncModuleEmitter` with ``runtime_package``)
        synchronicity1_synchronizer_path: Optional ``module:attribute`` path to a Synchronicity 1 synchronizer used
            for compatibility type resolution and emitted runtime configuration.

    Returns:
        String containing compiled wrapper code for this module
    """
    _validate_module_path(synchronizer_module, label="synchronizer module")
    if synchronizer_module == module.target_module:
        raise ValueError(f"Synchronizer module {synchronizer_module!r} conflicts with the generated wrapper module")
    synchronicity1_synchronizer, _synchronicity1_synchronizer_ref = _resolve_synchronicity1_synchronizer(
        synchronicity1_synchronizer_path
    )
    return _compile_module(
        module,
        synchronizer_module=synchronizer_module,
        runtime_package=runtime_package,
        emitter=emitter,
        forbidden_wrapper_modules=forbidden_wrapper_modules,
        synchronicity1_synchronizer=synchronicity1_synchronizer,
    )


def _compile_module(
    module: Module,
    *,
    synchronizer_module: str,
    runtime_package: str,
    emitter: ModuleEmitter | None,
    forbidden_wrapper_modules: frozenset[str] | None,
    synchronicity1_synchronizer: Synchronicity1Synchronizer | None,
) -> str:
    ir = build_module_ir(
        module,
        synchronizer_module=synchronizer_module,
        forbidden_wrapper_modules=forbidden_wrapper_modules,
        synchronicity1_synchronizer=synchronicity1_synchronizer,
    )
    gen = emitter or SyncAsyncModuleEmitter(runtime_package=runtime_package)
    return gen.emit_module(ir)


def compile_modules(
    modules: list[Module],
    *,
    synchronizer_module: str,
    runtime_package: str = "synchronicity2",
    emitter: ModuleEmitter | None = None,
    synchronicity1_synchronizer_path: str | None = None,
) -> dict[str, str]:
    """
    Compile wrapped items into separate module files.

    Args:
        modules: List of Module instances to compile
        synchronizer_module: Generated module containing the synchronizer shared by every wrapper in this invocation.
        runtime_package: Dotted import path for runtime modules (``types``, ``descriptor``,
            ``synchronizer``) referenced in generated code. Use a vendored package for
            wheels that should not depend on the PyPI ``synchronicity2`` distribution.
        emitter: Optional codegen backend (defaults to sync/async wrapper emitter)
        synchronicity1_synchronizer_path: Optional importable ``module:attribute`` path for Synchronicity 1
            compatibility.

    Returns:
        Dict mapping module names to their generated code
    """
    gen = emitter or SyncAsyncModuleEmitter(runtime_package=runtime_package)

    _validate_module_path(synchronizer_module, label="synchronizer module")
    target_modules = {module.target_module for module in modules}
    if synchronizer_module in target_modules:
        raise ValueError(f"Synchronizer module {synchronizer_module!r} conflicts with a generated wrapper module")
    implementation_modules = {
        item.__module__ for module in modules for item in module._module_items() if hasattr(item, "__module__")
    }
    if synchronizer_module in implementation_modules:
        raise ValueError(f"Synchronizer module {synchronizer_module!r} conflicts with an implementation module")

    synchronicity1_synchronizer, synchronicity1_synchronizer_ref = _resolve_synchronicity1_synchronizer(
        synchronicity1_synchronizer_path
    )
    if synchronicity1_synchronizer_ref is not None and synchronizer_module == synchronicity1_synchronizer_ref.module:
        raise ValueError(f"Synchronizer module {synchronizer_module!r} conflicts with its Synchronicity 1 owner module")
    result = {}
    forbidden_wrapper_modules = frozenset((*target_modules, synchronizer_module))
    for module in modules:
        code = _compile_module(
            module,
            synchronizer_module=synchronizer_module,
            runtime_package=runtime_package,
            emitter=gen,
            forbidden_wrapper_modules=forbidden_wrapper_modules,
            synchronicity1_synchronizer=synchronicity1_synchronizer,
        )
        if code:
            result[module.target_module] = code

    if result:
        result[synchronizer_module] = _emit_synchronizer_module(
            runtime_package=runtime_package,
            synchronicity1_synchronizer_ref=synchronicity1_synchronizer_ref,
        )
    return result


def _emit_synchronizer_module(
    *,
    runtime_package: str,
    synchronicity1_synchronizer_ref: ObjectReferenceIR | None,
) -> str:
    compatibility_import = ""
    constructor_argument = ""
    if synchronicity1_synchronizer_ref is not None:
        ref = synchronicity1_synchronizer_ref
        compatibility_import = f"import {ref.module}\n"
        constructor_argument = f"synchronicity1_synchronizer={ref.module}.{ref.qualname}"
    return f"""# Generated by synchronicity2.
# This module should not be modified directly.

{compatibility_import}from {runtime_package}.synchronizer import Synchronizer

synchronizer = Synchronizer({constructor_argument})
"""


def _validate_module_path(path: str, *, label: str) -> None:
    if not path or any(not part.isidentifier() for part in path.split(".")):
        raise ValueError(f"Invalid {label} {path!r}; expected a qualified Python module path")


def _resolve_synchronicity1_synchronizer(
    path: str | None,
) -> tuple[Synchronicity1Synchronizer | None, ObjectReferenceIR | None]:
    if path is None:
        return None, None

    module_name, separator, qualname = path.partition(":")
    if not separator or not module_name or not qualname:
        raise ValueError(f"Invalid Synchronicity 1 synchronizer path {path!r}; expected 'module:attribute'")
    if any(not part.isidentifier() for part in (*module_name.split("."), *qualname.split("."))):
        raise ValueError(f"Invalid Synchronicity 1 synchronizer path {path!r}")

    try:
        module = importlib.import_module(module_name)
        synchronizer: object = module
        for part in qualname.split("."):
            synchronizer = getattr(synchronizer, part)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Could not resolve Synchronicity 1 synchronizer path {path!r}: {exc}") from exc

    from synchronicity import Synchronizer as Synchronicity1Synchronizer

    if not isinstance(synchronizer, Synchronicity1Synchronizer):
        raise TypeError(f"{path!r} does not refer to a Synchronicity 1 Synchronizer")
    return synchronizer, ObjectReferenceIR(module_name, qualname)
