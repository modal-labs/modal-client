# Copyright Modal Labs 2026
"""Emit TypeVar and ParamSpec declarations from IR."""

from __future__ import annotations

from ..ir.declarations import TypeParameterIR


def type_parameter_definition_lines(specs: tuple[TypeParameterIR, ...]) -> list[str]:
    lines: list[str] = []
    for spec in specs:
        if spec.is_paramspec:
            lines.append(f'{spec.name} = typing.ParamSpec("{spec.name}")')
            continue
        args: list[str] = [f'"{spec.name}"']
        args.extend(spec.constraint_parts)
        if spec.bound_value is not None:
            args.append(f"bound={spec.bound_value}")
        if spec.covariant:
            args.append("covariant=True")
        if spec.contravariant:
            args.append("contravariant=True")
        lines.append(f"{spec.name} = typing.TypeVar({', '.join(args)})")
    return lines
