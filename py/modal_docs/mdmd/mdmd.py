# Copyright Modal Labs 2023
"""mdmd - MoDal MarkDown"""

import html
import inspect
import re
import typing
import warnings
from collections.abc import Callable
from enum import Enum, EnumMeta
from types import ModuleType

import synchronicity.synchronizer

from .signatures import get_signature, parse_params_from_signature, strip_signature
from .types import ParsedDoc, ParsedParam, ParsedRaise


def _escape_svelte_html_attr(value: str) -> str:
    """Escape text for double-quoted HTML / Svelte component attributes.

    Uses :func:`html.escape` for normal HTML attribute safety. Svelte additionally
    treats ``{`` and ``}`` as template expression boundaries, so those are encoded
    as numeric character references (not covered by HTML escapers).
    """
    s = html.escape(value, quote=True)
    return s.replace("{", "&#123;").replace("}", "&#125;")


def clean_docstring(docstring: str | None) -> str:
    if docstring is None:
        docstring = ""
    else:
        docstring = inspect.cleandoc(docstring)

    docstring = "\n".join(
        l
        for l in docstring.split("\n")
        if "mdmd:line-hidden" not in l and "mdmd:namespace" not in l and "mdmd:exported" not in l
    )

    if docstring and not docstring.endswith("\n"):
        docstring += "\n"

    return docstring


def parse_docstring(name: str, signature: str, docstring: str) -> ParsedDoc:
    lines = inspect.cleandoc(docstring).splitlines()

    def extract_section(headers: list[str]) -> tuple[str | None, tuple[int, int] | None]:
        start_idx = None
        for idx, line in enumerate(lines):
            if any(line.startswith(header) for header in headers):
                start_idx = idx
                break
        if start_idx is None:
            return None, None

        # A section runs until a sibling/outdent: a non-blank line indented no further right than the
        # section header line (Google/NumPy style: ``Args:`` at column 0, body indented under it).
        # Blank lines do not end a section (so Args can have blank lines between items, Examples
        # between fences and prose). Fenced Markdown (`` ``` ``) is handled first so a flush-left
        # fence opening still starts a block instead of being treated as outdented body text.
        header_line = lines[start_idx]
        header_indent = len(header_line) - len(header_line.lstrip())
        end_idx = start_idx + 1
        in_fence = False
        while end_idx < len(lines):
            line = lines[end_idx]
            stripped = line.strip()
            if in_fence:
                end_idx += 1
                if stripped.startswith("```"):
                    in_fence = False
                continue
            if stripped.startswith("```"):
                in_fence = True
                end_idx += 1
                continue
            if not stripped:
                end_idx += 1
                continue
            line_indent = len(line) - len(line.lstrip())
            if line_indent <= header_indent:
                break
            end_idx += 1

        return "\n".join(lines[start_idx:end_idx]).strip(), (start_idx, end_idx)

    def section_body_without_header(section: str | None) -> str | None:
        if section is None:
            return None

        body_lines = section.splitlines()[1:]
        if not any(line.strip() for line in body_lines):
            return None

        indent = min(len(line) - len(line.lstrip()) for line in body_lines if line.strip())
        normalized_lines = [line[indent:] if line.strip() else "" for line in body_lines]
        return "\n".join(normalized_lines).strip()

    args_section, args_range = extract_section(["Args:"])
    returns_section, returns_range = extract_section(["Returns:", "Yields:"])
    raises_section, raises_range = extract_section(["Raises:"])
    examples_section, examples_range = extract_section(["Examples:", "Example:"])

    section_ranges = [
        section_range for section_range in [args_range, returns_range, raises_range, examples_range] if section_range
    ]

    description_lines = []
    for idx, line in enumerate(lines):
        if any(start <= idx < end for start, end in section_ranges):
            continue
        description_lines.append(line)
    description = "\n".join(description_lines).strip()

    def parse_args(section: str | None) -> dict[str, ParsedParam]:
        if section is None:
            return {}
        body_lines = section.splitlines()[1:]
        non_empty_body_lines = [line for line in body_lines if line.strip()]
        if not non_empty_body_lines:
            return {}

        item_indent = min(len(line) - len(line.lstrip()) for line in non_empty_body_lines)
        params: dict[str, ParsedParam] = {}
        current_name: str | None = None
        current_type = ""
        current_description_lines: list[str] = []

        def flush_current() -> None:
            nonlocal current_name, current_type, current_description_lines
            if current_name is None:
                return
            description = " ".join(current_description_lines).strip() or None
            params[current_name] = ParsedParam(
                name=current_name, type=current_type, default=None, description=description
            )
            current_name = None
            current_type = ""
            current_description_lines = []

        for line in body_lines:
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip())
            stripped = line.strip()

            if indent == item_indent and ":" in stripped:
                lhs, rhs = stripped.split(":", 1)
                match = re.match(r"^([*]{0,2}[A-Za-z_]\w*)(?:\s*\(([^)]+)\))?$", lhs.strip())
                if match:
                    flush_current()
                    current_name = match.group(1)
                    current_type = match.group(2) or ""
                    if rhs.strip():
                        current_description_lines.append(rhs.strip())
                    continue

            if current_name is not None:
                current_description_lines.append(stripped)

        flush_current()
        return params

    def parse_raises(section: str | None) -> list[ParsedRaise]:
        if section is None:
            return []
        body_lines = section.splitlines()[1:]
        non_empty_body_lines = [line for line in body_lines if line.strip()]
        if not non_empty_body_lines:
            return []

        item_indent = min(len(line) - len(line.lstrip()) for line in non_empty_body_lines)
        parsed_raises: list[ParsedRaise] = []
        current_type: str | None = None
        current_description_lines: list[str] = []

        def flush_current() -> None:
            nonlocal current_type, current_description_lines
            if current_type is None:
                return
            parsed_raises.append(
                ParsedRaise(type=current_type, description=" ".join(current_description_lines).strip())
            )
            current_type = None
            current_description_lines = []

        for line in body_lines:
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip())
            stripped = line.strip()

            if indent == item_indent and ":" in stripped:
                lhs, rhs = stripped.split(":", 1)
                flush_current()
                current_type = lhs.strip()
                if rhs.strip():
                    current_description_lines.append(rhs.strip())
                continue

            if current_type is not None:
                current_description_lines.append(stripped)

        flush_current()
        return parsed_raises

    docstring_params = parse_args(args_section)
    signature_params: list[ParsedParam] = [] if not signature.strip() else parse_params_from_signature(signature)

    params: list[ParsedParam] = []
    for sig_param in signature_params:
        param_name = sig_param.name
        if param_name not in docstring_params:
            # warnings.warn(f"Parameter {param_name} not found in docstring for {name}")
            continue
        param = docstring_params[param_name]
        param.default = sig_param.default
        if not param.type:
            param.type = sig_param.type

        params.append(param)

    return ParsedDoc(
        name=name,
        description=description,
        params=params,
        returns=section_body_without_header(returns_section),
        raises=parse_raises(raises_section),
        examples=section_body_without_header(examples_section),
    )


def _markdown_body_from_parsed_doc(parsed: ParsedDoc) -> str:
    """Render description, parameters, returns, raises, and usage (examples) for Markdown output."""
    output: list[str] = []
    output.append(parsed.description or "")
    output.append("")
    if parsed.params:
        output.append("**Parameters**\n")
        for param in parsed.params:
            name_esc = _escape_svelte_html_attr(param.name)
            type_esc = _escape_svelte_html_attr(param.type)
            default_attr = (
                f' defaultValue="{_escape_svelte_html_attr(param.default)}"' if param.default is not None else ""
            )
            desc_esc = _escape_svelte_html_attr(param.description or "")
            output.append(f'<Parameter name="{name_esc}" type="{type_esc}"{default_attr} description="{desc_esc}" />')
        output.append("")

    if parsed.returns:
        output.append("**Returns**\n")
        output.append(parsed.returns)
        output.append("")

    if parsed.raises:
        output.append("**Raises**\n")
        for parsed_raise in parsed.raises:
            output.append(f"- `{parsed_raise.type}`: {parsed_raise.description}")
        output.append("")

    if parsed.examples:
        output.append("**Usage**\n")
        output.append(parsed.examples)
        output.append("")

    return "\n".join(output)


def function_str(name: str, func) -> str:
    signature = get_signature(name, func)
    signature = "\n".join(l for l in signature.split("\n") if "mdmd:line-hidden" not in l)
    docstring = clean_docstring(func.__doc__)
    parsed_docstring = parse_docstring(name, signature, docstring)

    stripped_signature = strip_signature(signature)

    return "\n".join(
        [
            f"```python\n{stripped_signature}\n```",
            _markdown_body_from_parsed_doc(parsed_docstring),
        ]
    )


def _is_typeddict(obj) -> bool:
    """Check if a class is a TypedDict."""
    if hasattr(typing, "is_typeddict"):
        return typing.is_typeddict(obj)
    # Fallback: TypedDicts have these special attributes
    return (
        inspect.isclass(obj)
        and issubclass(obj, dict)
        and hasattr(obj, "__required_keys__")
        and hasattr(obj, "__optional_keys__")
    )


def _typeddict_str(name, obj) -> str:
    """Generate documentation for a TypedDict class."""
    hints = typing.get_type_hints(obj)
    optional_keys: frozenset[str] = getattr(obj, "__optional_keys__", frozenset())

    # Build the class declaration showing fields
    lines = [f"class {name}(TypedDict):"]
    for field_name, field_type in hints.items():
        type_str = inspect.formatannotation(field_type)
        if field_name in optional_keys:
            lines.append(f"    {field_name}: NotRequired[{type_str}]")
        else:
            lines.append(f"    {field_name}: {type_str}")

    decl = "```python\n" + "\n".join(lines) + "\n```\n\n"

    parts = [decl]
    docstring = clean_docstring(obj.__doc__)
    if docstring:
        parts.append(docstring + "\n")

    return "".join(parts)


def class_str(name, obj, title_level="##", decl_override: str | None = None, member_prefix: str = ""):
    def qual_name(cls):
        if cls.__module__ == "builtins":
            return cls.__name__
        return f"{cls.__module__}.{cls.__name__}"

    if _is_typeddict(obj):
        return _typeddict_str(name, obj)

    if decl_override is not None:
        decl = f"```python\n{decl_override}\n```\n\n"
    else:
        bases = [qual_name(b) for b in obj.__bases__]
        bases_str = f"({', '.join(bases)})" if bases else ""
        decl = f"""```python
class {name}{bases_str}
```\n\n"""
    parts = ["\n", decl]
    docstring = clean_docstring(obj.__doc__)
    class_doc_markdown: str | None = None
    if docstring:
        parsed_class_doc = parse_docstring(name, "", docstring)
        class_doc_markdown = _markdown_body_from_parsed_doc(parsed_class_doc)

    if isinstance(obj, EnumMeta) and not docstring:
        # Python 3.11 removed the docstring from enums
        class_doc_markdown = "An enumeration.\n"

    if class_doc_markdown:
        parts.append(class_doc_markdown + "\n")

    if isinstance(obj, EnumMeta):
        enum_vals = "\n".join(f"* `{k}`" for k in obj.__members__.keys())
        parts.append(f"The possible values are:\n\n{enum_vals}\n")

    else:
        init = inspect.unwrap(obj.__init__)

        if (inspect.isfunction(init) or inspect.ismethod(init)) and not object_is_private("constructor", init):
            parts.append(function_str("__init__", init))

    member_title_level = title_level + "#"

    entries = {}

    def rec_update_attributes(cls):
        # first bases, then class itself
        for base_cls in cls.__bases__:
            rec_update_attributes(base_cls)
        entries.update(cls.__dict__)

    rec_update_attributes(obj)

    for member_name, member in entries.items():
        if isinstance(member, synchronicity.synchronizer.classproperty):
            member_obj = getattr(obj, member_name)
            member_cls = type(member_obj)
            decl = f"{member_name}: {member_cls.__name__}"
            parts.append(f"\n{member_title_level} {member_name}\n\n")
            parts.append(
                class_str(
                    member_name,
                    member_cls,
                    title_level=title_level + "#",
                    decl_override=decl,
                    member_prefix=f"{member_name}.",
                )
            )
            continue
        elif isinstance(member, classmethod) or isinstance(member, staticmethod):
            # get the original function definition instead of the descriptor object
            member = getattr(obj, member_name)
        elif isinstance(member, property):
            # Check if this property returns a namespace class (marked with mdmd:namespace)
            # that should be documented inline (e.g., Sandbox.filesystem -> SandboxFilesystem)
            fget = member.fget
            try:
                return_type = typing.get_type_hints(fget).get("return") if fget else None
            except Exception:
                return_type = None
            if (
                return_type is not None
                and inspect.isclass(return_type)
                and (return_type.__doc__ or "").lstrip().startswith("mdmd:namespace")
            ):
                decl = f"{member_name}: {return_type.__name__.lstrip('_')}"
                parts.append(f"\n{member_title_level} {member_name}\n\n")
                parts.append(
                    class_str(
                        member_name,
                        return_type,
                        title_level=title_level + "#",
                        decl_override=decl,
                        member_prefix=f"{member_name}.",
                    )
                )
                continue
            member = fget
        elif isinstance(member, (synchronicity.synchronizer.FunctionWithAio, synchronicity.synchronizer.MethodWithAio)):
            member = member._func

        if object_is_private(member_name, member):
            continue

        if callable(member):
            parts.append(f"\n{member_title_level} {member_prefix}{member_name}\n\n")
            parts.append(function_str(member_name, member))

    return "".join(parts)


def module_str(header, module, title_level="#", filter_items: Callable[[ModuleType, str], bool] = None):
    header = [f"{title_level} {header}\n\n"]
    docstring = clean_docstring(module.__doc__)
    if docstring:
        header.append(docstring + "\n")

    object_docs = []
    member_title_level = title_level + "#"
    for qual_name, name, item in module_items(module, filter_items):
        try:
            if hasattr(item, "__wrapped__"):
                item = item.__wrapped__
        except KeyError:
            pass
        except:
            print("failed on", qual_name, name, item)
            raise
        if inspect.isclass(item):
            classdoc = class_str(name, item, title_level=member_title_level)
            object_docs.append(f"{member_title_level} {qual_name}\n\n")
            object_docs.append(classdoc)
        elif callable(item):
            funcdoc = function_str(name, item)
            object_docs.append(f"{member_title_level} {qual_name}\n\n")
            object_docs.append(funcdoc)
        else:
            item_doc = getattr(module, f"__doc__{name}", None)
            if item_doc:
                # variable documentation
                object_docs.append(f"{member_title_level} {qual_name}\n\n")
                object_docs.append(item_doc)
            else:
                warnings.warn(f"Not sure how to document: {name} ({item}")

    if object_docs:
        return "".join(header + object_docs)
    return ""


def object_is_private(name, obj):
    docstring = inspect.getdoc(obj)
    if docstring is None:
        docstring = ""
    module = getattr(obj, "__module__", None)  # obj is class
    if not module:
        cls = getattr(obj, "__class__", None)  # obj is instance
        if cls:
            module = getattr(cls, "__module__", None)
    if module == "builtins":
        return True

    if docstring.lstrip().startswith("mdmd:hidden") or name.startswith("_"):
        return True

    return False


def module_reexports_all(module: ModuleType) -> bool:
    """Use `mdmd:exported` for modules that re-export objects defined elsewhere via __all__."""
    docstring = module.__doc__ or ""
    return any("mdmd:exported" in line for line in docstring.split("\n"))


def default_filter(module, item_name):
    """Include non-private objects defined in the module itself or its private counterpart."""
    item = getattr(module, item_name)
    if object_is_private(item_name, item) or inspect.ismodule(item):
        return False
    member_module = getattr(item, "__module__", type(item).__module__)
    if member_module == module.__name__:
        return True
    if module_reexports_all(module) and item_name in getattr(module, "__all__", ()):
        return True
    # Also allow items from the corresponding private module (e.g., modal._foo for modal.foo)
    parts = module.__name__.rsplit(".", 1)
    if len(parts) == 2:
        private_module = f"{parts[0]}._{parts[1]}"
    else:
        private_module = f"_{parts[0]}"
    return member_module == private_module


def package_filter(module_prefix: str):
    """Include non-private objects defined in any module with the prefix `module_prefix`"""

    def return_filter(module, item_name):
        item = getattr(module, item_name)
        if object_is_private(item_name, item) or inspect.ismodule(item):
            return False
        member_module = getattr(item, "__module__", type(item).__module__)
        return member_module.startswith(module_prefix)

    return return_filter


def module_items(module, filter_items: Callable[[ModuleType, str], bool] = None):
    """Returns filtered members of module"""
    if filter_items is None:
        # default filter is to only include classes and functions declared (or whose type is declared) in the file
        filter_items = default_filter

    for member_name, member in inspect.getmembers(module):
        # only modal items
        if not filter_items(module, member_name):
            continue

        qual_name = f"{module.__name__}.{member_name}"
        yield qual_name, member_name, member


class Category(Enum):
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
