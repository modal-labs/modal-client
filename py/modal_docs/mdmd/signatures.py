# Copyright Modal Labs 2023
import ast
import inspect
import io
import re
import textwrap
import tokenize
import warnings

from synchronicity.synchronizer import FunctionWithAio

from .types import ParsedParam


def _signature_from_ast(func) -> tuple[str, str]:
    """Get function signature, including decorators and comments, from source code

    Traverses functools.wraps-wrappings to get source of underlying function.

    Has the advantage over inspect.signature that it can get decorators, default arguments and comments verbatim
    from the function definition.
    """
    src = inspect.getsource(func)
    src = textwrap.dedent(src)

    def get_source_segment(src, fromline, fromcol, toline, tocol) -> str:
        lines = src.split("\n")
        lines = lines[fromline - 1 : toline]
        lines[-1] = lines[-1][:tocol]
        lines[0] = lines[0][fromcol:]
        return "\n".join(lines)

    tree = ast.parse(src)
    func_def = list(ast.iter_child_nodes(tree))[0]
    assert isinstance(func_def, (ast.FunctionDef, ast.AsyncFunctionDef))
    decorator_starts = [(item.lineno, item.col_offset - 1) for item in func_def.decorator_list]
    declaration_start = min([(func_def.lineno, func_def.col_offset)] + decorator_starts)
    body_start = min((item.lineno, item.col_offset) for item in func_def.body)

    return (
        func_def.name,
        get_source_segment(src, declaration_start[0], declaration_start[1], body_start[0], body_start[1] - 1).strip(),
    )


def _dequote_forward_refs(annotation: str) -> str:
    """Normalize quoted dotted forward refs in annotations."""
    return re.sub(r"""(["'])([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)\1""", r"\2", annotation)


def _annotation_source(source: str, annotation: ast.expr | None) -> str:
    if annotation is None:
        return ""

    annotation_source = ast.get_source_segment(source, annotation) or ""
    return _dequote_forward_refs(annotation_source.strip())


def _default_source(source: str, default: ast.expr | None) -> str | None:
    if default is None:
        return None
    default_source = ast.get_source_segment(source, default)
    if default_source is None:
        return None
    return default_source.strip()


_STRIP_SIGNATURE_WRAP_WIDTH = 80
_STRIP_SIGNATURE_WRAP_INDENT = "    "


def _wrap_stripped_call_signature(line: str, width: int = _STRIP_SIGNATURE_WRAP_WIDTH) -> str:
    """Wrap a single-line call form ``name(a, b, c)`` to fit ``width``, breaking on whitespace."""
    if len(line) <= width:
        return line

    words = line.split()
    if not words:
        return line

    indent = _STRIP_SIGNATURE_WRAP_INDENT
    cont_budget = width - len(indent)

    lines_out: list[str] = []
    buf: list[str] = []
    buf_len = 0
    # First line uses full width; continuation lines reserve space for ``indent``.
    budget = width

    def flush() -> None:
        nonlocal buf, buf_len, budget
        if not buf:
            return
        text = " ".join(buf)
        if lines_out:
            lines_out.append(f"{indent}{text}")
        else:
            lines_out.append(text)
        buf = []
        buf_len = 0
        budget = cont_budget

    for w in words:
        extra = len(w) if not buf else len(w) + 1
        if buf_len + extra <= budget:
            buf.append(w)
            buf_len += extra
            continue

        if buf:
            flush()

        if len(w) > budget:
            sub = textwrap.wrap(
                w,
                width=budget,
                break_long_words=True,
                break_on_hyphens=False,
            )
            for i, piece in enumerate(sub):
                if not lines_out and i == 0:
                    lines_out.append(piece)
                else:
                    lines_out.append(f"{indent}{piece}")
            budget = cont_budget
            continue

        buf = [w]
        buf_len = len(w)

    flush()
    return "\n".join(lines_out)


def strip_signature(signature: str) -> str:
    """Strip out types and decorators from a signature leaving only the function name, arguments, and defaults."""
    signature = "\n".join(l for l in signature.split("\n") if "mdmd:line-hidden" not in l)
    signature = textwrap.dedent(signature)
    token_stream = tokenize.generate_tokens(io.StringIO(signature).readline)
    signature = tokenize.untokenize([token for token in token_stream if token.type != tokenize.COMMENT])
    signature = signature.rstrip()
    if signature and not signature.endswith(":"):
        signature += ":"

    parse_source = signature
    try:
        tree = ast.parse(parse_source)
    except IndentationError:
        parse_source = f"{parse_source}\n    pass\n"
        tree = ast.parse(parse_source)

    func_def = next(node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
    args = func_def.args

    positional = [*args.posonlyargs, *args.args]
    default_offset = len(positional) - len(args.defaults)
    positional_defaults = [None] * default_offset + list(args.defaults)

    parts: list[str] = []
    for i, (arg_node, default_node) in enumerate(zip(positional, positional_defaults, strict=True)):
        if i == len(args.posonlyargs) and args.posonlyargs:
            parts.append("/")
        if default_node is not None:
            default_src = _default_source(parse_source, default_node)
            parts.append(f"{arg_node.arg}={default_src}" if default_src is not None else arg_node.arg)
        else:
            parts.append(arg_node.arg)

    if args.vararg is not None:
        parts.append(f"*{args.vararg.arg}")
    elif args.kwonlyargs:
        parts.append("*")

    for arg_node, default_node in zip(args.kwonlyargs, args.kw_defaults):
        if default_node is not None:
            default_src = _default_source(parse_source, default_node)
            parts.append(f"{arg_node.arg}={default_src}" if default_src is not None else arg_node.arg)
        else:
            parts.append(arg_node.arg)

    if args.kwarg is not None:
        parts.append(f"**{args.kwarg.arg}")

    single = f"{func_def.name}({', '.join(parts)})"
    return _wrap_stripped_call_signature(single)


def get_dataclass_field_annotations(cls) -> dict[str, str]:
    """Map each field name to its verbatim annotation source for a dataclass.

    Reads annotations from the class source so forms like ``str | None`` or
    ``InputStatus`` are preserved rather than rendered as fully-qualified,
    runtime-resolved types (``modal.types.InputStatus``). Only annotations
    declared directly in the class body are returned; callers should fall back
    to the runtime field type for anything missing (e.g. inherited fields).
    """
    try:
        src = textwrap.dedent(inspect.getsource(cls))
        tree = ast.parse(src)
    except (OSError, TypeError, SyntaxError):
        return {}

    class_def = next((node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)), None)
    if class_def is None:
        return {}

    annotations: dict[str, str] = {}
    for node in class_def.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            annotation = _annotation_source(src, node.annotation)
            if annotation:
                annotations[node.target.id] = annotation
    return annotations


def parse_params_from_signature(signature: str) -> list[ParsedParam]:
    signature = "\n".join(l for l in signature.split("\n") if "mdmd:line-hidden" not in l)
    token_stream = tokenize.generate_tokens(io.StringIO(signature).readline)
    signature = tokenize.untokenize([token for token in token_stream if token.type != tokenize.COMMENT])
    signature = signature.rstrip()
    if signature and not signature.endswith(":"):
        signature += ":"

    parse_source = signature
    try:
        tree = ast.parse(parse_source)
    except IndentationError:
        parse_source = f"{parse_source}\n    pass\n"
        tree = ast.parse(parse_source)

    func_def = next(node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
    args = func_def.args

    params: list[ParsedParam] = []

    positional = [*args.posonlyargs, *args.args]
    default_offset = len(positional) - len(args.defaults)
    positional_defaults = [None] * default_offset + list(args.defaults)

    for arg_node, default_node in zip(positional, positional_defaults):
        params.append(
            ParsedParam(
                arg_node.arg,
                _annotation_source(parse_source, arg_node.annotation),
                _default_source(parse_source, default_node),
                None,
            )
        )

    if args.vararg is not None:
        params.append(
            ParsedParam(
                f"*{args.vararg.arg}",
                _annotation_source(parse_source, args.vararg.annotation),
                None,
                None,
            )
        )

    for arg_node, default_node in zip(args.kwonlyargs, args.kw_defaults):
        params.append(
            ParsedParam(
                arg_node.arg,
                _annotation_source(parse_source, arg_node.annotation),
                _default_source(parse_source, default_node),
                None,
            )
        )

    if args.kwarg is not None:
        params.append(
            ParsedParam(
                f"**{args.kwarg.arg}",
                _annotation_source(parse_source, args.kwarg.annotation),
                None,
                None,
            )
        )

    return params


def get_signature(name, callable) -> str:
    """A problem with using *only* this method is that the wrapping method signature will not be respected.
    TODO: use source parsing *only* to extract default arguments, comments (and possibly decorators) and "merge"
          that definition with the outer-most definition."""

    if not (inspect.isfunction(callable) or inspect.ismethod(callable) or isinstance(callable, FunctionWithAio)):
        assert hasattr(callable, "__call__")
        callable = callable.__call__

    try:
        original_name, definition_source = _signature_from_ast(callable)
    except Exception:
        warnings.warn(f"Could not get source signature for {name}. Using fallback.")
        original_name = name
        definition_source = f"def {name}{inspect.signature(callable)}"

    if original_name != name:
        # ugly name and definition replacement hack when needed
        definition_source = definition_source.replace(f"def {original_name}", f"def {name}")

    if (
        "async def" in definition_source
        and not inspect.iscoroutinefunction(callable)
        and not inspect.isasyncgenfunction(callable)
    ):
        # hack to "reset" signature to a blocking one if the underlying source definition is async
        # but the wrapper function isn't (like when synchronicity wraps an async function as a blocking one)
        definition_source = definition_source.replace("async def", "def")
        definition_source = definition_source.replace("asynccontextmanager", "contextmanager")
        definition_source = definition_source.replace("AsyncIterator", "Iterator")

    # remove any synchronicity-internal decorators
    definition_source, _ = re.subn(r"^\s*@synchronizer\..*\n", "", definition_source)

    return definition_source
