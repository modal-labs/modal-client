# Copyright Modal Labs 2023
import enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Type,
    Union,
)

from modal._types import typechecked
from modal_proto import api_pb2

from ._utils.async_utils import synchronize_api, synchronizer
from ._utils.function_utils import method_has_params
from .config import logger
from .exception import InvalidError, deprecation_warning
from .functions import _Function


class _PartialFunctionFlags(enum.IntFlag):
    FUNCTION: int = 1
    BUILD: int = 2
    ENTER_PRE_CHECKPOINT: int = 4
    ENTER_POST_CHECKPOINT: int = 8
    EXIT: int = 16


class _PartialFunction:
    """Intermediate function, produced by @method or @web_endpoint"""

    raw_f: Callable[..., Any]
    flags: _PartialFunctionFlags
    webhook_config: Optional[api_pb2.WebhookConfig]
    is_generator: Optional[bool]
    keep_warm: Optional[int]

    def __init__(
        self,
        raw_f: Callable[..., Any],
        flags: _PartialFunctionFlags,
        webhook_config: Optional[api_pb2.WebhookConfig] = None,
        is_generator: Optional[bool] = None,
        keep_warm: Optional[int] = None,
    ):
        self.raw_f = raw_f
        self.flags = flags
        self.webhook_config = webhook_config
        self.is_generator = is_generator
        self.keep_warm = keep_warm
        self.wrapped = False  # Make sure that this was converted into a FunctionHandle

    def __get__(self, obj, objtype=None) -> _Function:
        # This only happens inside user methods when they refer to other methods
        k = self.raw_f.__name__
        if obj:  # Cls().fun
            function = getattr(obj, "_modal_functions")[k]
        else:  # Cls.fun
            function = getattr(objtype, "_modal_functions")[k]
        return function

    def __del__(self):
        if (self.flags & _PartialFunctionFlags.FUNCTION) and self.wrapped is False:
            logger.warning(
                f"Method or web function {self.raw_f} was never turned into a function."
                " Did you forget a @stub.function or @stub.cls decorator?"
            )

    def add_flags(self, flags) -> "_PartialFunction":
        # Helper method used internally when stacking decorators
        self.wrapped = True
        return _PartialFunction(
            raw_f=self.raw_f,
            flags=(self.flags | flags),
            webhook_config=self.webhook_config,
            keep_warm=self.keep_warm,
        )


PartialFunction = synchronize_api(_PartialFunction)


def _find_partial_methods_for_cls(user_cls: Type, flags: _PartialFunctionFlags) -> Dict[str, _PartialFunction]:
    """Grabs all method on a user class"""
    partial_functions: Dict[str, PartialFunction] = {}
    for parent_cls in user_cls.mro():
        if parent_cls is not object:
            for k, v in parent_cls.__dict__.items():
                if isinstance(v, PartialFunction):
                    partial_function = synchronizer._translate_in(v)  # TODO: remove need for?
                    if partial_function.flags & flags:
                        partial_functions[k] = partial_function

    return partial_functions


def _find_callables_for_cls(user_cls: Type, flags: _PartialFunctionFlags) -> Dict[str, Callable]:
    """Grabs all method on a user class, and returns callables. Includes legacy methods."""
    functions: Dict[str, Callable] = {}

    # Build up a list of legacy attributes to check
    check_attrs: List[str] = []
    if flags & _PartialFunctionFlags.BUILD:
        check_attrs += ["__build__", "__abuild__"]
    if flags & _PartialFunctionFlags.ENTER_POST_CHECKPOINT:
        check_attrs += ["__enter__", "__aenter__"]
    if flags & _PartialFunctionFlags.EXIT:
        check_attrs += ["__exit__", "__aexit__"]

    # Grab legacy lifecycle methods
    for attr in check_attrs:
        if hasattr(user_cls, attr):
            suggested = attr.strip("_")
            if is_async := suggested.startswith("a"):
                suggested = suggested[1:]
            async_suggestion = " (on an async method)" if is_async else ""
            message = (
                f"Using `{attr}` methods for class lifecycle management is deprecated."
                f" Please try using the `modal.{suggested}` decorator{async_suggestion} instead."
                " See https://modal.com/docs/guide/lifecycle-functions for more information."
            )
            deprecation_warning((2024, 2, 21), message, show_source=True)
            functions[attr] = getattr(user_cls, attr)

    # Grab new decorator-based methods
    for k, pf in _find_partial_methods_for_cls(user_cls, flags).items():
        functions[k] = pf.raw_f

    return functions


def _find_callables_for_obj(user_obj: Any, flags: _PartialFunctionFlags) -> Dict[str, Callable]:
    """Grabs all methods for an object, and binds them to the class"""
    user_cls: Type = type(user_obj)
    return {k: meth.__get__(user_obj) for k, meth in _find_callables_for_cls(user_cls, flags).items()}


def _method(
    _warn_parentheses_missing=None,
    *,
    # Set this to True if it's a non-generator function returning
    # a [sync/async] generator object
    is_generator: Optional[bool] = None,
    keep_warm: Optional[int] = None,  # An optional number of containers to always keep warm.
) -> Callable[[Callable[..., Any]], _PartialFunction]:
    """Decorator for methods that should be transformed into a Modal Function registered against this class's stub.

    **Usage:**

    ```python
    @stub.cls(cpu=8)
    class MyCls:

        @modal.method()
        def f(self):
            ...
    ```
    """
    if _warn_parentheses_missing:
        raise InvalidError("Positional arguments are not allowed. Did you forget parentheses? Suggestion: `@method()`.")

    def wrapper(raw_f: Callable[..., Any]) -> _PartialFunction:
        if isinstance(raw_f, _PartialFunction) and raw_f.webhook_config:
            raw_f.wrapped = True  # suppress later warning
            raise InvalidError(
                "Web endpoints on classes should not be wrapped by `@method`. Suggestion: remove the `@method` decorator."
            )
        return _PartialFunction(raw_f, _PartialFunctionFlags.FUNCTION, is_generator=is_generator, keep_warm=keep_warm)

    return wrapper


def _parse_custom_domains(custom_domains: Optional[Iterable[str]] = None) -> List[api_pb2.CustomDomainConfig]:
    assert not isinstance(custom_domains, str), "custom_domains must be `Iterable[str]` but is `str` instead."
    _custom_domains: List[api_pb2.CustomDomainConfig] = []
    if custom_domains is not None:
        for custom_domain in custom_domains:
            _custom_domains.append(api_pb2.CustomDomainConfig(name=custom_domain))

    return _custom_domains


@typechecked
def _web_endpoint(
    _warn_parentheses_missing=None,
    *,
    method: str = "GET",  # REST method for the created endpoint.
    label: Optional[str] = None,  # Label for created endpoint. Final subdomain will be <workspace>--<label>.modal.run.
    wait_for_response: bool = True,  # Whether requests should wait for and return the function response.
    custom_domains: Optional[
        Iterable[str]
    ] = None,  # Create an endpoint using a custom domain fully-qualified domain name (FQDN).
) -> Callable[[Callable[..., Any]], _PartialFunction]:
    """Register a basic web endpoint with this application.

    This is the simple way to create a web endpoint on Modal. The function
    behaves as a [FastAPI](https://fastapi.tiangolo.com/) handler and should
    return a response object to the caller.

    Endpoints created with `@stub.web_endpoint` are meant to be simple, single
    request handlers and automatically have
    [CORS](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS) enabled.
    For more flexibility, use `@stub.asgi_app`.

    To learn how to use Modal with popular web frameworks, see the
    [guide on web endpoints](https://modal.com/docs/guide/webhooks).
    """
    if isinstance(_warn_parentheses_missing, str):
        # Probably passing the method string as a positional argument.
        raise InvalidError('Positional arguments are not allowed. Suggestion: `@web_endpoint(method="GET")`.')
    elif _warn_parentheses_missing:
        raise InvalidError(
            "Positional arguments are not allowed. Did you forget parentheses? Suggestion: `@web_endpoint()`."
        )

    def wrapper(raw_f: Callable[..., Any]) -> _PartialFunction:
        if isinstance(raw_f, _Function):
            raw_f = raw_f.get_raw_f()
            raise InvalidError(
                f"Applying decorators for {raw_f} in the wrong order!\nUsage:\n\n"
                "@stub.function()\n@stub.web_endpoint()\ndef my_webhook():\n    ..."
            )
        if not wait_for_response:
            _response_mode = api_pb2.WEBHOOK_ASYNC_MODE_TRIGGER
        else:
            _response_mode = api_pb2.WEBHOOK_ASYNC_MODE_AUTO  # the default

        # self._loose_webhook_configs.add(raw_f)

        return _PartialFunction(
            raw_f,
            _PartialFunctionFlags.FUNCTION,
            api_pb2.WebhookConfig(
                type=api_pb2.WEBHOOK_TYPE_FUNCTION,
                method=method,
                requested_suffix=label,
                async_mode=_response_mode,
                custom_domains=_parse_custom_domains(custom_domains),
            ),
        )

    return wrapper


@typechecked
def _asgi_app(
    _warn_parentheses_missing=None,
    *,
    label: Optional[str] = None,  # Label for created endpoint. Final subdomain will be <workspace>--<label>.modal.run.
    wait_for_response: bool = True,  # Whether requests should wait for and return the function response.
    custom_domains: Optional[
        Iterable[str]
    ] = None,  # Create an endpoint using a custom domain fully-qualified domain name.
) -> Callable[[Callable[..., Any]], _PartialFunction]:
    """Decorator for registering an ASGI app with a Modal function.

    Asynchronous Server Gateway Interface (ASGI) is a standard for Python
    synchronous and asynchronous apps, supported by all popular Python web
    libraries. This is an advanced decorator that gives full flexibility in
    defining one or more web endpoints on Modal.

    **Usage:**

    ```python
    from typing import Callable

    @stub.function()
    @modal.asgi_app()
    def create_asgi() -> Callable:
        ...
    ```

    To learn how to use Modal with popular web frameworks, see the
    [guide on web endpoints](https://modal.com/docs/guide/webhooks).
    """
    if isinstance(_warn_parentheses_missing, str):
        raise InvalidError('Positional arguments are not allowed. Suggestion: `@asgi_app(label="foo")`.')
    elif _warn_parentheses_missing:
        raise InvalidError(
            "Positional arguments are not allowed. Did you forget parentheses? Suggestion: `@asgi_app()`."
        )

    def wrapper(raw_f: Callable[..., Any]) -> _PartialFunction:
        if not wait_for_response:
            _response_mode = api_pb2.WEBHOOK_ASYNC_MODE_TRIGGER
        else:
            _response_mode = api_pb2.WEBHOOK_ASYNC_MODE_AUTO  # the default

        return _PartialFunction(
            raw_f,
            _PartialFunctionFlags.FUNCTION,
            api_pb2.WebhookConfig(
                type=api_pb2.WEBHOOK_TYPE_ASGI_APP,
                requested_suffix=label,
                async_mode=_response_mode,
                custom_domains=_parse_custom_domains(custom_domains),
            ),
        )

    return wrapper


@typechecked
def _wsgi_app(
    _warn_parentheses_missing=None,
    *,
    label: Optional[str] = None,  # Label for created endpoint. Final subdomain will be <workspace>--<label>.modal.run.
    wait_for_response: bool = True,  # Whether requests should wait for and return the function response.
    custom_domains: Optional[
        Iterable[str]
    ] = None,  # Create an endpoint using a custom domain fully-qualified domain name.
) -> Callable[[Callable[..., Any]], _PartialFunction]:
    """Decorator for registering a WSGI app with a Modal function.

    Web Server Gateway Interface (WSGI) is a standard for synchronous Python web apps.
    It has been [succeeded by the ASGI interface](https://asgi.readthedocs.io/en/latest/introduction.html#wsgi-compatibility) which is compatible with ASGI and supports
    additional functionality such as web sockets. Modal supports ASGI via [`asgi_app`](/docs/reference/modal.asgi_app).

    **Usage:**

    ```python
    from typing import Callable

    @stub.function()
    @modal.wsgi_app()
    def create_wsgi() -> Callable:
        ...
    ```

    To learn how to use this decorator with popular web frameworks, see the
    [guide on web endpoints](https://modal.com/docs/guide/webhooks).
    """
    if isinstance(_warn_parentheses_missing, str):
        raise InvalidError('Positional arguments are not allowed. Suggestion: `@wsgi_app(label="foo")`.')
    elif _warn_parentheses_missing:
        raise InvalidError(
            "Positional arguments are not allowed. Did you forget parentheses? Suggestion: `@wsgi_app()`."
        )

    def wrapper(raw_f: Callable[..., Any]) -> _PartialFunction:
        if not wait_for_response:
            _response_mode = api_pb2.WEBHOOK_ASYNC_MODE_TRIGGER
        else:
            _response_mode = api_pb2.WEBHOOK_ASYNC_MODE_AUTO  # the default

        return _PartialFunction(
            raw_f,
            _PartialFunctionFlags.FUNCTION,
            api_pb2.WebhookConfig(
                type=api_pb2.WEBHOOK_TYPE_WSGI_APP,
                requested_suffix=label,
                async_mode=_response_mode,
                custom_domains=_parse_custom_domains(custom_domains),
            ),
        )

    return wrapper


def _disallow_wrapping_method(f: _PartialFunction, wrapper: str) -> None:
    if f.flags & _PartialFunctionFlags.FUNCTION:
        f.wrapped = True  # Hack to avoid warning about not using @stub.cls()
        raise InvalidError(f"Cannot use `@{wrapper}` decorator with `@method`.")


@typechecked
def _build(
    _warn_parentheses_missing=None,
) -> Callable[[Union[Callable[[Any], Any], _PartialFunction]], _PartialFunction]:
    """
    Decorator for methods that should execute at _build time_ to create a new layer
    in a `modal.Image`.

    See the [lifeycle function guide](https://modal.com/docs/guide/lifecycle-functions#build) for more information.

    **Usage**

    ```python notest
    @stub.cls(gpu="A10G")
    class AlpacaLoRAModel:
        @build()
        def download_models(self):
            model = LlamaForCausalLM.from_pretrained(
                base_model,
            )
            PeftModel.from_pretrained(model, lora_weights)
            LlamaTokenizer.from_pretrained(base_model)
    ```
    """
    if _warn_parentheses_missing:
        raise InvalidError("Positional arguments are not allowed. Did you forget parentheses? Suggestion: `@build()`.")

    def wrapper(f: Union[Callable[[Any], Any], _PartialFunction]) -> _PartialFunction:
        if isinstance(f, _PartialFunction):
            _disallow_wrapping_method(f, "build")
            return f.add_flags(_PartialFunctionFlags.BUILD)
        else:
            return _PartialFunction(f, _PartialFunctionFlags.BUILD)

    return wrapper


@typechecked
def _enter(
    _warn_parentheses_missing=None,
    *,
    snap: bool = False,
) -> Callable[[Union[Callable[[Any], Any], _PartialFunction]], _PartialFunction]:
    """Decorator for methods which should be executed when a new container is started.

    See the [lifeycle function guide](https://modal.com/docs/guide/lifecycle-functions#enter) for more information."""
    if _warn_parentheses_missing:
        raise InvalidError("Positional arguments are not allowed. Did you forget parentheses? Suggestion: `@enter()`.")

    if snap:
        flag = _PartialFunctionFlags.ENTER_PRE_CHECKPOINT
    else:
        flag = _PartialFunctionFlags.ENTER_POST_CHECKPOINT

    def wrapper(f: Union[Callable[[Any], Any], _PartialFunction]) -> _PartialFunction:
        if isinstance(f, _PartialFunction):
            _disallow_wrapping_method(f, "enter")
            return f.add_flags(flag)
        else:
            return _PartialFunction(f, flag)

    return wrapper


ExitHandlerType = Union[
    # Original, __exit__ style method signature (now deprecated)
    Callable[[Any, Optional[Type[BaseException]], Optional[BaseException], Any], None],
    # Forward-looking unparameterized method
    Callable[[Any], None],
]


@typechecked
def _exit(_warn_parentheses_missing=None) -> Callable[[ExitHandlerType], _PartialFunction]:
    """Decorator for methods which should be executed when a container is about to exit.

    See the [lifeycle function guide](https://modal.com/docs/guide/lifecycle-functions#exit) for more information."""
    if _warn_parentheses_missing:
        raise InvalidError("Positional arguments are not allowed. Did you forget parentheses? Suggestion: `@exit()`.")

    def wrapper(f: ExitHandlerType) -> _PartialFunction:
        if isinstance(f, _PartialFunction):
            _disallow_wrapping_method(f, "exit")
        if method_has_params(f):
            message = (
                "Support for decorating parameterized methods with `@exit` has been deprecated."
                " To avoid future errors, please update your code by removing the parameters."
            )
            deprecation_warning((2024, 2, 23), message)
        return _PartialFunction(f, _PartialFunctionFlags.EXIT)

    return wrapper


method = synchronize_api(_method)
web_endpoint = synchronize_api(_web_endpoint)
asgi_app = synchronize_api(_asgi_app)
wsgi_app = synchronize_api(_wsgi_app)
build = synchronize_api(_build)
enter = synchronize_api(_enter)
exit = synchronize_api(_exit)
