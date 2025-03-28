# Copyright Modal Labs 2023
import enum
import inspect
import typing
from collections.abc import Coroutine, Iterable
from typing import (
    Any,
    Callable,
    Optional,
    Union,
)

import typing_extensions

from modal_proto import api_pb2

from ._functions import _Function
from ._utils.async_utils import synchronizer
from ._utils.deprecation import deprecation_warning
from ._utils.function_utils import callable_has_non_self_non_default_params, callable_has_non_self_params
from .config import logger
from .exception import InvalidError

MAX_MAX_BATCH_SIZE = 1000
MAX_BATCH_WAIT_MS = 10 * 60 * 1000  # 10 minutes

if typing.TYPE_CHECKING:
    import modal.partial_function


class _PartialFunctionFlags(enum.IntFlag):
    FUNCTION = 1
    BUILD = 2
    ENTER_PRE_SNAPSHOT = 4
    ENTER_POST_SNAPSHOT = 8
    EXIT = 16
    BATCHED = 32
    CLUSTERED = 64  # Experimental: Clustered functions

    @staticmethod
    def all() -> int:
        return ~_PartialFunctionFlags(0)


P = typing_extensions.ParamSpec("P")
ReturnType = typing_extensions.TypeVar("ReturnType", covariant=True)
OriginalReturnType = typing_extensions.TypeVar("OriginalReturnType", covariant=True)


class _PartialFunction(typing.Generic[P, ReturnType, OriginalReturnType]):
    """Intermediate function, produced by @enter, @build, @method, @web_endpoint, or @batched"""

    raw_f: Callable[P, ReturnType]
    flags: _PartialFunctionFlags
    webhook_config: Optional[api_pb2.WebhookConfig]
    is_generator: bool
    batch_max_size: Optional[int]
    batch_wait_ms: Optional[int]
    force_build: bool
    cluster_size: Optional[int]  # Experimental: Clustered functions
    build_timeout: Optional[int]
    max_concurrent_inputs: Optional[int]
    target_concurrent_inputs: Optional[int]

    def __init__(
        self,
        raw_f: Callable[P, ReturnType],
        flags: _PartialFunctionFlags,
        *,
        webhook_config: Optional[api_pb2.WebhookConfig] = None,
        is_generator: Optional[bool] = None,
        batch_max_size: Optional[int] = None,
        batch_wait_ms: Optional[int] = None,
        cluster_size: Optional[int] = None,  # Experimental: Clustered functions
        force_build: bool = False,
        build_timeout: Optional[int] = None,
        max_concurrent_inputs: Optional[int] = None,
        target_concurrent_inputs: Optional[int] = None,
    ):
        self.raw_f = raw_f
        self.flags = flags
        self.webhook_config = webhook_config
        if is_generator is None:
            # auto detect - doesn't work if the function *returns* a generator
            final_is_generator = inspect.isgeneratorfunction(raw_f) or inspect.isasyncgenfunction(raw_f)
        else:
            final_is_generator = is_generator

        self.is_generator = final_is_generator
        self.wrapped = False  # Make sure that this was converted into a FunctionHandle
        self.batch_max_size = batch_max_size
        self.batch_wait_ms = batch_wait_ms
        self.cluster_size = cluster_size  # Experimental: Clustered functions
        self.force_build = force_build
        self.build_timeout = build_timeout
        self.max_concurrent_inputs = max_concurrent_inputs
        self.target_concurrent_inputs = target_concurrent_inputs

    def _get_raw_f(self) -> Callable[P, ReturnType]:
        return self.raw_f

    def _is_web_endpoint(self) -> bool:
        if self.webhook_config is None:
            return False
        return self.webhook_config.type != api_pb2.WEBHOOK_TYPE_UNSPECIFIED

    def __get__(self, obj, objtype=None) -> _Function[P, ReturnType, OriginalReturnType]:
        # to type checkers, any @method or similar function on a modal class, would appear to be
        # of the type PartialFunction and this descriptor would be triggered when accessing it,
        #
        # However, modal classes are *actually* Cls instances (which isn't reflected in type checkers
        # due to Python's lack of type chekcing intersection types), so at runtime the Cls instance would
        # use its __getattr__ rather than this descriptor.
        k = self.raw_f.__name__
        if obj:  # accessing the method on an instance of a class, e.g. `MyClass().fun``
            if hasattr(obj, "_modal_functions"):
                # This happens inside "local" user methods when they refer to other methods,
                # e.g. Foo().parent_method.remote() calling self.other_method.remote()
                return getattr(obj, "_modal_functions")[k]
            else:
                # special edge case: referencing a method of an instance of an
                # unwrapped class (not using app.cls()) with @methods
                # not sure what would be useful here, but let's return a bound version of the underlying function,
                # since the class is just a vanilla class at this point
                # This wouldn't let the user access `.remote()` and `.local()` etc. on the function
                return self.raw_f.__get__(obj, objtype)

        else:  # accessing a method directly on the class, e.g. `MyClass.fun`
            # This happens mainly during serialization of the wrapped underlying class of a Cls
            # since we don't have the instance info here we just return the PartialFunction itself
            # to let it be bound to a variable and become a Function later on
            return self  # type: ignore  # this returns a PartialFunction in a special internal case

    def __del__(self):
        if (self.flags & _PartialFunctionFlags.FUNCTION) and self.wrapped is False:
            logger.warning(
                f"Method or web function {self.raw_f} was never turned into a function."
                " Did you forget a @app.function or @app.cls decorator?"
            )

    def add_flags(self, flags) -> "_PartialFunction":
        # Helper method used internally when stacking decorators
        self.wrapped = True
        return _PartialFunction(
            raw_f=self.raw_f,
            flags=(self.flags | flags),
            webhook_config=self.webhook_config,
            batch_max_size=self.batch_max_size,
            batch_wait_ms=self.batch_wait_ms,
            force_build=self.force_build,
            build_timeout=self.build_timeout,
            max_concurrent_inputs=self.max_concurrent_inputs,
            target_concurrent_inputs=self.target_concurrent_inputs,
        )


def _find_partial_methods_for_user_cls(user_cls: type[Any], flags: int) -> dict[str, _PartialFunction]:
    """Grabs all method on a user class, and returns partials. Includes legacy methods."""
    from .partial_function import PartialFunction  # wrapped type

    partial_functions: dict[str, _PartialFunction] = {}
    for parent_cls in reversed(user_cls.mro()):
        if parent_cls is not object:
            for k, v in parent_cls.__dict__.items():
                if isinstance(v, PartialFunction):  # type: ignore[reportArgumentType]   # synchronicity wrapper types
                    _partial_function: _PartialFunction = typing.cast(_PartialFunction, synchronizer._translate_in(v))
                    if _partial_function.flags & flags:
                        partial_functions[k] = _partial_function

    return partial_functions


def _find_callables_for_obj(user_obj: Any, flags: int) -> dict[str, Callable[..., Any]]:
    """Grabs all methods for an object, and binds them to the class"""
    user_cls: type = type(user_obj)
    return {k: pf.raw_f.__get__(user_obj) for k, pf in _find_partial_methods_for_user_cls(user_cls, flags).items()}


class _MethodDecoratorType:
    @typing.overload
    def __call__(
        self,
        func: "modal.partial_function.PartialFunction[typing_extensions.Concatenate[Any, P], ReturnType, OriginalReturnType]",  # noqa
    ) -> "modal.partial_function.PartialFunction[P, ReturnType, OriginalReturnType]": ...

    @typing.overload
    def __call__(
        self, func: "Callable[typing_extensions.Concatenate[Any, P], Coroutine[Any, Any, ReturnType]]"
    ) -> "modal.partial_function.PartialFunction[P, ReturnType, Coroutine[Any, Any, ReturnType]]": ...

    @typing.overload
    def __call__(
        self, func: "Callable[typing_extensions.Concatenate[Any, P], ReturnType]"
    ) -> "modal.partial_function.PartialFunction[P, ReturnType, ReturnType]": ...

    def __call__(self, func): ...


# TODO(elias): fix support for coroutine type unwrapping for methods (static typing)
def _method(
    _warn_parentheses_missing=None,
    *,
    # Set this to True if it's a non-generator function returning
    # a [sync/async] generator object
    is_generator: Optional[bool] = None,
) -> _MethodDecoratorType:
    """Decorator for methods that should be transformed into a Modal Function registered against this class's App.

    **Usage:**

    ```python
    @app.cls(cpu=8)
    class MyCls:

        @modal.method()
        def f(self):
            ...
    ```
    """
    if _warn_parentheses_missing is not None:
        raise InvalidError(
            "Positional arguments are not allowed. Did you forget parentheses? Suggestion: `@modal.method()`."
        )

    def wrapper(raw_f: Callable[..., Any]) -> _PartialFunction:
        nonlocal is_generator
        if isinstance(raw_f, _PartialFunction) and raw_f.webhook_config:
            raw_f.wrapped = True  # suppress later warning
            raise InvalidError(
                "Web endpoints on classes should not be wrapped by `@method`. "
                "Suggestion: remove the `@method` decorator."
            )
        if isinstance(raw_f, _PartialFunction) and raw_f.batch_max_size is not None:
            raw_f.wrapped = True  # suppress later warning
            raise InvalidError(
                "Batched function on classes should not be wrapped by `@method`. "
                "Suggestion: remove the `@method` decorator."
            )
        return _PartialFunction(raw_f, _PartialFunctionFlags.FUNCTION, is_generator=is_generator)

    return wrapper  # type: ignore   # synchronicity issue with wrapped vs unwrapped types and protocols


def _parse_custom_domains(custom_domains: Optional[Iterable[str]] = None) -> list[api_pb2.CustomDomainConfig]:
    assert not isinstance(custom_domains, str), "custom_domains must be `Iterable[str]` but is `str` instead."
    _custom_domains: list[api_pb2.CustomDomainConfig] = []
    if custom_domains is not None:
        for custom_domain in custom_domains:
            _custom_domains.append(api_pb2.CustomDomainConfig(name=custom_domain))

    return _custom_domains


def _fastapi_endpoint(
    _warn_parentheses_missing=None,
    *,
    method: str = "GET",  # REST method for the created endpoint.
    label: Optional[str] = None,  # Label for created endpoint. Final subdomain will be <workspace>--<label>.modal.run.
    custom_domains: Optional[Iterable[str]] = None,  # Custom fully-qualified domain name (FQDN) for the endpoint.
    docs: bool = False,  # Whether to enable interactive documentation for this endpoint at /docs.
    requires_proxy_auth: bool = False,  # Require Modal-Key and Modal-Secret HTTP Headers on requests.
) -> Callable[[Callable[P, ReturnType]], _PartialFunction[P, ReturnType, ReturnType]]:
    """Convert a function into a basic web endpoint by wrapping it with a FastAPI App.

    Modal will internally use [FastAPI](https://fastapi.tiangolo.com/) to expose a
    simple, single request handler. If you are defining your own `FastAPI` application
    (e.g. if you want to define multiple routes), use `@modal.asgi_app` instead.

    The endpoint created with this decorator will automatically have
    [CORS](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS) enabled
    and can leverage many of FastAPI's features.

    For more information on using Modal with popular web frameworks, see our
    [guide on web endpoints](https://modal.com/docs/guide/webhooks).

    *Added in v0.73.82*: This function replaces the deprecated `@web_endpoint` decorator.
    """
    if isinstance(_warn_parentheses_missing, str):
        # Probably passing the method string as a positional argument.
        raise InvalidError(
            f'Positional arguments are not allowed. Suggestion: `@modal.fastapi_endpoint(method="{method}")`.'
        )
    elif _warn_parentheses_missing is not None:
        raise InvalidError(
            "Positional arguments are not allowed. Did you forget parentheses? Suggestion: `@modal.fastapi_endpoint()`."
        )

    def wrapper(raw_f: Callable[..., Any]) -> _PartialFunction:
        if isinstance(raw_f, _Function):
            raw_f = raw_f.get_raw_f()
            raise InvalidError(
                f"Applying decorators for {raw_f} in the wrong order!\nUsage:\n\n"
                "@app.function()\n@app.fastapi_endpoint()\ndef my_webhook():\n    ..."
            )

        return _PartialFunction(
            raw_f,
            _PartialFunctionFlags.FUNCTION,
            webhook_config=api_pb2.WebhookConfig(
                type=api_pb2.WEBHOOK_TYPE_FUNCTION,
                method=method,
                web_endpoint_docs=docs,
                requested_suffix=label or "",
                async_mode=api_pb2.WEBHOOK_ASYNC_MODE_AUTO,
                custom_domains=_parse_custom_domains(custom_domains),
                requires_proxy_auth=requires_proxy_auth,
            ),
        )

    return wrapper


def _web_endpoint(
    _warn_parentheses_missing=None,
    *,
    method: str = "GET",  # REST method for the created endpoint.
    label: Optional[str] = None,  # Label for created endpoint. Final subdomain will be <workspace>--<label>.modal.run.
    docs: bool = False,  # Whether to enable interactive documentation for this endpoint at /docs.
    custom_domains: Optional[
        Iterable[str]
    ] = None,  # Create an endpoint using a custom domain fully-qualified domain name (FQDN).
    requires_proxy_auth: bool = False,  # Require Modal-Key and Modal-Secret HTTP Headers on requests.
) -> Callable[[Callable[P, ReturnType]], _PartialFunction[P, ReturnType, ReturnType]]:
    """Register a basic web endpoint with this application.

    DEPRECATED: This decorator has been renamed to `@modal.fastapi_endpoint`.

    This is the simple way to create a web endpoint on Modal. The function
    behaves as a [FastAPI](https://fastapi.tiangolo.com/) handler and should
    return a response object to the caller.

    Endpoints created with `@modal.web_endpoint` are meant to be simple, single
    request handlers and automatically have
    [CORS](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS) enabled.
    For more flexibility, use `@modal.asgi_app`.

    To learn how to use Modal with popular web frameworks, see the
    [guide on web endpoints](https://modal.com/docs/guide/webhooks).
    """
    if isinstance(_warn_parentheses_missing, str):
        # Probably passing the method string as a positional argument.
        raise InvalidError('Positional arguments are not allowed. Suggestion: `@modal.web_endpoint(method="GET")`.')
    elif _warn_parentheses_missing is not None:
        raise InvalidError(
            "Positional arguments are not allowed. Did you forget parentheses? Suggestion: `@modal.web_endpoint()`."
        )

    deprecation_warning(
        (2025, 3, 5), "The `@modal.web_endpoint` decorator has been renamed to `@modal.fastapi_endpoint`."
    )

    def wrapper(raw_f: Callable[..., Any]) -> _PartialFunction:
        if isinstance(raw_f, _Function):
            raw_f = raw_f.get_raw_f()
            raise InvalidError(
                f"Applying decorators for {raw_f} in the wrong order!\nUsage:\n\n"
                "@app.function()\n@modal.web_endpoint()\ndef my_webhook():\n    ..."
            )

        return _PartialFunction(
            raw_f,
            _PartialFunctionFlags.FUNCTION,
            webhook_config=api_pb2.WebhookConfig(
                type=api_pb2.WEBHOOK_TYPE_FUNCTION,
                method=method,
                web_endpoint_docs=docs,
                requested_suffix=label,
                async_mode=api_pb2.WEBHOOK_ASYNC_MODE_AUTO,
                custom_domains=_parse_custom_domains(custom_domains),
                requires_proxy_auth=requires_proxy_auth,
            ),
        )

    return wrapper


def _asgi_app(
    _warn_parentheses_missing=None,
    *,
    label: Optional[str] = None,  # Label for created endpoint. Final subdomain will be <workspace>--<label>.modal.run.
    custom_domains: Optional[Iterable[str]] = None,  # Deploy this endpoint on a custom domain.
    requires_proxy_auth: bool = False,  # Require Modal-Key and Modal-Secret HTTP Headers on requests.
) -> Callable[[Callable[..., Any]], _PartialFunction]:
    """Decorator for registering an ASGI app with a Modal function.

    Asynchronous Server Gateway Interface (ASGI) is a standard for Python
    synchronous and asynchronous apps, supported by all popular Python web
    libraries. This is an advanced decorator that gives full flexibility in
    defining one or more web endpoints on Modal.

    **Usage:**

    ```python
    from typing import Callable

    @app.function()
    @modal.asgi_app()
    def create_asgi() -> Callable:
        ...
    ```

    To learn how to use Modal with popular web frameworks, see the
    [guide on web endpoints](https://modal.com/docs/guide/webhooks).
    """
    if isinstance(_warn_parentheses_missing, str):
        raise InvalidError(f'Positional arguments are not allowed. Suggestion: `@modal.asgi_app(label="{label}")`.')
    elif _warn_parentheses_missing is not None:
        raise InvalidError(
            "Positional arguments are not allowed. Did you forget parentheses? Suggestion: `@modal.asgi_app()`."
        )

    def wrapper(raw_f: Callable[..., Any]) -> _PartialFunction:
        if callable_has_non_self_params(raw_f):
            if callable_has_non_self_non_default_params(raw_f):
                raise InvalidError(
                    f"ASGI app function {raw_f.__name__} can't have parameters. See https://modal.com/docs/guide/webhooks#asgi."
                )
            else:
                deprecation_warning(
                    (2024, 9, 4),
                    f"ASGI app function {raw_f.__name__} has default parameters, but shouldn't have any parameters - "
                    f"Modal will drop support for default parameters in a future release.",
                )

        if inspect.iscoroutinefunction(raw_f):
            raise InvalidError(
                f"ASGI app function {raw_f.__name__} is an async function. Only sync Python functions are supported."
            )

        return _PartialFunction(
            raw_f,
            _PartialFunctionFlags.FUNCTION,
            webhook_config=api_pb2.WebhookConfig(
                type=api_pb2.WEBHOOK_TYPE_ASGI_APP,
                requested_suffix=label,
                async_mode=api_pb2.WEBHOOK_ASYNC_MODE_AUTO,
                custom_domains=_parse_custom_domains(custom_domains),
                requires_proxy_auth=requires_proxy_auth,
            ),
        )

    return wrapper


def _wsgi_app(
    _warn_parentheses_missing=None,
    *,
    label: Optional[str] = None,  # Label for created endpoint. Final subdomain will be <workspace>--<label>.modal.run.
    custom_domains: Optional[Iterable[str]] = None,  # Deploy this endpoint on a custom domain.
    requires_proxy_auth: bool = False,  # Require Modal-Key and Modal-Secret HTTP Headers on requests.
) -> Callable[[Callable[..., Any]], _PartialFunction]:
    """Decorator for registering a WSGI app with a Modal function.

    Web Server Gateway Interface (WSGI) is a standard for synchronous Python web apps.
    It has been [succeeded by the ASGI interface](https://asgi.readthedocs.io/en/latest/introduction.html#wsgi-compatibility)
    which is compatible with ASGI and supports additional functionality such as web sockets.
    Modal supports ASGI via [`asgi_app`](/docs/reference/modal.asgi_app).

    **Usage:**

    ```python
    from typing import Callable

    @app.function()
    @modal.wsgi_app()
    def create_wsgi() -> Callable:
        ...
    ```

    To learn how to use this decorator with popular web frameworks, see the
    [guide on web endpoints](https://modal.com/docs/guide/webhooks).
    """
    if isinstance(_warn_parentheses_missing, str):
        raise InvalidError(f'Positional arguments are not allowed. Suggestion: `@modal.wsgi_app(label="{label}")`.')
    elif _warn_parentheses_missing is not None:
        raise InvalidError(
            "Positional arguments are not allowed. Did you forget parentheses? Suggestion: `@modal.wsgi_app()`."
        )

    def wrapper(raw_f: Callable[..., Any]) -> _PartialFunction:
        if callable_has_non_self_params(raw_f):
            if callable_has_non_self_non_default_params(raw_f):
                raise InvalidError(
                    f"WSGI app function {raw_f.__name__} can't have parameters. See https://modal.com/docs/guide/webhooks#wsgi."
                )
            else:
                deprecation_warning(
                    (2024, 9, 4),
                    f"WSGI app function {raw_f.__name__} has default parameters, but shouldn't have any parameters - "
                    f"Modal will drop support for default parameters in a future release.",
                )

        if inspect.iscoroutinefunction(raw_f):
            raise InvalidError(
                f"WSGI app function {raw_f.__name__} is an async function. Only sync Python functions are supported."
            )

        return _PartialFunction(
            raw_f,
            _PartialFunctionFlags.FUNCTION,
            webhook_config=api_pb2.WebhookConfig(
                type=api_pb2.WEBHOOK_TYPE_WSGI_APP,
                requested_suffix=label,
                async_mode=api_pb2.WEBHOOK_ASYNC_MODE_AUTO,
                custom_domains=_parse_custom_domains(custom_domains),
                requires_proxy_auth=requires_proxy_auth,
            ),
        )

    return wrapper


def _web_server(
    port: int,
    *,
    startup_timeout: float = 5.0,  # Maximum number of seconds to wait for the web server to start.
    label: Optional[str] = None,  # Label for created endpoint. Final subdomain will be <workspace>--<label>.modal.run.
    custom_domains: Optional[Iterable[str]] = None,  # Deploy this endpoint on a custom domain.
    requires_proxy_auth: bool = False,  # Require Modal-Key and Modal-Secret HTTP Headers on requests.
) -> Callable[[Callable[..., Any]], _PartialFunction]:
    """Decorator that registers an HTTP web server inside the container.

    This is similar to `@asgi_app` and `@wsgi_app`, but it allows you to expose a full HTTP server
    listening on a container port. This is useful for servers written in other languages like Rust,
    as well as integrating with non-ASGI frameworks like aiohttp and Tornado.

    **Usage:**

    ```python
    import subprocess

    @app.function()
    @modal.web_server(8000)
    def my_file_server():
        subprocess.Popen("python -m http.server -d / 8000", shell=True)
    ```

    The above example starts a simple file server, displaying the contents of the root directory.
    Here, requests to the web endpoint will go to external port 8000 on the container. The
    `http.server` module is included with Python, but you could run anything here.

    Internally, the web server is transparently converted into a web endpoint by Modal, so it has
    the same serverless autoscaling behavior as other web endpoints.

    For more info, see the [guide on web endpoints](https://modal.com/docs/guide/webhooks).
    """
    if not isinstance(port, int) or port < 1 or port > 65535:
        raise InvalidError("First argument of `@web_server` must be a local port, such as `@web_server(8000)`.")
    if startup_timeout <= 0:
        raise InvalidError("The `startup_timeout` argument of `@web_server` must be positive.")

    def wrapper(raw_f: Callable[..., Any]) -> _PartialFunction:
        return _PartialFunction(
            raw_f,
            _PartialFunctionFlags.FUNCTION,
            webhook_config=api_pb2.WebhookConfig(
                type=api_pb2.WEBHOOK_TYPE_WEB_SERVER,
                requested_suffix=label,
                async_mode=api_pb2.WEBHOOK_ASYNC_MODE_AUTO,
                custom_domains=_parse_custom_domains(custom_domains),
                web_server_port=port,
                web_server_startup_timeout=startup_timeout,
                requires_proxy_auth=requires_proxy_auth,
            ),
        )

    return wrapper


def _disallow_wrapping_method(f: _PartialFunction, wrapper: str) -> None:
    if f.flags & _PartialFunctionFlags.FUNCTION:
        f.wrapped = True  # Hack to avoid warning about not using @app.cls()
        raise InvalidError(f"Cannot use `@{wrapper}` decorator with `@method`.")


def _build(
    _warn_parentheses_missing=None, *, force: bool = False, timeout: int = 86400
) -> Callable[[Union[Callable[[Any], Any], _PartialFunction]], _PartialFunction]:
    """
    Decorator for methods that execute at _build time_ to create a new Image layer.

    **Deprecated**: This function is deprecated. We recommend using `modal.Volume`
    to store large assets (such as model weights) instead of writing them to the
    Image during the build process. For other use cases, you can replace this
    decorator with the `Image.run_function` method.

    **Usage**

    ```python notest
    @app.cls(gpu="A10G")
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
    if _warn_parentheses_missing is not None:
        raise InvalidError(
            "Positional arguments are not allowed. Did you forget parentheses? Suggestion: `@modal.build()`."
        )

    deprecation_warning(
        (2025, 1, 15),
        "The `@modal.build` decorator is deprecated and will be removed in a future release."
        "\n\nWe now recommend storing large assets (such as model weights) using a `modal.Volume`"
        " instead of writing them directly into the `modal.Image` filesystem."
        " For other use cases we recommend using `Image.run_function` instead."
        "\n\nSee https://modal.com/docs/guide/modal-1-0-migration for more information.",
    )

    def wrapper(f: Union[Callable[[Any], Any], _PartialFunction]) -> _PartialFunction:
        if isinstance(f, _PartialFunction):
            _disallow_wrapping_method(f, "build")
            f.force_build = force
            f.build_timeout = timeout
            return f.add_flags(_PartialFunctionFlags.BUILD)
        else:
            return _PartialFunction(f, _PartialFunctionFlags.BUILD, force_build=force, build_timeout=timeout)

    return wrapper


def _enter(
    _warn_parentheses_missing=None,
    *,
    snap: bool = False,
) -> Callable[[Union[Callable[[Any], Any], _PartialFunction]], _PartialFunction]:
    """Decorator for methods which should be executed when a new container is started.

    See the [lifeycle function guide](https://modal.com/docs/guide/lifecycle-functions#enter) for more information."""
    if _warn_parentheses_missing is not None:
        raise InvalidError(
            "Positional arguments are not allowed. Did you forget parentheses? Suggestion: `@modal.enter()`."
        )

    if snap:
        flag = _PartialFunctionFlags.ENTER_PRE_SNAPSHOT
    else:
        flag = _PartialFunctionFlags.ENTER_POST_SNAPSHOT

    def wrapper(f: Union[Callable[[Any], Any], _PartialFunction]) -> _PartialFunction:
        if isinstance(f, _PartialFunction):
            _disallow_wrapping_method(f, "enter")
            return f.add_flags(flag)
        else:
            return _PartialFunction(f, flag)

    return wrapper


ExitHandlerType = Union[
    # NOTE: return types of these callables should be `Union[None, Awaitable[None]]` but
    #       synchronicity type stubs would strip Awaitable so we use Any for now
    # Original, __exit__ style method signature (now deprecated)
    Callable[[Any, Optional[type[BaseException]], Optional[BaseException], Any], Any],
    # Forward-looking unparametrized method
    Callable[[Any], Any],
]


def _exit(_warn_parentheses_missing=None) -> Callable[[ExitHandlerType], _PartialFunction]:
    """Decorator for methods which should be executed when a container is about to exit.

    See the [lifeycle function guide](https://modal.com/docs/guide/lifecycle-functions#exit) for more information."""
    if _warn_parentheses_missing is not None:
        raise InvalidError(
            "Positional arguments are not allowed. Did you forget parentheses? Suggestion: `@modal.exit()`."
        )

    def wrapper(f: ExitHandlerType) -> _PartialFunction:
        if isinstance(f, _PartialFunction):
            _disallow_wrapping_method(f, "exit")

        return _PartialFunction(f, _PartialFunctionFlags.EXIT)

    return wrapper


def _batched(
    _warn_parentheses_missing=None,
    *,
    max_batch_size: int,
    wait_ms: int,
) -> Callable[[Callable[..., Any]], _PartialFunction]:
    """Decorator for functions or class methods that should be batched.

    **Usage**

    ```python notest
    @app.function()
    @modal.batched(max_batch_size=4, wait_ms=1000)
    async def batched_multiply(xs: list[int], ys: list[int]) -> list[int]:
        return [x * y for x, y in zip(xs, xs)]

    # call batched_multiply with individual inputs
    batched_multiply.remote.aio(2, 100)
    ```

    See the [dynamic batching guide](https://modal.com/docs/guide/dynamic-batching) for more information.
    """
    if _warn_parentheses_missing is not None:
        raise InvalidError(
            "Positional arguments are not allowed. Did you forget parentheses? Suggestion: `@modal.batched()`."
        )
    if max_batch_size < 1:
        raise InvalidError("max_batch_size must be a positive integer.")
    if max_batch_size >= MAX_MAX_BATCH_SIZE:
        raise InvalidError(f"max_batch_size must be less than {MAX_MAX_BATCH_SIZE}.")
    if wait_ms < 0:
        raise InvalidError("wait_ms must be a non-negative integer.")
    if wait_ms >= MAX_BATCH_WAIT_MS:
        raise InvalidError(f"wait_ms must be less than {MAX_BATCH_WAIT_MS}.")

    def wrapper(raw_f: Callable[..., Any]) -> _PartialFunction:
        if isinstance(raw_f, _Function):
            raw_f = raw_f.get_raw_f()
            raise InvalidError(
                f"Applying decorators for {raw_f} in the wrong order!\nUsage:\n\n"
                "@app.function()\n@modal.batched()\ndef batched_function():\n    ..."
            )
        return _PartialFunction(
            raw_f,
            _PartialFunctionFlags.FUNCTION | _PartialFunctionFlags.BATCHED,
            batch_max_size=max_batch_size,
            batch_wait_ms=wait_ms,
        )

    return wrapper


def _concurrent(
    _warn_parentheses_missing=None,
    *,
    max_inputs: int,  # Hard limit on each container's input concurrency
    target_inputs: Optional[int] = None,  # Input concurrency that Modal's autoscaler should target
) -> Callable[[Union[Callable[..., Any], _PartialFunction]], _PartialFunction]:
    """Decorator that allows individual containers to handle multiple inputs concurrently.

    The concurrency mechanism depends on whether the function is async or not:
    - Async functions will run inputs on a single thread as asyncio tasks.
    - Synchronous functions will use multi-threading. The code must be thread-safe.

    Input concurrency will be most useful for workflows that are IO-bound
    (e.g., making network requests) or when running an inference server that supports
    dynamic batching.

    When `target_inputs` is set, Modal's autoscaler will try to provision resources
    such that each container is running that many inputs concurrently, rather than
    autoscaling based on `max_inputs`. Containers may burst up to up to `max_inputs`
    if resources are insufficient to remain at the target concurrency, e.g. when the
    arrival rate of inputs increases. This can trade-off a small increase in average
    latency to avoid larger tail latencies from input queuing.

    **Examples:**
    ```python
    # Stack the decorator under `@app.function()` to enable input concurrency
    @app.function()
    @modal.concurrent(max_inputs=100)
    async def f(data):
        # Async function; will be scheduled as asyncio task
        ...

    # With `@app.cls()`, apply the decorator at the class level, not on individual methods
    @app.cls()
    @modal.concurrent(max_inputs=100, target_inputs=80)
    class C:
        @modal.method()
        def f(self, data):
            # Sync function; must be thread-safe
            ...

    ```

    """
    if _warn_parentheses_missing is not None:
        raise InvalidError(
            "Positional arguments are not allowed. Did you forget parentheses? Suggestion: `@modal.concurrent()`."
        )

    if target_inputs and target_inputs > max_inputs:
        raise InvalidError("`target_inputs` parameter cannot be greater than `max_inputs`.")

    def wrapper(obj: Union[Callable[..., Any], _PartialFunction]) -> _PartialFunction:
        if isinstance(obj, _PartialFunction):
            # Risky that we need to mutate the parameters here; should make this safer
            obj.max_concurrent_inputs = max_inputs
            obj.target_concurrent_inputs = target_inputs
            obj.add_flags(_PartialFunctionFlags.FUNCTION)
            return obj

        return _PartialFunction(
            obj,
            _PartialFunctionFlags.FUNCTION,
            max_concurrent_inputs=max_inputs,
            target_concurrent_inputs=target_inputs,
        )

    return wrapper
