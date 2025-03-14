# Copyright Modal Labs 2024
import importlib
import typing
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import modal._object
import modal._runtime.container_io_manager
import modal.cls
from modal import Function
from modal._functions import _Function
from modal._partial_function import _find_partial_methods_for_user_cls, _PartialFunctionFlags
from modal._utils.async_utils import synchronizer
from modal._utils.function_utils import LocalFunctionError, is_async as get_is_async, is_global_object
from modal.exception import ExecutionError, InvalidError
from modal_proto import api_pb2

if typing.TYPE_CHECKING:
    import modal.app
    import modal.partial_function
    from modal._runtime.asgi import LifespanManager


@dataclass
class FinalizedFunction:
    callable: Callable[..., Any]
    is_async: bool
    is_generator: bool
    data_format: int  # api_pb2.DataFormat
    lifespan_manager: Optional["LifespanManager"] = None


class Service(metaclass=ABCMeta):
    """Common interface for singular functions and class-based "services"

    There are differences in the importing/finalization logic, and this
    "protocol"/abc basically defines a common interface for the two types
    of "Services" after the point of import.
    """

    user_cls_instance: Any
    app: Optional["modal.app._App"]
    service_deps: Optional[Sequence["modal._object._Object"]]

    @abstractmethod
    def get_finalized_functions(
        self, fun_def: api_pb2.Function, container_io_manager: "modal._runtime.container_io_manager.ContainerIOManager"
    ) -> dict[str, "FinalizedFunction"]: ...


def construct_webhook_callable(
    user_defined_callable: Callable,
    webhook_config: api_pb2.WebhookConfig,
    container_io_manager: "modal._runtime.container_io_manager.ContainerIOManager",
):
    # Note: aiohttp is a significant dependency of the `asgi` module, so we import it locally
    from modal._runtime import asgi

    # For webhooks, the user function is used to construct an asgi app:
    if webhook_config.type == api_pb2.WEBHOOK_TYPE_ASGI_APP:
        # Function returns an asgi_app, which we can use as a callable.
        return asgi.asgi_app_wrapper(user_defined_callable(), container_io_manager)

    elif webhook_config.type == api_pb2.WEBHOOK_TYPE_WSGI_APP:
        # Function returns an wsgi_app, which we can use as a callable
        return asgi.wsgi_app_wrapper(user_defined_callable(), container_io_manager)

    elif webhook_config.type == api_pb2.WEBHOOK_TYPE_FUNCTION:
        # Function is a webhook without an ASGI app. Create one for it.
        return asgi.asgi_app_wrapper(
            asgi.magic_fastapi_app(user_defined_callable, webhook_config.method, webhook_config.web_endpoint_docs),
            container_io_manager,
        )

    elif webhook_config.type == api_pb2.WEBHOOK_TYPE_WEB_SERVER:
        # Function spawns an HTTP web server listening at a port.
        user_defined_callable()

        # We intentionally try to connect to the external interface instead of the loopback
        # interface here so users are forced to expose the server. This allows us to potentially
        # change the implementation to use an external bridge in the future.
        host = asgi.get_ip_address(b"eth0")
        port = webhook_config.web_server_port
        startup_timeout = webhook_config.web_server_startup_timeout
        asgi.wait_for_web_server(host, port, timeout=startup_timeout)
        return asgi.asgi_app_wrapper(asgi.web_server_proxy(host, port), container_io_manager)
    else:
        raise InvalidError(f"Unrecognized web endpoint type {webhook_config.type}")


@dataclass
class ImportedFunction(Service):
    user_cls_instance: Any
    app: Optional["modal.app._App"]
    service_deps: Optional[Sequence["modal._object._Object"]]

    _user_defined_callable: Callable[..., Any]

    def get_finalized_functions(
        self, fun_def: api_pb2.Function, container_io_manager: "modal._runtime.container_io_manager.ContainerIOManager"
    ) -> dict[str, "FinalizedFunction"]:
        # Check this property before we turn it into a method (overriden by webhooks)
        is_async = get_is_async(self._user_defined_callable)
        # Use the function definition for whether this is a generator (overriden by webhooks)
        is_generator = fun_def.function_type == api_pb2.Function.FUNCTION_TYPE_GENERATOR

        webhook_config = fun_def.webhook_config
        if not webhook_config.type:
            # for non-webhooks, the runnable is straight forward:
            return {
                "": FinalizedFunction(
                    callable=self._user_defined_callable,
                    is_async=is_async,
                    is_generator=is_generator,
                    data_format=api_pb2.DATA_FORMAT_PICKLE,
                )
            }

        web_callable, lifespan_manager = construct_webhook_callable(
            self._user_defined_callable, fun_def.webhook_config, container_io_manager
        )

        return {
            "": FinalizedFunction(
                callable=web_callable,
                lifespan_manager=lifespan_manager,
                is_async=True,
                is_generator=True,
                data_format=api_pb2.DATA_FORMAT_ASGI,
            )
        }


@dataclass
class ImportedClass(Service):
    user_cls_instance: Any
    app: Optional["modal.app._App"]
    service_deps: Optional[Sequence["modal._object._Object"]]

    _partial_functions: dict[str, "modal._partial_function._PartialFunction"]

    def get_finalized_functions(
        self, fun_def: api_pb2.Function, container_io_manager: "modal._runtime.container_io_manager.ContainerIOManager"
    ) -> dict[str, "FinalizedFunction"]:
        finalized_functions = {}
        for method_name, partial in self._partial_functions.items():
            partial = synchronizer._translate_in(partial)  # ugly
            user_func = partial.raw_f
            # Check this property before we turn it into a method (overriden by webhooks)
            is_async = get_is_async(user_func)
            # Use the function definition for whether this is a generator (overriden by webhooks)
            is_generator = partial.is_generator
            webhook_config = partial.webhook_config

            bound_func = user_func.__get__(self.user_cls_instance)

            if not webhook_config or webhook_config.type == api_pb2.WEBHOOK_TYPE_UNSPECIFIED:
                # for non-webhooks, the runnable is straight forward:
                finalized_function = FinalizedFunction(
                    callable=bound_func,
                    is_async=is_async,
                    is_generator=is_generator,
                    data_format=api_pb2.DATA_FORMAT_PICKLE,
                )
            else:
                web_callable, lifespan_manager = construct_webhook_callable(
                    bound_func, webhook_config, container_io_manager
                )
                finalized_function = FinalizedFunction(
                    callable=web_callable,
                    lifespan_manager=lifespan_manager,
                    is_async=True,
                    is_generator=True,
                    data_format=api_pb2.DATA_FORMAT_ASGI,
                )
            finalized_functions[method_name] = finalized_function
        return finalized_functions


def get_user_class_instance(cls: typing.Union[type, modal.cls.Cls], args: tuple, kwargs: dict[str, Any]) -> typing.Any:
    """Returns instance of the underlying class to be used as the `self`

    The input `cls` can either be the raw Python class the user has declared ("user class"),
    or an @app.cls-decorated version of it which is a modal.Cls-instance wrapping the user class.
    """
    if isinstance(cls, modal.cls.Cls):
        # globally @app.cls-decorated class
        modal_obj: modal.cls.Obj = cls(*args, **kwargs)
        modal_obj._entered = True  # ugly but prevents .local() from triggering additional enter-logic
        # TODO: unify lifecycle logic between .local() and container_entrypoint
        user_cls_instance = modal_obj._cached_user_cls_instance()
    else:
        # undecorated class (non-global decoration or serialized)
        user_cls_instance = cls(*args, **kwargs)

    return user_cls_instance


def import_single_function_service(
    function_def: api_pb2.Function,
    ser_cls,  # used only for @build functions
    ser_fun,
    cls_args,  #  used only for @build functions
    cls_kwargs,  #  used only for @build functions
) -> Service:
    """Imports a function dynamically, and locates the app.

    This is somewhat complex because we're dealing with 3 quite different type of functions:
    1. Functions defined in global scope and decorated in global scope (Function objects)
    2. Functions defined in global scope but decorated elsewhere (these will be raw callables)
    3. Serialized functions

    In addition, we also need to handle
    * Normal functions
    * Methods on classes (in which case we need to instantiate the object)

    This helper also handles web endpoints, ASGI/WSGI servers, and HTTP servers.

    In order to locate the app, we try two things:
    * If the function is a Function, we can get the app directly from it
    * Otherwise, use the app name and look it up from a global list of apps: this
      typically only happens in case 2 above, or in sometimes for case 3

    Note that `import_function` is *not* synchronized, because we need it to run on the main
    thread. This is so that any user code running in global scope (which executes as a part of
    the import) runs on the right thread.
    """
    user_defined_callable: Callable
    service_deps: Optional[Sequence["modal._object._Object"]] = None
    active_app: Optional[modal.app._App] = None

    if ser_fun is not None:
        # This is a serialized function we already fetched from the server
        cls, user_defined_callable = ser_cls, ser_fun
    else:
        # Load the module dynamically
        module = importlib.import_module(function_def.module_name)
        qual_name: str = function_def.function_name

        if not is_global_object(qual_name):
            raise LocalFunctionError("Attempted to load a function defined in a function scope")

        parts = qual_name.split(".")
        if len(parts) == 1:
            # This is a function
            cls = None
            f = getattr(module, qual_name)
            if isinstance(f, Function):
                function = synchronizer._translate_in(f)
                service_deps = function.deps(only_explicit_mounts=True)
                user_defined_callable = function.get_raw_f()
                active_app = function._app
            else:
                user_defined_callable = f
        elif len(parts) == 2:
            # This path should only be triggered by @build class builder methods and can be removed
            # once @build is deprecated.
            assert not function_def.use_method_name  # new "placeholder methods" should not be invoked directly!
            assert function_def.is_builder_function
            cls_name, fun_name = parts
            cls = getattr(module, cls_name)
            if isinstance(cls, modal.cls.Cls):
                # The cls decorator is in global scope
                _cls = synchronizer._translate_in(cls)
                user_defined_callable = _cls._callables[fun_name]
                # Intentionally not including these, since @build functions don't actually
                # forward the information from their parent class.
                # service_deps = _cls._get_class_service_function().deps(only_explicit_mounts=True)
                active_app = _cls._app
            else:
                # This is non-decorated class
                user_defined_callable = getattr(cls, fun_name)
        else:
            raise InvalidError(f"Invalid function qualname {qual_name}")

    # Instantiate the class if it's defined
    if cls:
        # This code is only used for @build methods on classes
        user_cls_instance = get_user_class_instance(cls, cls_args, cls_kwargs)
        # Bind the function to the instance as self (using the descriptor protocol!)
        user_defined_callable = user_defined_callable.__get__(user_cls_instance)
    else:
        user_cls_instance = None

    return ImportedFunction(
        user_cls_instance,
        active_app,
        service_deps,
        user_defined_callable,
    )


def import_class_service(
    function_def: api_pb2.Function,
    ser_cls,
    cls_args,
    cls_kwargs,
) -> Service:
    """
    This imports a full class to be able to execute any @method or webhook decorated methods.

    See import_function.
    """
    active_app: Optional["modal.app._App"]
    service_deps: Optional[Sequence["modal._object._Object"]]
    cls: typing.Union[type, modal.cls.Cls]

    if function_def.definition_type == api_pb2.Function.DEFINITION_TYPE_SERIALIZED:
        assert ser_cls is not None
        cls = ser_cls
    else:
        # Load the module dynamically
        module = importlib.import_module(function_def.module_name)
        qual_name: str = function_def.function_name

        if not is_global_object(qual_name):
            raise LocalFunctionError("Attempted to load a class defined in a function scope")

        parts = qual_name.split(".")
        if not (
            len(parts) == 2 and parts[1] == "*"
        ):  # the "function name" of a class service "function placeholder" is expected to be "ClassName.*"
            raise ExecutionError(
                f"Internal error: Invalid 'service function' identifier {qual_name}. Please contact Modal support"
            )

        assert not function_def.use_method_name  # new "placeholder methods" should not be invoked directly!
        cls_name = parts[0]
        cls = getattr(module, cls_name)

    if isinstance(cls, modal.cls.Cls):
        # The cls decorator is in global scope
        _cls = synchronizer._translate_in(cls)
        method_partials = _cls._get_partial_functions()
        service_function: _Function = _cls._class_service_function
        service_deps = service_function.deps(only_explicit_mounts=True)
        active_app = service_function.app
    else:
        # Undecorated user class - find all methods
        method_partials = _find_partial_methods_for_user_cls(cls, _PartialFunctionFlags.all())
        service_deps = None
        active_app = None

    user_cls_instance = get_user_class_instance(cls, cls_args, cls_kwargs)

    return ImportedClass(
        user_cls_instance,
        active_app,
        service_deps,
        # TODO (elias/deven): instead of using method_partials here we should use a set of api_pb2.MethodDefinition
        method_partials,
    )
