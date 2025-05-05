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
from modal._utils.async_utils import synchronizer
from modal._utils.function_utils import LocalFunctionError, is_async as get_is_async, is_global_object
from modal.app import _App
from modal.config import logger
from modal.exception import ExecutionError, InvalidError
from modal_proto import api_pb2

if typing.TYPE_CHECKING:
    import modal._functions
    import modal._partial_function
    import modal.app
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
    app: "modal.app._App"
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
    app: modal.app._App
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
    app: "modal.app._App"
    service_deps: Optional[Sequence["modal._object._Object"]]

    _partial_functions: dict[str, "modal._partial_function._PartialFunction"]

    def get_finalized_functions(
        self, fun_def: api_pb2.Function, container_io_manager: "modal._runtime.container_io_manager.ContainerIOManager"
    ) -> dict[str, "FinalizedFunction"]:
        finalized_functions = {}
        for method_name, _partial in self._partial_functions.items():
            user_func = _partial.raw_f
            assert user_func
            # Check this property before we turn it into a method (overriden by webhooks)
            is_async = get_is_async(user_func)
            # Use the function definition for whether this is a generator (overriden by webhooks)
            is_generator = _partial.params.is_generator
            webhook_config = _partial.params.webhook_config

            bound_func = user_func.__get__(self.user_cls_instance)

            if not webhook_config or webhook_config.type == api_pb2.WEBHOOK_TYPE_UNSPECIFIED:
                # for non-webhooks, the runnable is straight forward:
                finalized_function = FinalizedFunction(
                    callable=bound_func,
                    is_async=is_async,
                    is_generator=bool(is_generator),
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


def get_user_class_instance(_cls: modal.cls._Cls, args: tuple[Any, ...], kwargs: dict[str, Any]) -> typing.Any:
    """Returns instance of the underlying class to be used as the `self`

    For the time being, this is an instance of the underlying user defined type, with
    some extra attributes like parameter values and _modal_functions set, allowing
    its methods to be used as modal Function objects with .remote() and .local() etc.

    TODO: Could possibly change this to use an Obj to clean up the data model? would invalidate isinstance checks though
    """
    cls = typing.cast(modal.cls.Cls, synchronizer._translate_out(_cls))  # ugly
    modal_obj: modal.cls.Obj = cls(*args, **kwargs)
    modal_obj._entered = True  # ugly but prevents .local() from triggering additional enter-logic
    # TODO: unify lifecycle logic between .local() and container_entrypoint
    user_cls_instance = modal_obj._cached_user_cls_instance()
    return user_cls_instance


def import_single_function_service(
    function_def: api_pb2.Function,
    ser_cls: Optional[type],  # used only for @build functions
    ser_fun: Optional[Callable[..., Any]],
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
    active_app: modal.app._App

    user_cls_or_cls: typing.Union[None, type, modal.cls.Cls]
    user_cls_instance = None

    if ser_fun is not None:
        # This is a serialized function we already fetched from the server
        user_cls_or_cls, user_defined_callable = ser_cls, ser_fun
        active_app = get_active_app_fallback(function_def)
    else:
        # Load the module dynamically
        module = importlib.import_module(function_def.module_name)
        qual_name: str = function_def.function_name

        if not is_global_object(qual_name):
            raise LocalFunctionError("Attempted to load a function defined in a function scope")

        parts = qual_name.split(".")
        if len(parts) == 1:
            # This is a function
            user_cls_or_cls = None
            f = getattr(module, qual_name)
            if isinstance(f, Function):
                _function: modal._functions._Function[Any, Any, Any] = synchronizer._translate_in(f)  # type: ignore
                service_deps = _function.deps(only_explicit_mounts=True)
                user_defined_callable = _function.get_raw_f()
                assert _function._app  # app should always be set on a decorated function
                active_app = _function._app
            else:
                user_defined_callable = f
                active_app = get_active_app_fallback(function_def)

        elif len(parts) == 2:
            # This path should only be triggered by @build class builder methods and can be removed
            # once @build is deprecated.
            assert not function_def.use_method_name  # new "placeholder methods" should not be invoked directly!
            assert function_def.is_builder_function
            cls_name, fun_name = parts
            user_cls_or_cls = getattr(module, cls_name)
            if isinstance(user_cls_or_cls, modal.cls.Cls):
                # The cls decorator is in global scope
                _cls = typing.cast(modal.cls._Cls, synchronizer._translate_in(user_cls_or_cls))
                user_defined_callable = _cls._callables[fun_name]
                # Intentionally not including these, since @build functions don't actually
                # forward the information from their parent class.
                # service_deps = _cls._get_class_service_function().deps(only_explicit_mounts=True)
                assert _cls._app
                active_app = _cls._app
            else:
                # This is non-decorated class
                user_defined_callable = getattr(user_cls_or_cls, fun_name)  # unbound method
                active_app = get_active_app_fallback(function_def)
        else:
            raise InvalidError(f"Invalid function qualname {qual_name}")

    # Instantiate the class if it's defined
    if user_cls_or_cls:
        if isinstance(user_cls_or_cls, modal.cls.Cls):
            # This code is only used for @build methods on classes
            _cls = typing.cast(modal.cls._Cls, user_cls_or_cls)
            user_cls_instance = get_user_class_instance(_cls, (), {})
            # Bind the unbound method to the instance as self (using the descriptor protocol!)
        else:
            # serialized=True or "undecorated"
            user_cls_instance = user_cls_or_cls()

        user_defined_callable = user_defined_callable.__get__(user_cls_instance)

    return ImportedFunction(
        user_cls_instance,
        active_app,
        service_deps,
        user_defined_callable,
    )


def import_class_service(
    function_def: api_pb2.Function,
    service_function_hydration_data: api_pb2.Object,
    class_id: str,
    client: "modal.client.Client",
    ser_user_cls: Optional[type],
    cls_args,
    cls_kwargs,
) -> Service:
    """
    This imports a full class to be able to execute any @method or webhook decorated methods.

    See import_function.
    """
    active_app: Optional["modal.app._App"]
    service_deps: Optional[Sequence["modal._object._Object"]]
    cls_or_user_cls: typing.Union[type, modal.cls.Cls]

    if function_def.definition_type == api_pb2.Function.DEFINITION_TYPE_SERIALIZED:
        assert ser_user_cls is not None
        cls_or_user_cls = ser_user_cls
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
        cls_or_user_cls = getattr(module, cls_name)

    if isinstance(cls_or_user_cls, modal.cls.Cls):
        _cls = typing.cast(modal.cls._Cls, synchronizer._translate_in(cls_or_user_cls))
        class_service_function: _Function = _cls._get_class_service_function()
        service_deps = class_service_function.deps(only_explicit_mounts=True)
        active_app = class_service_function.app
    else:
        # Undecorated user class (serialized or local scope-decoration).
        service_deps = None  # we can't infer service deps for now
        active_app = get_active_app_fallback(function_def)
        _client = typing.cast("modal.client._Client", synchronizer._translate_in(client))
        _service_function: modal._functions._Function[Any, Any, Any] = modal._functions._Function._new_hydrated(
            service_function_hydration_data.object_id,
            _client,
            service_function_hydration_data.function_handle_metadata,
            is_another_app=True,  # this skips re-loading the function, which is required since it doesn't have a loader
        )
        _cls = modal.cls._Cls.from_local(cls_or_user_cls, active_app, _service_function)
        # hydration of the class itself - just sets the id and triggers some side effects
        # that transfers metadata from the service function to the class. TODO: cleanup!
        _cls._hydrate(class_id, _client, api_pb2.ClassHandleMetadata())

    method_partials: dict[str, "modal._partial_function._PartialFunction"] = _cls._get_partial_functions()
    user_cls_instance = get_user_class_instance(_cls, cls_args, cls_kwargs)

    return ImportedClass(
        user_cls_instance,
        active_app,
        service_deps,
        # TODO (elias/deven): instead of using method_partials here we should use a set of api_pb2.MethodDefinition
        method_partials,
    )


def get_active_app_fallback(function_def: api_pb2.Function) -> _App:
    # This branch is reached in the special case that the imported function/class is:
    # 1) not serialized, and
    # 2) isn't a FunctionHandle - i.e, not decorated at definition time
    # Look at all instantiated apps - if there is only one with the indicated name, use that one
    app_name: Optional[str] = function_def.app_name or None  # coalesce protobuf field to None
    matching_apps = _App._all_apps.get(app_name, [])
    if len(matching_apps) == 1:
        active_app: _App = matching_apps[0]
        return active_app

    if len(matching_apps) > 1:
        if app_name is not None:
            warning_sub_message = f"app with the same name ('{app_name}')"
        else:
            warning_sub_message = "unnamed app"
        logger.warning(
            f"You have more than one {warning_sub_message}. "
            "It's recommended to name all your Apps uniquely when using multiple apps"
        )

    # If we don't have an active app, create one on the fly
    # The app object is used to carry the app layout etc
    return _App()
