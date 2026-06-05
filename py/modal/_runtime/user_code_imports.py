# Copyright Modal Labs 2024
import importlib
import inspect
import os
import signal
import typing
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable, ContextManager, Generator, Sequence

import modal._object
import modal.cls
import modal.server
from modal import Function
from modal._functions import _Function
from modal._partial_function import (
    _find_callables_for_obj,
    _PartialFunctionFlags,
)
from modal._runtime.user_code_event_loop import UserCodeEventLoop
from modal._utils.async_utils import synchronizer
from modal._utils.function_utils import (
    LocalFunctionError,
    callable_has_non_self_params,
    is_async as get_is_async,
    is_global_object,
)
from modal.app import _App
from modal.config import logger
from modal.exception import ExecutionError, InvalidError
from modal.experimental.flash import _FlashContainerEntry
from modal_proto import api_pb2

if typing.TYPE_CHECKING:
    import modal._functions
    import modal._partial_function
    import modal._runtime.container_io_manager
    import modal._runtime.task_lifecycle_manager
    import modal.app
    from modal._runtime.asgi import LifespanManager


@dataclass
class FinalizedFunction:
    callable: Callable[..., Any]
    is_async: bool
    is_generator: bool
    supported_output_formats: Sequence["api_pb2.DataFormat.ValueType"]
    lifespan_manager: "LifespanManager | None" = None


def call_lifecycle_functions(
    event_loop: UserCodeEventLoop,
    task_lifecycle_manager: "modal._runtime.task_lifecycle_manager.TaskLifecycleManager",
    funcs: Sequence[Callable[..., Any]],
) -> None:
    """Call function(s), can be sync or async, but any return values are ignored."""
    with task_lifecycle_manager.handle_task_lifecycle_exception():
        for func in funcs:
            # We are deprecating parametrized exit methods but want to gracefully handle old code.
            args = (None, None, None) if callable_has_non_self_params(func) else ()
            res = func(*args)
            if inspect.iscoroutine(res):
                event_loop.run(res)


@contextmanager
def lifecycle_asgi(
    event_loop: UserCodeEventLoop,
    task_lifecycle_manager: "modal._runtime.task_lifecycle_manager.TaskLifecycleManager",
    finalized_functions: dict[str, FinalizedFunction],
) -> Generator[None, None, None]:
    lifespan_background_tasks = []
    try:
        for finalized_function in finalized_functions.values():
            if finalized_function.lifespan_manager:
                lifespan_background_tasks.append(
                    event_loop.create_task(finalized_function.lifespan_manager.background_task())
                )
                with task_lifecycle_manager.handle_task_lifecycle_exception():
                    event_loop.run(finalized_function.lifespan_manager.lifespan_startup())
        yield
    finally:
        try:
            # run lifespan shutdown for asgi apps
            for finalized_function in finalized_functions.values():
                if finalized_function.lifespan_manager:
                    with task_lifecycle_manager.handle_task_lifecycle_exception():
                        event_loop.run(finalized_function.lifespan_manager.lifespan_shutdown())
        finally:
            # no need to keep the lifespan asgi call around - we send it no more messages
            for task in lifespan_background_tasks:
                task.cancel()


def disable_signals():
    int_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    usr1_handler = signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    return int_handler, usr1_handler


def try_enable_signals(int_handler, usr1_handler):
    if int_handler is not None and usr1_handler is not None:
        signal.signal(signal.SIGINT, int_handler)
        signal.signal(signal.SIGUSR1, usr1_handler)


def volume_commit(
    task_lifecycle_manager: "modal._runtime.task_lifecycle_manager.TaskLifecycleManager",
    function_def: api_pb2.Function,
):
    task_lifecycle_manager.volume_commit(
        [v.volume_id for v in function_def.volume_mounts if v.allow_background_commits],
    )


class Service(metaclass=ABCMeta):
    """Common interface for singular functions and class-based "services"

    There are differences in the importing/finalization logic, and this
    "protocol"/abc basically defines a common interface for the two types
    of "Services" after the point of import.
    """

    user_cls_instance: Any
    app: "modal.app._App"
    service_deps: Sequence["modal._object._Object"] | None
    function_def: api_pb2.Function

    @abstractmethod
    def get_finalized_functions(
        self,
        fun_def: api_pb2.Function,
        container_io_manager: "modal._runtime.container_io_manager.ContainerIOManager",
    ) -> dict[str, "FinalizedFunction"]: ...

    @contextmanager
    def lifecycle_presnapshot(
        self,
        event_loop: UserCodeEventLoop,
        task_lifecycle_manager: "modal._runtime.task_lifecycle_manager.TaskLifecycleManager",
    ) -> Generator[None, None, None]:
        # Default no-op implementation for services without pre-snapshot lifecycle handling
        yield

    @contextmanager
    def lifecycle_postsnapshot(
        self,
        event_loop: UserCodeEventLoop,
        task_lifecycle_manager: "modal._runtime.task_lifecycle_manager.TaskLifecycleManager",
    ) -> Generator[None, None, None]:
        # Default no-op implementation for services without post-snapshot lifecycle handling
        yield

    @contextmanager
    def lifecycle_context(
        self,
        event_loop: UserCodeEventLoop,
        task_lifecycle_manager: "modal._runtime.task_lifecycle_manager.TaskLifecycleManager",
        snapshot_context_manager: ContextManager[None] = nullcontext(),
        after_snapshot: Callable[[], None] | None = None,
        disable_signals_on_exit: bool = True,
    ) -> Generator[None, None, None]:
        """
        Manages the lifecycle of the user code:
        1. Runs pre-snapshot 'enter' methods
        2. Calls maybe_snapshot(function_def, snapshot_context_manager, task_lifecycle_manager)
        3. Creates breakpoint wrapper
        4. Runs post-snapshot 'enter' methods
        5. Yield finalized_functions for execution
        6. Handles cleanup (lifespan shutdown, 'exit' methods)
        7. Disable signals
        8. Volume commit
        """
        int_handler, usr1_handler = None, None
        try:
            # 1. Pre-snapshot Enter
            with self.lifecycle_presnapshot(event_loop, task_lifecycle_manager):
                # 2. Snapshot -- If this container is being used to create a checkpoint, checkpoint the container after
                # global imports and initialization. Checkpointed containers run from this point onwards.
                maybe_snapshot(self.function_def, snapshot_context_manager, task_lifecycle_manager)
                # 3. After snapshot functionality like create_breakpoint_wrapper(container_io_manager)
                if after_snapshot:
                    after_snapshot()
                # 4. Post-snapshot Enter
                with self.lifecycle_postsnapshot(event_loop, task_lifecycle_manager):
                    # 5. Yield
                    try:
                        yield
                    finally:
                        if disable_signals_on_exit:
                            # 7. Disable signals
                            int_handler, usr1_handler = disable_signals()
        finally:
            # 8. Volume commit - runs OUTSIDE all lifecycle managers so exit handlers
            # have a chance to write to disk before we commit volumes
            try:
                volume_commit(task_lifecycle_manager, self.function_def)
            finally:
                try_enable_signals(int_handler, usr1_handler)

    @contextmanager
    def function_execution_context(
        self,
        event_loop: UserCodeEventLoop,
        container_io_manager: "modal._runtime.container_io_manager.ContainerIOManager",
    ) -> Generator[dict[str, "FinalizedFunction"], None, None]:
        """
        Handles the execution of user code for functions:
        1. Initializes finalized functions (and ASGI/WSGI lifespan)
        2. Starts ASGI/WSGI lifespans

        """
        task_lifecycle_manager = container_io_manager.get_task_lifecycle_manager()
        int_handler, usr1_handler = None, None
        try:
            with self.lifecycle_context(
                event_loop=event_loop,
                task_lifecycle_manager=task_lifecycle_manager,
                snapshot_context_manager=container_io_manager.snapshot_context_manager(),
                after_snapshot=container_io_manager._install_breakpoint_hook,
                disable_signals_on_exit=False,
            ):
                # Get Functions
                with task_lifecycle_manager.handle_task_lifecycle_exception():
                    finalized_functions = self.get_finalized_functions(self.function_def, container_io_manager)
                # 1. Start ASGI lifespan
                with lifecycle_asgi(event_loop, task_lifecycle_manager, finalized_functions):
                    # 2. Yield Finalized Functions
                    try:
                        yield finalized_functions
                    finally:
                        int_handler, usr1_handler = disable_signals()
        finally:
            try_enable_signals(int_handler, usr1_handler)


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
        raise InvalidError(f"Unrecognized Web Function type {webhook_config.type}")


def maybe_snapshot(
    function_def: api_pb2.Function,
    snapshot_context_manager: ContextManager[None],
    task_lifecycle_manager: "modal._runtime.task_lifecycle_manager.TaskLifecycleManager",
):
    if function_def.is_checkpointing_function and os.environ.get("MODAL_ENABLE_SNAP_RESTORE") == "1":
        with snapshot_context_manager:
            task_lifecycle_manager.memory_snapshot()


@dataclass
class ImportedFunction(Service):
    app: modal.app._App
    service_deps: Sequence["modal._object._Object"] | None
    user_cls_instance = None
    function_def: api_pb2.Function

    _user_defined_callable: Callable[..., Any]

    def get_finalized_functions(
        self,
        fun_def: api_pb2.Function,
        container_io_manager: "modal._runtime.container_io_manager.ContainerIOManager",
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
                    supported_output_formats=fun_def.supported_output_formats
                    # FIXME (elias): the following `or [api_pb2.DATA_FORMAT_PICKLE, api_pb2.DATA_FORMAT_CBOR]` is only
                    # needed for tests
                    or [api_pb2.DATA_FORMAT_PICKLE, api_pb2.DATA_FORMAT_CBOR],
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
                # FIXME (elias): the following `or [api_pb2.DATA_FORMAT_ASGI]` is only needed for tests
                supported_output_formats=fun_def.supported_output_formats or [api_pb2.DATA_FORMAT_ASGI],
            )
        }


class _LifecycleManager:
    """Lifecycle manager for class-based services (Cls and Server).

    Handles pre snapshot, post snapshot, and exit lifecycle handling
    """

    user_cls_instance: Any
    function_def: api_pb2.Function

    @contextmanager
    def lifecycle_presnapshot(
        self,
        event_loop: UserCodeEventLoop,
        task_lifecycle_manager: "modal._runtime.task_lifecycle_manager.TaskLifecycleManager",
    ):
        # Identify all "enter" methods that need to run before we snapshot.
        if not self.function_def.is_auto_snapshot:
            pre_snapshot_methods = _find_callables_for_obj(
                self.user_cls_instance, _PartialFunctionFlags.ENTER_PRE_SNAPSHOT
            )
            call_lifecycle_functions(event_loop, task_lifecycle_manager, list(pre_snapshot_methods.values()))
        yield

    @contextmanager
    def lifecycle_postsnapshot(
        self,
        event_loop: UserCodeEventLoop,
        task_lifecycle_manager: "modal._runtime.task_lifecycle_manager.TaskLifecycleManager",
    ):
        # Identify the "enter" methods to run after resuming from a snapshot.
        flash_entry = _FlashContainerEntry(self.function_def.http_config, is_server=self.function_def.is_server)
        if not self.function_def.is_auto_snapshot:
            post_snapshot_methods = _find_callables_for_obj(
                self.user_cls_instance, _PartialFunctionFlags.ENTER_POST_SNAPSHOT
            )
            call_lifecycle_functions(event_loop, task_lifecycle_manager, list(post_snapshot_methods.values()))
            flash_entry.enter()
        try:
            yield
        finally:
            if not self.function_def.is_auto_snapshot:
                flash_entry.stop()
                exit_methods = _find_callables_for_obj(self.user_cls_instance, _PartialFunctionFlags.EXIT)
                call_lifecycle_functions(event_loop, task_lifecycle_manager, list(exit_methods.values()))
                flash_entry.close()


@dataclass
class ImportedClass(_LifecycleManager, Service):
    user_cls_instance: Any
    app: "modal.app._App"
    service_deps: Sequence["modal._object._Object"] | None

    _partial_functions: dict[str, "modal._partial_function._PartialFunction"]
    function_def: api_pb2.Function

    def get_finalized_functions(
        self,
        fun_def: api_pb2.Function,
        container_io_manager: "modal._runtime.container_io_manager.ContainerIOManager",
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
            method_def = fun_def.method_definitions[method_name]

            bound_func = user_func.__get__(self.user_cls_instance)

            if not webhook_config or webhook_config.type == api_pb2.WEBHOOK_TYPE_UNSPECIFIED:
                # for non-webhooks, the runnable is straight forward:
                finalized_function = FinalizedFunction(
                    callable=bound_func,
                    is_async=is_async,
                    is_generator=bool(is_generator),
                    # FIXME (elias): the following `or [api_pb2.DATA_FORMAT_PICKLE, api_pb2.DATA_FORMAT_CBOR]` is only
                    # needed for tests
                    supported_output_formats=method_def.supported_output_formats
                    or [api_pb2.DATA_FORMAT_PICKLE, api_pb2.DATA_FORMAT_CBOR],
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
                    # FIXME (elias): the following `or [api_pb2.DATA_FORMAT_ASGI]` is only needed for tests
                    supported_output_formats=method_def.supported_output_formats or [api_pb2.DATA_FORMAT_ASGI],
                )
            finalized_functions[method_name] = finalized_function
        return finalized_functions


@dataclass
class ImportedServer(_LifecycleManager, Service):
    user_cls_instance: Any
    app: "modal.app._App"
    service_deps: Sequence["modal._object._Object"] | None
    function_def: api_pb2.Function

    def get_finalized_functions(
        self,
        fun_def: api_pb2.Function,
        container_io_manager: "modal._runtime.container_io_manager.ContainerIOManager",
    ) -> dict[str, "FinalizedFunction"]:
        return {}


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
    ser_fun: Callable[..., Any] | None,
) -> Service:
    """Imports a function dynamically, and locates the app.

    This is somewhat complex because we're dealing with 3 quite different type of functions:
    1. Functions defined in global scope and decorated in global scope (Function objects)
    2. Functions defined in global scope but decorated elsewhere (these will be raw callables)
    3. Serialized functions

    In addition, we also need to handle
    * Normal functions
    * Methods on classes (in which case we need to instantiate the object)

    This helper also handles Web Functions (fastapi_endpoint, asgi_app, wsgi_app, web_server).

    In order to locate the app, we try two things:
    * If the function is a Function, we can get the app directly from it
    * Otherwise, use the app name and look it up from a global list of apps: this
      typically only happens in case 2 above, or in sometimes for case 3

    Note that `import_function` is *not* synchronized, because we need it to run on the main
    thread. This is so that any user code running in global scope (which executes as a part of
    the import) runs on the right thread.
    """
    user_defined_callable: Callable
    service_deps: Sequence["modal._object._Object"] | None = None
    active_app: modal.app._App

    if ser_fun is not None:
        # This is a serialized function we already fetched from the server
        user_defined_callable = ser_fun
        active_app = get_active_app_fallback(function_def)
    else:
        # Load the module dynamically
        module = importlib.import_module(function_def.module_name)

        # Fall back to function_name just to be safe around the migration
        # Going forward, implementation_name should always be set
        qual_name: str = function_def.implementation_name or function_def.function_name

        if not is_global_object(qual_name):
            raise LocalFunctionError("Attempted to load a function defined in a function scope")

        parts = qual_name.split(".")
        if len(parts) != 1:
            raise InvalidError(f"Invalid function qualname {qual_name}")

        f = getattr(module, qual_name)
        if isinstance(f, Function):
            _function: modal._functions._Function[Any, Any, Any] = synchronizer._translate_in(f)  # type: ignore
            service_deps = _function.deps(only_explicit_mounts=True)
            user_defined_callable = _function.get_raw_f()
            assert _function._app  # app should always be set on a decorated function
            active_app = _function._app
        else:
            # function isn't decorated in global scope
            user_defined_callable = f
            active_app = get_active_app_fallback(function_def)

    return ImportedFunction(
        app=active_app,
        service_deps=service_deps,
        function_def=function_def,
        _user_defined_callable=user_defined_callable,
    )


def _get_cls_or_user_cls(
    function_def: api_pb2.Function,
    ser_user_cls: type | None,
) -> type | modal.cls.Cls | modal.server.Server:
    if function_def.definition_type == api_pb2.Function.DEFINITION_TYPE_SERIALIZED:
        assert ser_user_cls is not None
        cls_or_user_cls = ser_user_cls
    else:
        # Load the module dynamically for non-serialized class.
        module = importlib.import_module(function_def.module_name)
        qual_name: str = function_def.function_name

        if not is_global_object(qual_name):
            raise LocalFunctionError("Attempted to load a class defined in a function scope")

        parts = qual_name.split(".")
        # Class service functions have pattern "ClassName.*", servers use "ClassName"
        if not (len(parts) == 1 or (len(parts) == 2 and parts[1] == "*")):
            raise ExecutionError(
                f"Internal error: Invalid 'service function' identifier {qual_name}. Please contact Modal support"
            )

        assert not function_def.use_method_name  # new "placeholder methods" should not be invoked directly!
        cls_name = parts[0]
        cls_or_user_cls = getattr(module, cls_name)
    return cls_or_user_cls


def import_class_service(
    function_def: api_pb2.Function,
    service_function_hydration_data: api_pb2.Object,
    class_id: str,
    _client: "modal.client._Client",
    ser_user_cls: type | None,
    cls_args,
    cls_kwargs,
) -> Service:
    """
    This imports a full class to be able to execute any @method or webhook decorated methods.

    See import_function.
    """
    active_app: "modal.app._App | None"
    service_deps: Sequence["modal._object._Object"] | None
    cls_or_user_cls: type | modal.cls.Cls

    cls_or_user_cls = typing.cast(
        type | modal.cls.Cls,
        _get_cls_or_user_cls(function_def, ser_user_cls),
    )

    if isinstance(cls_or_user_cls, modal.cls.Cls):
        _cls = typing.cast(modal.cls._Cls, synchronizer._translate_in(cls_or_user_cls))
        class_service_function: _Function = _cls._get_class_service_function()
        service_deps = class_service_function.deps(only_explicit_mounts=True)
        active_app = class_service_function.app
    else:
        # Undecorated user class (serialized or local scope-decoration).
        service_deps = None  # we can't infer service deps for now
        active_app = get_active_app_fallback(function_def)
        _service_function: modal._functions._Function[Any, Any, Any] = modal._functions._Function._new_hydrated(
            service_function_hydration_data.object_id,
            _client,
            service_function_hydration_data.function_handle_metadata,
            skip_reload=True,  # this skips re-loading the function, which is required since it doesn't have a loader
        )
        _cls = modal.cls._Cls.from_local(cls_or_user_cls, active_app, _service_function)
        # hydration of the class itself - just sets the id and triggers some side effects
        # that transfers metadata from the service function to the class. TODO: cleanup!
        _cls._hydrate(class_id, _client, api_pb2.ClassHandleMetadata())

    method_partials: dict[str, "modal._partial_function._PartialFunction"] = _cls._get_partial_functions()
    user_cls_instance = get_user_class_instance(_cls, cls_args, cls_kwargs)

    return ImportedClass(
        user_cls_instance=user_cls_instance,
        app=active_app,
        service_deps=service_deps,
        # TODO (elias/deven): instead of using method_partials here we should use a set of api_pb2.MethodDefinition
        _partial_functions=method_partials,
        function_def=function_def,
    )


def import_server_service(
    function_def: api_pb2.Function,
    service_function_hydration_data: api_pb2.Object,
    _client: "modal.client._Client",
    ser_user_cls: type | None,
) -> Service:
    """
    This imports a class as a server to server HTTP requests.

    See import_function.
    """
    active_app: "modal.app._App | None"
    service_deps: Sequence["modal._object._Object"] | None
    cls_or_user_cls: type | modal.server.Server

    cls_or_user_cls = typing.cast(
        type | modal.server.Server,
        _get_cls_or_user_cls(function_def, ser_user_cls),
    )

    if isinstance(cls_or_user_cls, modal.server.Server):
        _server = typing.cast(modal._server._Server, synchronizer._translate_in(cls_or_user_cls))
        server_service_function: _Function = _server._get_service_function()
        service_deps = server_service_function.deps(only_explicit_mounts=True)
        active_app = _server._get_app()

    else:
        # Undecorated user class (serialized or local scope-decoration).
        service_deps = None  # we can't infer service deps for now
        active_app = get_active_app_fallback(function_def)
        _service_function: modal._functions._Function[Any, Any, Any] = modal._functions._Function._new_hydrated(
            service_function_hydration_data.object_id,
            _client,
            service_function_hydration_data.function_handle_metadata,
            skip_reload=True,  # this skips re-loading the function, which is required since it doesn't have a loader
        )

        _server = modal._server._Server._from_local(cls_or_user_cls, active_app, _service_function)

    user_cls = _server._get_user_cls()
    # Create server object with lifecycle methods registered.
    return ImportedServer(
        user_cls_instance=user_cls(),
        app=active_app,
        service_deps=service_deps,
        function_def=function_def,
    )


def get_active_app_fallback(function_def: api_pb2.Function) -> _App:
    # This branch is reached in the special case that the imported function/class is:
    # 1) not serialized, and
    # 2) isn't a FunctionHandle - i.e, not decorated at definition time
    # Look at all instantiated apps - if there is only one with the indicated name, use that one
    app_name: str | None = function_def.app_name or None  # coalesce protobuf field to None
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
