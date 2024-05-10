# Copyright Modal Labs 2022
import asyncio
import base64
import importlib
import inspect
import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Type

from google.protobuf.message import Message
from synchronicity import Interface

from modal_proto import api_pb2

from ._asgi import (
    asgi_app_wrapper,
    get_ip_address,
    wait_for_web_server,
    web_server_proxy,
    webhook_asgi_app,
    wsgi_app_wrapper,
)
from ._container_io_manager import ContainerIOManager, UserException, _ContainerIOManager
from ._proxy_tunnel import proxy_tunnel
from ._serialization import deserialize
from ._utils.async_utils import TaskContext, synchronizer
from ._utils.function_utils import (
    LocalFunctionError,
    is_async as get_is_async,
    is_global_function,
    method_has_params,
)
from .app import App, _App
from .client import Client, _Client
from .cls import Cls
from .config import logger
from .exception import ExecutionError, InputCancellation, InvalidError, deprecation_warning
from .execution_context import _set_current_context_ids, interact
from .functions import Function, _Function
from .partial_function import _find_callables_for_obj, _PartialFunctionFlags
from .running_app import RunningApp

if TYPE_CHECKING:
    from types import ModuleType


@dataclass
class ImportedFunction:
    obj: Any
    fun: Callable
    app: Optional[_App]
    is_async: bool
    is_generator: bool
    data_format: int  # api_pb2.DataFormat
    input_concurrency: int
    is_auto_snapshot: bool
    function: _Function


class UserCodeEventLoop:
    """Run an async event loop as a context manager and handle signals.

    This will run all *user supplied* async code, i.e. async functions, as well as async enter/exit managers

    The following signals are handled while a coroutine is running on the event loop until
    completion (and then handlers are deregistered):

    - `SIGUSR1`: converted to an async task cancellation. Note that this only affects the event
      loop, and the signal handler defined here doesn't run for sync functions.
    - `SIGINT`: Unless the global signal handler has been set to SIGIGN, the loop's signal handler
        is set to cancel the current task and raise KeyboardInterrupt to the caller.
    """

    def __enter__(self):
        self.loop = asyncio.new_event_loop()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.loop.run_until_complete(self.loop.shutdown_asyncgens())
        if sys.version_info[:2] >= (3, 9):
            self.loop.run_until_complete(self.loop.shutdown_default_executor())  # Introduced in Python 3.9
        self.loop.close()

    def run(self, coro):
        task = asyncio.ensure_future(coro, loop=self.loop)
        self._sigints = 0

        def _sigint_handler():
            # cancel the task in order to have run_until_complete return soon and
            # prevent a bunch of unwanted tracebacks when shutting down the
            # event loop.

            # this basically replicates the sigint handler installed by asyncio.run()
            self._sigints += 1
            if self._sigints == 1:
                # first sigint is graceful
                task.cancel()
                return
            raise KeyboardInterrupt()  # this should normally not happen, but the second sigint would "hard kill" the event loop!

        ignore_sigint = signal.getsignal(signal.SIGINT) == signal.SIG_IGN
        if not ignore_sigint:
            self.loop.add_signal_handler(signal.SIGINT, _sigint_handler)

        # Before Python 3.9 there is no argument to Task.cancel
        if sys.version_info[:2] >= (3, 9):
            self.loop.add_signal_handler(signal.SIGUSR1, task.cancel, "Input was cancelled by user")
        else:
            self.loop.add_signal_handler(signal.SIGUSR1, task.cancel)

        try:
            return self.loop.run_until_complete(task)
        except asyncio.CancelledError:
            if self._sigints > 0:
                raise KeyboardInterrupt()
        finally:
            self.loop.remove_signal_handler(signal.SIGUSR1)
            if not ignore_sigint:
                self.loop.remove_signal_handler(signal.SIGINT)


def call_function_sync(
    container_io_manager,  #: ContainerIOManager,  TODO: this type is generated at runtime
    imp_fun: ImportedFunction,
):
    def run_input(input_id: str, function_call_id: str, args: Any, kwargs: Any) -> None:
        started_at = time.time()
        reset_context = _set_current_context_ids(input_id, function_call_id)
        with container_io_manager.handle_input_exception(input_id, started_at):
            logger.debug(f"Starting input {input_id} (sync)")
            res = imp_fun.fun(*args, **kwargs)
            logger.debug(f"Finished input {input_id} (sync)")

            # TODO(erikbern): any exception below shouldn't be considered a user exception
            if imp_fun.is_generator:
                if not inspect.isgenerator(res):
                    raise InvalidError(f"Generator function returned value of type {type(res)}")

                # Send up to this many outputs at a time.
                generator_queue: asyncio.Queue[Any] = container_io_manager._queue_create(1024)
                generator_output_task = container_io_manager.generator_output_task(
                    function_call_id,
                    imp_fun.data_format,
                    generator_queue,
                    _future=True,  # Synchronicity magic to return a future.
                )

                item_count = 0
                for value in res:
                    container_io_manager._queue_put(generator_queue, value)
                    item_count += 1

                container_io_manager._queue_put(generator_queue, _ContainerIOManager._GENERATOR_STOP_SENTINEL)
                generator_output_task.result()  # Wait to finish sending generator outputs.
                message = api_pb2.GeneratorDone(items_total=item_count)
                container_io_manager.push_output(input_id, started_at, message, api_pb2.DATA_FORMAT_GENERATOR_DONE)
            else:
                if inspect.iscoroutine(res) or inspect.isgenerator(res) or inspect.isasyncgen(res):
                    raise InvalidError(
                        f"Sync (non-generator) function return value of type {type(res)}."
                        " You might need to use @app.function(..., is_generator=True)."
                    )
                container_io_manager.push_output(input_id, started_at, res, imp_fun.data_format)
        reset_context()

    if imp_fun.input_concurrency > 1:
        # We can't use `concurrent.futures.ThreadPoolExecutor` here because in Python 3.11+, this
        # class has no workaround that allows us to exit the Python interpreter process without
        # waiting for the worker threads to finish. We need this behavior on SIGINT.

        import queue
        import threading

        spawned_workers = 0
        inputs: queue.Queue[Any] = queue.Queue()
        finished = threading.Event()

        def worker_thread():
            while not finished.is_set():
                try:
                    args = inputs.get(timeout=1)
                except queue.Empty:
                    continue
                try:
                    run_input(*args)
                except BaseException:
                    # This should basically never happen, since only KeyboardInterrupt is the only error that can
                    # bubble out of from handle_input_exception and those wouldn't be raised outside the main thread
                    pass
                inputs.task_done()

        for input_id, function_call_id, args, kwargs in container_io_manager.run_inputs_outputs(
            imp_fun.input_concurrency
        ):
            if spawned_workers < imp_fun.input_concurrency:
                threading.Thread(target=worker_thread, daemon=True).start()
                spawned_workers += 1
            inputs.put((input_id, function_call_id, args, kwargs))

        finished.set()
        inputs.join()

    else:
        for input_id, function_call_id, args, kwargs in container_io_manager.run_inputs_outputs(
            imp_fun.input_concurrency
        ):
            try:
                run_input(input_id, function_call_id, args, kwargs)
            except:
                raise


async def call_function_async(
    container_io_manager,  #: ContainerIOManager,  TODO: this type is generated at runtime
    imp_fun: ImportedFunction,
):
    async def run_input(input_id: str, function_call_id: str, args: Any, kwargs: Any) -> None:
        started_at = time.time()
        reset_context = _set_current_context_ids(input_id, function_call_id)
        async with container_io_manager.handle_input_exception.aio(input_id, started_at):
            logger.debug(f"Starting input {input_id} (async)")
            res = imp_fun.fun(*args, **kwargs)
            logger.debug(f"Finished input {input_id} (async)")

            # TODO(erikbern): any exception below shouldn't be considered a user exception
            if imp_fun.is_generator:
                if not inspect.isasyncgen(res):
                    raise InvalidError(f"Async generator function returned value of type {type(res)}")

                # Send up to this many outputs at a time.
                generator_queue: asyncio.Queue[Any] = await container_io_manager._queue_create.aio(1024)
                generator_output_task = asyncio.create_task(
                    container_io_manager.generator_output_task.aio(
                        function_call_id,
                        imp_fun.data_format,
                        generator_queue,
                    )
                )

                item_count = 0
                async for value in res:
                    await container_io_manager._queue_put.aio(generator_queue, value)
                    item_count += 1

                await container_io_manager._queue_put.aio(generator_queue, _ContainerIOManager._GENERATOR_STOP_SENTINEL)
                await generator_output_task  # Wait to finish sending generator outputs.
                message = api_pb2.GeneratorDone(items_total=item_count)
                await container_io_manager.push_output.aio(
                    input_id, started_at, message, api_pb2.DATA_FORMAT_GENERATOR_DONE
                )
            else:
                if not inspect.iscoroutine(res) or inspect.isgenerator(res) or inspect.isasyncgen(res):
                    raise InvalidError(
                        f"Async (non-generator) function returned value of type {type(res)}"
                        " You might need to use @app.function(..., is_generator=True)."
                    )
                value = await res
                await container_io_manager.push_output.aio(input_id, started_at, value, imp_fun.data_format)
        reset_context()

    if imp_fun.input_concurrency > 1:
        # all run_input coroutines will have completed by the time we leave the execution context
        # but the wrapping *tasks* may not yet have been resolved, so we add a 0.01s
        # for them to resolve gracefully:
        async with TaskContext(0.01) as task_context:
            async for input_id, function_call_id, args, kwargs in container_io_manager.run_inputs_outputs.aio(
                imp_fun.input_concurrency
            ):
                # Note that run_inputs_outputs will not return until the concurrency semaphore has
                # released all its slots so that they can be acquired by the run_inputs_outputs finalizer
                # This prevents leaving the task_context before outputs have been created
                # TODO: refactor to make this a bit more easy to follow?
                task_context.create_task(run_input(input_id, function_call_id, args, kwargs))
    else:
        async for input_id, function_call_id, args, kwargs in container_io_manager.run_inputs_outputs.aio(
            imp_fun.input_concurrency
        ):
            await run_input(input_id, function_call_id, args, kwargs)


def import_function(
    function_def: api_pb2.Function,
    ser_cls,
    ser_fun,
    ser_params: Optional[bytes],
    container_io_manager,
    client: Client,
) -> ImportedFunction:
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

    Note that `import_function` is *not* synchronized, becase we need it to run on the main
    thread. This is so that any user code running in global scope (which executes as a part of
    the import) runs on the right thread.
    """
    module: Optional[ModuleType] = None
    cls: Optional[Type] = None
    fun: Callable
    function: Optional[_Function] = None
    active_app: Optional[_App] = None
    pty_info: api_pb2.PTYInfo = function_def.pty_info

    if ser_fun is not None:
        # This is a serialized function we already fetched from the server
        cls, fun = ser_cls, ser_fun
    else:
        # Load the module dynamically
        module = importlib.import_module(function_def.module_name)
        qual_name: str = function_def.function_name

        if not is_global_function(qual_name):
            raise LocalFunctionError("Attempted to load a function defined in a function scope")

        parts = qual_name.split(".")
        if len(parts) == 1:
            # This is a function
            cls = None
            f = getattr(module, qual_name)
            if isinstance(f, Function):
                function = synchronizer._translate_in(f)
                fun = function.get_raw_f()
                active_app = function._app
            else:
                fun = f
        elif len(parts) == 2:
            # This is a method on a class
            cls_name, fun_name = parts
            cls = getattr(module, cls_name)
            if isinstance(cls, Cls):
                # The cls decorator is in global scope
                _cls = synchronizer._translate_in(cls)
                fun = _cls._callables[fun_name]
                function = _cls._functions.get(fun_name)
                active_app = _cls._app
            else:
                # This is a raw class
                fun = getattr(cls, fun_name)
        else:
            raise InvalidError(f"Invalid function qualname {qual_name}")

    # If the cls/function decorator was applied in local scope, but the app is global, we can look it up
    if active_app is None:
        # This branch is reached in the special case that the imported function is 1) not serialized, and 2) isn't a FunctionHandle - i.e, not decorated at definition time
        # Look at all instantiated apps - if there is only one with the indicated name, use that one
        app_name: Optional[str] = function_def.app_name or None  # coalesce protobuf field to None
        matching_apps = _App._all_apps.get(app_name, [])
        if len(matching_apps) > 1:
            if app_name is not None:
                warning_sub_message = f"app with the same name ('{app_name}')"
            else:
                warning_sub_message = "unnamed app"
            logger.warning(
                f"You have more than one {warning_sub_message}. It's recommended to name all your Apps uniquely when using multiple apps"
            )
        elif len(matching_apps) == 1:
            (active_app,) = matching_apps
        # there could also technically be zero found apps, but that should probably never be an issue since that would mean user won't use is_inside or other function handles anyway

    # Check this property before we turn it into a method (overriden by webhooks)
    is_async = get_is_async(fun)

    # Use the function definition for whether this is a generator (overriden by webhooks)
    is_generator = function_def.function_type == api_pb2.Function.FUNCTION_TYPE_GENERATOR

    # What data format is used for function inputs and outputs
    data_format = api_pb2.DATA_FORMAT_PICKLE

    # Container can fetch multiple inputs simultaneously
    if pty_info.pty_type == api_pb2.PTYInfo.PTY_TYPE_SHELL:
        # Concurrency doesn't apply for `modal shell`.
        input_concurrency = 1
    else:
        input_concurrency = function_def.allow_concurrent_inputs or 1

    # Instantiate the class if it's defined
    if cls:
        if ser_params:
            _client: _Client = synchronizer._translate_in(client)
            args, kwargs = deserialize(ser_params, _client)
        else:
            args, kwargs = (), {}
        obj = cls(*args, **kwargs)
        if isinstance(cls, Cls):
            obj = obj.get_obj()
        # Bind the function to the instance (using the descriptor protocol!)
        fun = fun.__get__(obj)
    else:
        obj = None

    if function_def.webhook_config.type:
        is_async = True
        is_generator = True
        data_format = api_pb2.DATA_FORMAT_ASGI

        if function_def.webhook_config.type == api_pb2.WEBHOOK_TYPE_ASGI_APP:
            # Function returns an asgi_app, which we can use as a callable.
            fun = asgi_app_wrapper(fun(), container_io_manager)

        elif function_def.webhook_config.type == api_pb2.WEBHOOK_TYPE_WSGI_APP:
            # Function returns an wsgi_app, which we can use as a callable.
            fun = wsgi_app_wrapper(fun(), container_io_manager)

        elif function_def.webhook_config.type == api_pb2.WEBHOOK_TYPE_FUNCTION:
            # Function is a webhook without an ASGI app. Create one for it.
            fun = asgi_app_wrapper(
                webhook_asgi_app(fun, function_def.webhook_config.method),
                container_io_manager,
            )

        elif function_def.webhook_config.type == api_pb2.WEBHOOK_TYPE_WEB_SERVER:
            # Function spawns an HTTP web server listening at a port.
            fun()

            # We intentionally try to connect to the external interface instead of the loopback
            # interface here so users are forced to expose the server. This allows us to potentially
            # change the implementation to use an external bridge in the future.
            host = get_ip_address(b"eth0")
            port = function_def.webhook_config.web_server_port
            startup_timeout = function_def.webhook_config.web_server_startup_timeout
            wait_for_web_server(host, port, timeout=startup_timeout)
            fun = asgi_app_wrapper(web_server_proxy(host, port), container_io_manager)

        else:
            raise InvalidError(f"Unrecognized web endpoint type {function_def.webhook_config.type}")

    return ImportedFunction(
        obj,
        fun,
        active_app,
        is_async,
        is_generator,
        data_format,
        input_concurrency,
        function_def.is_auto_snapshot,
        function,
    )


def call_lifecycle_functions(
    event_loop: UserCodeEventLoop,
    container_io_manager,  #: ContainerIOManager,  TODO: this type is generated at runtime
    funcs: Sequence[Callable],
) -> None:
    """Call function(s), can be sync or async, but any return values are ignored."""
    with container_io_manager.handle_user_exception():
        for func in funcs:
            # We are deprecating parameterized exit methods but want to gracefully handle old code.
            # We can remove this once the deprecation in the actual @exit decorator is enforced.
            args = (None, None, None) if method_has_params(func) else ()
            res = func(
                *args
            )  # in case func is non-async, it's executed here and sigint will by default interrupt it using a KeyboardInterrupt exception
            if inspect.iscoroutine(res):
                # if however func is async, we have to jump through some hoops
                event_loop.run(res)


def main(container_args: api_pb2.ContainerArguments, client: Client):
    # This is a bit weird but we need both the blocking and async versions of ContainerIOManager.
    # At some point, we should fix that by having built-in support for running "user code"
    container_io_manager = ContainerIOManager(container_args, client)

    with container_io_manager.heartbeats(), UserCodeEventLoop() as event_loop:
        # If this is a serialized function, fetch the definition from the server
        if container_args.function_def.definition_type == api_pb2.Function.DEFINITION_TYPE_SERIALIZED:
            ser_cls, ser_fun = container_io_manager.get_serialized_function()
        else:
            ser_cls, ser_fun = None, None

        # Initialize the function, importing user code.
        with container_io_manager.handle_user_exception():
            imp_fun = import_function(
                container_args.function_def,
                ser_cls,
                ser_fun,
                container_args.serialized_params,
                container_io_manager,
                client,
            )

        # Get ids and metadata for objects (primarily functions and classes) on the app
        container_app: RunningApp = container_io_manager.get_app_objects()

        # Initialize objects on the app.
        # This is basically only functions and classes - anything else is deprecated and will be unsupported soon
        if imp_fun.app is not None:
            app: App = synchronizer._translate_out(imp_fun.app, Interface.BLOCKING)
            app._init_container(client, container_app)

        # Hydrate all function dependencies.
        # TODO(erikbern): we an remove this once we
        # 1. Enable lazy hydration for all objects
        # 2. Fully deprecate .new() objects
        if imp_fun.function:
            _client: _Client = synchronizer._translate_in(client)  # TODO(erikbern): ugly
            dep_object_ids: List[str] = [dep.object_id for dep in container_args.function_def.object_dependencies]
            function_deps = imp_fun.function.deps(only_explicit_mounts=True)
            if len(function_deps) != len(dep_object_ids):
                raise ExecutionError(
                    f"Function has {len(function_deps)} dependencies"
                    f" but container got {len(dep_object_ids)} object ids."
                )
            for object_id, obj in zip(dep_object_ids, function_deps):
                metadata: Message = container_app.object_handle_metadata[object_id]
                obj._hydrate(object_id, _client, metadata)

        # Identify all "enter" methods that need to run before we snapshot.
        if imp_fun.obj is not None and not imp_fun.is_auto_snapshot:
            pre_snapshot_methods = _find_callables_for_obj(imp_fun.obj, _PartialFunctionFlags.ENTER_PRE_SNAPSHOT)
            call_lifecycle_functions(event_loop, container_io_manager, list(pre_snapshot_methods.values()))

        # If this container is being used to create a checkpoint, checkpoint the container after
        # global imports and innitialization. Checkpointed containers run from this point onwards.
        if container_args.function_def.is_checkpointing_function:
            container_io_manager.memory_snapshot()

        # Install hooks for interactive functions.
        if container_args.function_def.pty_info.pty_type != api_pb2.PTYInfo.PTY_TYPE_UNSPECIFIED:

            def breakpoint_wrapper():
                # note: it would be nice to not have breakpoint_wrapper() included in the backtrace
                interact()
                import pdb

                pdb.set_trace()

            sys.breakpointhook = breakpoint_wrapper

        # Identify the "enter" methods to run after resuming from a snapshot.
        if imp_fun.obj is not None and not imp_fun.is_auto_snapshot:
            post_snapshot_methods = _find_callables_for_obj(imp_fun.obj, _PartialFunctionFlags.ENTER_POST_SNAPSHOT)
            call_lifecycle_functions(event_loop, container_io_manager, list(post_snapshot_methods.values()))

        # Execute the function.
        try:
            if imp_fun.is_async:
                event_loop.run(call_function_async(container_io_manager, imp_fun))
            else:
                # Set up a signal handler for `SIGUSR1`, which gets translated to an InputCancellation
                # during function execution. This is sent to cancel inputs from the user.
                def _cancel_input_signal_handler(signum, stackframe):
                    raise InputCancellation("Input was cancelled by user")

                signal.signal(signal.SIGUSR1, _cancel_input_signal_handler)

                call_function_sync(container_io_manager, imp_fun)
        finally:
            # Run exit handlers. From this point onward, ignore all SIGINT signals that come from
            # graceful shutdowns originating on the worker, as well as stray SIGUSR1 signals that
            # may have been sent to cancel inputs.
            int_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
            usr1_handler = signal.signal(signal.SIGUSR1, signal.SIG_IGN)

            try:
                # Identify "exit" methods and run them.
                if imp_fun.obj is not None and not imp_fun.is_auto_snapshot:
                    exit_methods = _find_callables_for_obj(imp_fun.obj, _PartialFunctionFlags.EXIT)
                    call_lifecycle_functions(event_loop, container_io_manager, list(exit_methods.values()))

                # Finally, commit on exit to catch uncommitted volume changes and surface background
                # commit errors.
                container_io_manager.volume_commit(
                    [v.volume_id for v in container_args.function_def.volume_mounts if v.allow_background_commits]
                )
            finally:
                # Restore the original signal handler, needed for container_test hygiene since the
                # test runs `main()` multiple times in the same process.
                signal.signal(signal.SIGINT, int_handler)
                signal.signal(signal.SIGUSR1, usr1_handler)


if __name__ == "__main__":
    logger.debug("Container: starting")

    # Check and warn on deprecated Python version
    if sys.version_info[:2] == (3, 8):
        msg = (
            "You are using Python 3.8 in your remote environment. Modal will soon drop support for this version,"
            " and you will be unable to use this Image. Please update your Image definition."
        )
        deprecation_warning((2024, 5, 2), msg, show_source=False, pending=True)

    container_args = api_pb2.ContainerArguments()
    container_args.ParseFromString(base64.b64decode(sys.argv[1]))

    # Note that we're creating the client in a synchronous context, but it will be running in a separate thread.
    # This is good because if the function is long running then we the client can still send heartbeats
    # The only caveat is a bunch of calls will now cross threads, which adds a bit of overhead?
    client = Client.from_env()

    try:
        with proxy_tunnel(container_args.proxy_info):
            try:
                main(container_args, client)
            except UserException:
                logger.info("User exception caught, exiting")
    except KeyboardInterrupt:
        logger.debug("Container: interrupted")

    # Detect if any non-daemon threads are still running, which will prevent the Python interpreter
    # from shutting down. The sleep(0) here is needed for finished ThreadPoolExecutor resources to
    # shut down without triggering this warning (e.g., `@wsgi_app()`).
    time.sleep(0)
    lingering_threads: List[threading.Thread] = []
    for thread in threading.enumerate():
        current_thread = threading.get_ident()
        if thread.ident is not None and thread.ident != current_thread and not thread.daemon and thread.is_alive():
            lingering_threads.append(thread)
    if lingering_threads:
        thread_names = ", ".join(t.name for t in lingering_threads)
        logger.warning(
            f"Detected {len(lingering_threads)} background thread(s) [{thread_names}] still running after container exit. This will prevent runner shutdown for up to 30 seconds."
        )

    logger.debug("Container: done")
