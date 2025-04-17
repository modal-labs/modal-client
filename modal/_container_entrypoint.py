# Copyright Modal Labs 2022
# ruff: noqa: E402
import os

from modal._runtime.user_code_imports import (
    Service,
    import_class_service,
    import_single_function_service,
)

telemetry_socket = os.environ.get("MODAL_TELEMETRY_SOCKET")
if telemetry_socket:
    from ._runtime.telemetry import instrument_imports

    instrument_imports(telemetry_socket)

import asyncio
import concurrent.futures
import inspect
import queue
import signal
import sys
import threading
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Optional

from google.protobuf.message import Message

from modal._clustered_functions import initialize_clustered_function
from modal._partial_function import (
    _find_callables_for_obj,
    _PartialFunctionFlags,
)
from modal._serialization import deserialize, deserialize_params
from modal._utils.async_utils import TaskContext, synchronizer
from modal._utils.function_utils import (
    callable_has_non_self_params,
)
from modal.app import App, _App
from modal.client import Client, _Client
from modal.config import logger
from modal.exception import ExecutionError, InputCancellation, InvalidError
from modal.running_app import RunningApp, running_app_from_layout
from modal_proto import api_pb2

from ._runtime import execution_context
from ._runtime.container_io_manager import (
    ContainerIOManager,
    IOContext,
    UserException,
    _ContainerIOManager,
)

if TYPE_CHECKING:
    import modal._object
    import modal._runtime.container_io_manager


class DaemonizedThreadPool:
    # Used instead of ThreadPoolExecutor, since the latter won't allow
    # the interpreter to shut down before the currently running tasks
    # have finished
    def __init__(self, max_threads: int):
        self.max_threads = max_threads

    def __enter__(self):
        self.spawned_workers = 0
        self.inputs: queue.Queue[Any] = queue.Queue()
        self.finished = threading.Event()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.finished.set()

        if exc_type is None:
            self.inputs.join()
        else:
            # special case - allows us to exit the
            if self.inputs.unfinished_tasks:
                logger.info(
                    f"Exiting DaemonizedThreadPool with {self.inputs.unfinished_tasks} active "
                    f"inputs due to exception: {repr(exc_type)}"
                )

    def submit(self, func, *args):
        def worker_thread():
            while not self.finished.is_set():
                try:
                    _func, _args = self.inputs.get(timeout=1)
                except queue.Empty:
                    continue
                try:
                    _func(*_args)
                except BaseException:
                    logger.exception(f"Exception raised by {_func} in DaemonizedThreadPool worker!")
                self.inputs.task_done()

        if self.spawned_workers < self.max_threads:
            threading.Thread(target=worker_thread, daemon=True).start()
            self.spawned_workers += 1

        self.inputs.put((func, args))


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
        self.tasks = set()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.loop.run_until_complete(self.loop.shutdown_asyncgens())
        if sys.version_info[:2] >= (3, 9):
            self.loop.run_until_complete(self.loop.shutdown_default_executor())  # Introduced in Python 3.9

        for task in self.tasks:
            task.cancel()

        self.loop.close()

    def create_task(self, coro):
        task = self.loop.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

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

            # this should normally not happen, but the second sigint would "hard kill" the event loop!
            raise KeyboardInterrupt()

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


def call_function(
    user_code_event_loop: UserCodeEventLoop,
    container_io_manager: "modal._runtime.container_io_manager.ContainerIOManager",
    finalized_functions: dict[str, "modal._runtime.user_code_imports.FinalizedFunction"],
    batch_max_size: int,
    batch_wait_ms: int,
):
    async def run_input_async(io_context: IOContext) -> None:
        started_at = time.time()
        input_ids, function_call_ids = io_context.input_ids, io_context.function_call_ids
        reset_context = execution_context._set_current_context_ids(input_ids, function_call_ids)
        async with container_io_manager.handle_input_exception.aio(io_context, started_at):
            res = io_context.call_finalized_function()
            # TODO(erikbern): any exception below shouldn't be considered a user exception
            if io_context.finalized_function.is_generator:
                if not inspect.isasyncgen(res):
                    raise InvalidError(f"Async generator function returned value of type {type(res)}")

                # Send up to this many outputs at a time.
                generator_queue: asyncio.Queue[Any] = await container_io_manager._queue_create.aio(1024)
                generator_output_task = asyncio.create_task(
                    container_io_manager.generator_output_task.aio(
                        function_call_ids[0],
                        io_context.finalized_function.data_format,
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
                await container_io_manager.push_outputs.aio(
                    io_context,
                    started_at,
                    message,
                    api_pb2.DATA_FORMAT_GENERATOR_DONE,
                )
            else:
                if not inspect.iscoroutine(res) or inspect.isgenerator(res) or inspect.isasyncgen(res):
                    raise InvalidError(
                        f"Async (non-generator) function returned value of type {type(res)}"
                        " You might need to use @app.function(..., is_generator=True)."
                    )
                value = await res
                await container_io_manager.push_outputs.aio(
                    io_context,
                    started_at,
                    value,
                    io_context.finalized_function.data_format,
                )
        reset_context()

    def run_input_sync(io_context: IOContext) -> None:
        started_at = time.time()
        input_ids, function_call_ids = io_context.input_ids, io_context.function_call_ids
        reset_context = execution_context._set_current_context_ids(input_ids, function_call_ids)
        with container_io_manager.handle_input_exception(io_context, started_at):
            res = io_context.call_finalized_function()

            # TODO(erikbern): any exception below shouldn't be considered a user exception
            if io_context.finalized_function.is_generator:
                if not inspect.isgenerator(res):
                    raise InvalidError(f"Generator function returned value of type {type(res)}")

                # Send up to this many outputs at a time.
                generator_queue: asyncio.Queue[Any] = container_io_manager._queue_create(1024)
                generator_output_task: concurrent.futures.Future = container_io_manager.generator_output_task(  # type: ignore
                    function_call_ids[0],
                    io_context.finalized_function.data_format,
                    generator_queue,
                    _future=True,  # type: ignore  # Synchronicity magic to return a future.
                )

                item_count = 0
                for value in res:
                    container_io_manager._queue_put(generator_queue, value)
                    item_count += 1

                container_io_manager._queue_put(generator_queue, _ContainerIOManager._GENERATOR_STOP_SENTINEL)
                generator_output_task.result()  # Wait to finish sending generator outputs.
                message = api_pb2.GeneratorDone(items_total=item_count)
                container_io_manager.push_outputs(io_context, started_at, message, api_pb2.DATA_FORMAT_GENERATOR_DONE)
            else:
                if inspect.iscoroutine(res) or inspect.isgenerator(res) or inspect.isasyncgen(res):
                    raise InvalidError(
                        f"Sync (non-generator) function return value of type {type(res)}."
                        " You might need to use @app.function(..., is_generator=True)."
                    )
                container_io_manager.push_outputs(
                    io_context, started_at, res, io_context.finalized_function.data_format
                )
        reset_context()

    if container_io_manager.input_concurrency_enabled:
        with DaemonizedThreadPool(max_threads=container_io_manager.max_concurrency) as thread_pool:

            def make_async_cancel_callback(task):
                def f():
                    user_code_event_loop.loop.call_soon_threadsafe(task.cancel)

                return f

            did_sigint = False

            def cancel_callback_sync():
                nonlocal did_sigint
                # We only want one sigint even if multiple inputs are cancelled
                # A second sigint would forcibly shut down the event loop and spew
                # out a bunch of tracebacks, which we only want to happen in case
                # the worker kills this process after a failed self-termination
                if not did_sigint:
                    did_sigint = True
                    logger.warning(
                        "User cancelling input of non-async functions with input concurrency enabled.\n"
                        "This shuts down the container, causing concurrently running inputs to be "
                        "rescheduled in other containers."
                    )
                    os.kill(os.getpid(), signal.SIGINT)

            async def run_concurrent_inputs():
                # all run_input coroutines will have completed by the time we leave the execution context
                # but the wrapping *tasks* may not yet have been resolved, so we add a 0.01s
                # for them to resolve gracefully:
                async with TaskContext(0.01) as task_context:
                    async for io_context in container_io_manager.run_inputs_outputs.aio(
                        finalized_functions, batch_max_size, batch_wait_ms
                    ):
                        # Note that run_inputs_outputs will not return until all the input slots are released
                        # so that they can be acquired by the run_inputs_outputs finalizer
                        # This prevents leaving the task_context before outputs have been created
                        # TODO: refactor to make this a bit more easy to follow?
                        if io_context.finalized_function.is_async:
                            input_task = task_context.create_task(run_input_async(io_context))
                            io_context.set_cancel_callback(make_async_cancel_callback(input_task))
                        else:
                            # run sync input in thread
                            thread_pool.submit(run_input_sync, io_context)
                            io_context.set_cancel_callback(cancel_callback_sync)

            user_code_event_loop.run(run_concurrent_inputs())
    else:
        for io_context in container_io_manager.run_inputs_outputs(finalized_functions, batch_max_size, batch_wait_ms):
            # This goes to a registered signal handler for sync Modal functions, or to the
            # `UserCodeEventLoop` for async functions.
            #
            # We only send this signal on functions that do not have concurrent inputs enabled.
            # This allows us to do fine-grained input cancellation. On sync functions, the
            # SIGUSR1 signal should interrupt the main thread where user code is running,
            # raising an InputCancellation() exception. On async functions, the signal should
            # reach a handler in UserCodeEventLoop, which cancels the task.
            io_context.set_cancel_callback(lambda: os.kill(os.getpid(), signal.SIGUSR1))

            if io_context.finalized_function.is_async:
                user_code_event_loop.run(run_input_async(io_context))
            else:
                # Set up a custom signal handler for `SIGUSR1`, which gets translated to an InputCancellation
                # during function execution. This is sent to cancel inputs from the user
                def _cancel_input_signal_handler(signum, stackframe):
                    raise InputCancellation("Input was cancelled by user")

                usr1_handler = signal.signal(signal.SIGUSR1, _cancel_input_signal_handler)
                # run this sync code in the main thread, blocking the "userland" event loop
                # this lets us cancel it using a signal handler that raises an exception
                try:
                    run_input_sync(io_context)
                finally:
                    signal.signal(signal.SIGUSR1, usr1_handler)  # reset signal handler


def call_lifecycle_functions(
    event_loop: UserCodeEventLoop,
    container_io_manager,  #: ContainerIOManager,  TODO: this type is generated at runtime
    funcs: Sequence[Callable[..., Any]],
) -> None:
    """Call function(s), can be sync or async, but any return values are ignored."""
    with container_io_manager.handle_user_exception():
        for func in funcs:
            # We are deprecating parametrized exit methods but want to gracefully handle old code.
            # We can remove this once the deprecation in the actual @exit decorator is enforced.
            args = (None, None, None) if callable_has_non_self_params(func) else ()
            # in case func is non-async, it's executed here and sigint will by default
            # interrupt it using a KeyboardInterrupt exception
            res = func(*args)
            if inspect.iscoroutine(res):
                # if however func is async, we have to jump through some hoops
                event_loop.run(res)


def main(container_args: api_pb2.ContainerArguments, client: Client):
    # This is a bit weird but we need both the blocking and async versions of ContainerIOManager.
    # At some point, we should fix that by having built-in support for running "user code"
    container_io_manager = ContainerIOManager(container_args, client)
    active_app: _App
    service: Service
    function_def = container_args.function_def
    is_auto_snapshot: bool = function_def.is_auto_snapshot
    # The worker sets this flag to "1" for snapshot and restore tasks. Otherwise, this flag is unset,
    # in which case snapshots should be disabled.
    is_snapshotting_function = (
        function_def.is_checkpointing_function and os.environ.get("MODAL_ENABLE_SNAP_RESTORE") == "1"
    )

    _client: _Client = synchronizer._translate_in(client)  # TODO(erikbern): ugly

    # Call ContainerHello - currently a noop but might be used later for things
    container_io_manager.hello()

    with container_io_manager.heartbeats(is_snapshotting_function), UserCodeEventLoop() as event_loop:
        # If this is a serialized function, fetch the definition from the server
        if function_def.definition_type == api_pb2.Function.DEFINITION_TYPE_SERIALIZED:
            assert function_def.function_serialized or function_def.class_serialized

            if function_def.function_serialized:
                ser_fun = deserialize(function_def.function_serialized, _client)
            else:
                ser_fun = None

            if function_def.class_serialized:
                ser_usr_cls = deserialize(function_def.class_serialized, _client)
            else:
                ser_usr_cls = None
        else:
            ser_usr_cls, ser_fun = None, None

        # Initialize the function, importing user code.
        with container_io_manager.handle_user_exception():
            if container_args.serialized_params:
                param_args, param_kwargs = deserialize_params(container_args.serialized_params, function_def, _client)
            else:
                param_args = ()
                param_kwargs = {}

            with execution_context._import_context():
                if function_def.is_class:
                    # this is a bit ugly - match the function and class based on function name to get metadata
                    # This metadata is required in order to hydrate the class in case it's not globally
                    # decorated (or serialized)
                    service_base_function_id = container_args.app_layout.function_ids[function_def.function_name]
                    service_function_hydration_data = [
                        o for o in container_args.app_layout.objects if o.object_id == service_base_function_id
                    ][0]
                    class_id = container_args.app_layout.class_ids[function_def.function_name.removesuffix(".*")]

                    service = import_class_service(
                        function_def,
                        service_function_hydration_data,
                        class_id,
                        client,
                        ser_usr_cls,
                        param_args,
                        param_kwargs,
                    )
                else:
                    service = import_single_function_service(
                        function_def,
                        ser_usr_cls,
                        ser_fun,
                    )

            active_app = service.app

            if function_def.pty_info.pty_type == api_pb2.PTYInfo.PTY_TYPE_SHELL:
                # Concurrency and batching doesn't apply for `modal shell`.
                batch_max_size = 0
                batch_wait_ms = 0
            else:
                batch_max_size = function_def.batch_max_size or 0
                batch_wait_ms = function_def.batch_linger_ms or 0

        # Get ids and metadata for objects (primarily functions and classes) on the app
        container_app: RunningApp = running_app_from_layout(container_args.app_id, container_args.app_layout)

        # Initialize objects on the app.
        # This is basically only functions and classes - anything else is deprecated and will be unsupported soon
        app: App = synchronizer._translate_out(active_app)
        app._init_container(client, container_app)

        # Hydrate all function dependencies.
        # TODO(erikbern): we an remove this once we
        # 1. Enable lazy hydration for all objects
        # 2. Fully deprecate .new() objects
        if service.service_deps is not None:  # this is not set for serialized or non-global scope functions
            dep_object_ids: list[str] = [dep.object_id for dep in function_def.object_dependencies]
            if len(service.service_deps) != len(dep_object_ids):
                raise ExecutionError(
                    f"Function has {len(service.service_deps)} dependencies"
                    f" but container got {len(dep_object_ids)} object ids.\n"
                    f"Code deps: {service.service_deps}\n"
                    f"Object ids: {dep_object_ids}"
                )
            for object_id, obj in zip(dep_object_ids, service.service_deps):
                metadata: Message = container_app.object_handle_metadata[object_id]
                obj._hydrate(object_id, _client, metadata)

        # Initialize clustered functions.
        if function_def._experimental_group_size > 0:
            initialize_clustered_function(
                client,
                container_args.task_id,
                function_def._experimental_group_size,
            )

        # Identify all "enter" methods that need to run before we snapshot.
        if service.user_cls_instance is not None and not is_auto_snapshot:
            pre_snapshot_methods = _find_callables_for_obj(
                service.user_cls_instance, _PartialFunctionFlags.ENTER_PRE_SNAPSHOT
            )
            call_lifecycle_functions(event_loop, container_io_manager, list(pre_snapshot_methods.values()))

        # If this container is being used to create a checkpoint, checkpoint the container after
        # global imports and initialization. Checkpointed containers run from this point onwards.
        if is_snapshotting_function:
            container_io_manager.memory_snapshot()

        # Install hooks for interactive functions.
        def breakpoint_wrapper():
            # note: it would be nice to not have breakpoint_wrapper() included in the backtrace
            container_io_manager.interact(from_breakpoint=True)
            import pdb

            frame = inspect.currentframe().f_back

            pdb.Pdb().set_trace(frame)

        sys.breakpointhook = breakpoint_wrapper

        # Identify the "enter" methods to run after resuming from a snapshot.
        if service.user_cls_instance is not None and not is_auto_snapshot:
            post_snapshot_methods = _find_callables_for_obj(
                service.user_cls_instance, _PartialFunctionFlags.ENTER_POST_SNAPSHOT
            )
            call_lifecycle_functions(event_loop, container_io_manager, list(post_snapshot_methods.values()))

        with container_io_manager.handle_user_exception():
            finalized_functions = service.get_finalized_functions(function_def, container_io_manager)
        # Execute the function.
        lifespan_background_tasks = []
        try:
            for finalized_function in finalized_functions.values():
                if finalized_function.lifespan_manager:
                    lifespan_background_tasks.append(
                        event_loop.create_task(finalized_function.lifespan_manager.background_task())
                    )
                    with container_io_manager.handle_user_exception():
                        event_loop.run(finalized_function.lifespan_manager.lifespan_startup())
            call_function(
                event_loop,
                container_io_manager,
                finalized_functions,
                batch_max_size,
                batch_wait_ms,
            )
        finally:
            # Run exit handlers. From this point onward, ignore all SIGINT signals that come from
            # graceful shutdowns originating on the worker, as well as stray SIGUSR1 signals that
            # may have been sent to cancel inputs.
            int_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
            usr1_handler = signal.signal(signal.SIGUSR1, signal.SIG_IGN)

            try:
                try:
                    # run lifespan shutdown for asgi apps
                    for finalized_function in finalized_functions.values():
                        if finalized_function.lifespan_manager:
                            with container_io_manager.handle_user_exception():
                                event_loop.run(finalized_function.lifespan_manager.lifespan_shutdown())
                finally:
                    # no need to keep the lifespan asgi call around - we send it no more messages
                    for lifespan_background_task in lifespan_background_tasks:
                        lifespan_background_task.cancel()  # prevent dangling tasks

                    # Identify "exit" methods and run them.
                    # want to make sure this is called even if the lifespan manager fails
                    if service.user_cls_instance is not None and not is_auto_snapshot:
                        exit_methods = _find_callables_for_obj(service.user_cls_instance, _PartialFunctionFlags.EXIT)
                        call_lifecycle_functions(event_loop, container_io_manager, list(exit_methods.values()))

                # Finally, commit on exit to catch uncommitted volume changes and surface background
                # commit errors.
                container_io_manager.volume_commit(
                    [v.volume_id for v in function_def.volume_mounts if v.allow_background_commits]
                )
            finally:
                # Restore the original signal handler, needed for container_test hygiene since the
                # test runs `main()` multiple times in the same process.
                signal.signal(signal.SIGINT, int_handler)
                signal.signal(signal.SIGUSR1, usr1_handler)


if __name__ == "__main__":
    logger.debug("Container: starting")

    container_args = api_pb2.ContainerArguments()
    container_arguments_path: Optional[str] = os.environ.get("MODAL_CONTAINER_ARGUMENTS_PATH")
    if container_arguments_path is None:
        raise RuntimeError("No path to the container arguments file provided!")
    container_args.ParseFromString(open(container_arguments_path, "rb").read())

    # Note that we're creating the client in a synchronous context, but it will be running in a separate thread.
    # This is good because if the function is long running then we the client can still send heartbeats
    # The only caveat is a bunch of calls will now cross threads, which adds a bit of overhead?
    client = Client.from_env()

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
    lingering_threads: list[threading.Thread] = []
    for thread in threading.enumerate():
        current_thread = threading.get_ident()
        if thread.ident is not None and thread.ident != current_thread and not thread.daemon and thread.is_alive():
            lingering_threads.append(thread)
    if lingering_threads:
        thread_names = ", ".join(t.name for t in lingering_threads)
        logger.warning(
            f"Detected {len(lingering_threads)} background thread(s) [{thread_names}] still running "
            "after container exit. This will prevent runner shutdown for up to 30 seconds."
        )

    logger.debug("Container: done")
