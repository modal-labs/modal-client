# Copyright Modal Labs 2025
import asyncio
import dataclasses
import os
import time
import typing
import warnings
from collections.abc import AsyncGenerator
from multiprocessing.synchronize import Event
from typing import TYPE_CHECKING, Any, Optional, TypeVar

from grpclib import GRPCError, Status
from synchronicity.async_wrap import asynccontextmanager

import modal._runtime.execution_context
import modal_proto.api_pb2
from modal_proto import api_pb2

from ._functions import _Function
from ._object import _get_environment_name, _Object
from ._pty import get_pty_info
from ._resolver import Resolver
from ._traceback import print_server_warnings, traceback_contains_remote_call
from ._utils.async_utils import TaskContext, gather_cancel_on_exc, synchronize_api
from ._utils.deprecation import deprecation_error
from ._utils.git_utils import get_git_commit_info
from ._utils.grpc_utils import retry_transient_errors
from ._utils.name_utils import check_object_name, is_valid_tag
from .client import HEARTBEAT_INTERVAL, HEARTBEAT_TIMEOUT, _Client
from .cls import _Cls
from .config import config, logger
from .environments import _get_environment_cached
from .exception import InteractiveTimeoutError, InvalidError, RemoteError, _CliUserExecutionError
from .output import _get_output_manager, enable_output
from .running_app import RunningApp, running_app_from_layout
from .sandbox import _Sandbox
from .secret import _Secret
from .stream_type import StreamType

if TYPE_CHECKING:
    from .app import _App
else:
    _App = TypeVar("_App")


V = TypeVar("V")


async def _heartbeat(client: _Client, app_id: str) -> None:
    request = api_pb2.AppHeartbeatRequest(app_id=app_id)
    # TODO(erikbern): we should capture exceptions here
    # * if request fails: destroy the client
    # * if server says the app is gone: print a helpful warning about detaching
    await retry_transient_errors(client.stub.AppHeartbeat, request, attempt_timeout=HEARTBEAT_TIMEOUT)


async def _init_local_app_existing(client: _Client, existing_app_id: str, environment_name: str) -> RunningApp:
    # Get all the objects first
    obj_req = api_pb2.AppGetLayoutRequest(app_id=existing_app_id)
    obj_resp, _ = await gather_cancel_on_exc(
        retry_transient_errors(client.stub.AppGetLayout, obj_req),
        # Cache the environment associated with the app now as we will use it later
        _get_environment_cached(environment_name, client),
    )
    app_page_url = f"https://modal.com/apps/{existing_app_id}"  # TODO (elias): this should come from the backend
    return running_app_from_layout(
        existing_app_id,
        obj_resp.app_layout,
        app_page_url=app_page_url,
    )


async def _init_local_app_new(
    client: _Client,
    description: str,
    app_state: int,  # ValueType
    environment_name: str = "",
    interactive: bool = False,
) -> RunningApp:
    app_req = api_pb2.AppCreateRequest(
        description=description,
        environment_name=environment_name,
        app_state=app_state,  # type: ignore
    )
    app_resp, _ = await gather_cancel_on_exc(  # TODO: use TaskGroup?
        retry_transient_errors(client.stub.AppCreate, app_req),
        # Cache the environment associated with the app now as we will use it later
        _get_environment_cached(environment_name, client),
    )
    logger.debug(f"Created new app with id {app_resp.app_id}")
    return RunningApp(
        app_resp.app_id,
        app_page_url=app_resp.app_page_url,
        app_logs_url=app_resp.app_logs_url,
        interactive=interactive,
    )


async def _init_local_app_from_name(
    client: _Client,
    name: str,
    namespace: Any,
    environment_name: str = "",
) -> RunningApp:
    # Look up any existing deployment
    app_req = api_pb2.AppGetByDeploymentNameRequest(
        name=name,
        namespace=namespace,
        environment_name=environment_name,
    )
    app_resp = await retry_transient_errors(client.stub.AppGetByDeploymentName, app_req)
    existing_app_id = app_resp.app_id or None

    # Grab the app
    if existing_app_id is not None:
        return await _init_local_app_existing(client, existing_app_id, environment_name)
    else:
        return await _init_local_app_new(
            client, name, api_pb2.APP_STATE_INITIALIZING, environment_name=environment_name
        )


async def _create_all_objects(
    client: _Client,
    running_app: RunningApp,
    functions: dict[str, _Function],
    classes: dict[str, _Cls],
    environment_name: str,
) -> None:
    """Create objects that have been defined but not created on the server."""
    indexed_objects: dict[str, _Object] = {**functions, **classes}
    resolver = Resolver(
        client,
        environment_name=environment_name,
        app_id=running_app.app_id,
    )
    with resolver.display():
        # Get current objects, and reset all objects
        tag_to_object_id = {**running_app.function_ids, **running_app.class_ids}
        running_app.function_ids = {}
        running_app.class_ids = {}

        # Assign all objects
        for tag, obj in indexed_objects.items():
            # Reset object_id in case the app runs twice
            # TODO(erikbern): clean up the interface
            obj._unhydrate()

        # Preload all functions to make sure they have ids assigned before they are loaded.
        # This is important to make sure any enclosed function handle references in serialized
        # functions have ids assigned to them when the function is serialized.
        # Note: when handles/objs are merged, all objects will need to get ids pre-assigned
        # like this in order to be referrable within serialized functions
        async def _preload(tag, obj):
            existing_object_id = tag_to_object_id.get(tag)
            # Note: preload only currently implemented for Functions, returns None otherwise
            # this is to ensure that directly referenced functions from the global scope has
            # ids associated with them when they are serialized into other functions
            await resolver.preload(obj, existing_object_id)
            if obj.is_hydrated:
                tag_to_object_id[tag] = obj.object_id

        await TaskContext.gather(*(_preload(tag, obj) for tag, obj in indexed_objects.items()))

        async def _load(tag, obj):
            existing_object_id = tag_to_object_id.get(tag)
            await resolver.load(obj, existing_object_id)
            if _Function._is_id_type(obj.object_id):
                running_app.function_ids[tag] = obj.object_id
            elif _Cls._is_id_type(obj.object_id):
                running_app.class_ids[tag] = obj.object_id
            else:
                raise RuntimeError(f"Unexpected object {obj.object_id}")

        await TaskContext.gather(*(_load(tag, obj) for tag, obj in indexed_objects.items()))


async def _publish_app(
    client: _Client,
    running_app: RunningApp,
    app_state: int,  # api_pb2.AppState.value
    functions: dict[str, _Function],
    classes: dict[str, _Cls],
    name: str = "",  # Only relevant for deployments
    tag: str = "",  # Only relevant for deployments
    commit_info: Optional[api_pb2.CommitInfo] = None,  # Git commit information
) -> tuple[str, list[api_pb2.Warning]]:
    """Wrapper for AppPublish RPC."""

    definition_ids = {obj.object_id: obj._get_metadata().definition_id for obj in functions.values()}  # type: ignore

    request = api_pb2.AppPublishRequest(
        app_id=running_app.app_id,
        name=name,
        deployment_tag=tag,
        app_state=app_state,  # type: ignore  : should be a api_pb2.AppState.value
        function_ids=running_app.function_ids,
        class_ids=running_app.class_ids,
        definition_ids=definition_ids,
        commit_info=commit_info,
    )

    try:
        response = await retry_transient_errors(client.stub.AppPublish, request)
    except GRPCError as exc:
        if exc.status == Status.INVALID_ARGUMENT or exc.status == Status.FAILED_PRECONDITION:
            raise InvalidError(exc.message)
        raise

    print_server_warnings(response.server_warnings)
    return response.url, response.server_warnings


async def _disconnect(
    client: _Client,
    app_id: str,
    reason: "modal_proto.api_pb2.AppDisconnectReason.ValueType",
    exc_str: str = "",
) -> None:
    """Tell the server the client has disconnected for this app. Terminates all running tasks
    for ephemeral apps."""

    if exc_str:
        exc_str = exc_str[:1000]  # Truncate to 1000 chars

    logger.debug("Sending app disconnect/stop request")
    req_disconnect = api_pb2.AppClientDisconnectRequest(app_id=app_id, reason=reason, exception=exc_str)
    await retry_transient_errors(client.stub.AppClientDisconnect, req_disconnect)
    logger.debug("App disconnected")


async def _status_based_disconnect(client: _Client, app_id: str, exc_info: Optional[BaseException] = None):
    """Disconnect local session of a running app, sending relevant metadata

    exc_info: Exception if an exception caused the disconnect
    """
    if isinstance(exc_info, (KeyboardInterrupt, asyncio.CancelledError)):
        reason = api_pb2.APP_DISCONNECT_REASON_KEYBOARD_INTERRUPT
    elif exc_info is not None:
        if traceback_contains_remote_call(exc_info.__traceback__):
            reason = api_pb2.APP_DISCONNECT_REASON_REMOTE_EXCEPTION
        else:
            reason = api_pb2.APP_DISCONNECT_REASON_LOCAL_EXCEPTION
    else:
        reason = api_pb2.APP_DISCONNECT_REASON_ENTRYPOINT_COMPLETED
    if isinstance(exc_info, _CliUserExecutionError):
        exc_str = repr(exc_info.__cause__)
    elif exc_info:
        exc_str = repr(exc_info)
    else:
        exc_str = ""

    await _disconnect(client, app_id, reason, exc_str)


@asynccontextmanager
async def _run_app(
    app: _App,
    *,
    client: Optional[_Client] = None,
    detach: bool = False,
    environment_name: Optional[str] = None,
    interactive: bool = False,
) -> AsyncGenerator[_App, None]:
    """mdmd:hidden"""
    if environment_name is None:
        environment_name = typing.cast(str, config.get("environment"))

    if modal._runtime.execution_context._is_currently_importing:
        raise InvalidError("Can not run an app in global scope within a container")

    if app._running_app:
        raise InvalidError(
            "App is already running and can't be started again.\n"
            "You should not use `app.run` or `run_app` within a Modal `local_entrypoint`"
        )

    if app.description is None:
        import __main__

        if "__file__" in dir(__main__):
            app.set_description(os.path.basename(__main__.__file__))
        else:
            # Interactive mode does not have __file__.
            # https://docs.python.org/3/library/__main__.html#import-main
            app.set_description(__main__.__name__)

    if client is None:
        client = await _Client.from_env()

    app_state = api_pb2.APP_STATE_DETACHED if detach else api_pb2.APP_STATE_EPHEMERAL

    output_mgr = _get_output_manager()
    if interactive and output_mgr is None:
        warnings.warn(
            "Interactive mode is disabled because no output manager is active. "
            "Use 'with modal.enable_output():' to enable interactive mode and see logs.",
            stacklevel=2,
        )
        interactive = False

    running_app: RunningApp = await _init_local_app_new(
        client,
        app.description or "",
        environment_name=environment_name or "",
        app_state=app_state,
        interactive=interactive,
    )

    logs_timeout = config["logs_timeout"]
    async with app._set_local_app(client, running_app), TaskContext(grace=logs_timeout) as tc:
        # Start heartbeats loop to keep the client alive
        # we don't log heartbeat exceptions in detached mode
        # as losing the local connection will not affect the running app
        def heartbeat():
            return _heartbeat(client, running_app.app_id)

        heartbeat_loop = tc.infinite_loop(heartbeat, sleep=HEARTBEAT_INTERVAL, log_exception=not detach)
        logs_loop: Optional[asyncio.Task] = None

        if output_mgr is not None:
            # Defer import so this module is rich-safe
            # TODO(michael): The get_app_logs_loop function is itself rich-safe aside from accepting an OutputManager
            # as an argument, so with some refactoring we could avoid the need for this deferred import.
            from modal._output import get_app_logs_loop

            with output_mgr.make_live(output_mgr.step_progress("Initializing...")):
                initialized_msg = (
                    f"Initialized. [grey70]View run at [underline]{running_app.app_page_url}[/underline][/grey70]"
                )
                output_mgr.print(output_mgr.step_completed(initialized_msg))
                output_mgr.update_app_page_url(running_app.app_page_url or "ERROR:NO_APP_PAGE")

            # Start logs loop

            logs_loop = tc.create_task(
                get_app_logs_loop(client, output_mgr, app_id=running_app.app_id, app_logs_url=running_app.app_logs_url)
            )

        try:
            # Create all members
            await _create_all_objects(client, running_app, app._functions, app._classes, environment_name)

            # Publish the app
            await _publish_app(client, running_app, app_state, app._functions, app._classes)
        except asyncio.CancelledError as e:
            # this typically happens on sigint/ctrl-C during setup (the KeyboardInterrupt happens in the main thread)
            if output_mgr := _get_output_manager():
                output_mgr.print("Aborting app initialization...\n")

            await _status_based_disconnect(client, running_app.app_id, e)
            raise
        except BaseException as e:
            await _status_based_disconnect(client, running_app.app_id, e)
            raise

        try:
            # Show logs from dynamically created images.
            # TODO: better way to do this
            if output_mgr := _get_output_manager():
                output_mgr.enable_image_logs()

            # Yield to context
            if output_mgr := _get_output_manager():
                with output_mgr.show_status_spinner():
                    yield app
            else:
                yield app
            # successful completion!
            heartbeat_loop.cancel()
            await _status_based_disconnect(client, running_app.app_id, exc_info=None)
        except KeyboardInterrupt as e:
            # this happens only if sigint comes in during the yield block above
            if detach:
                if output_mgr := _get_output_manager():
                    output_mgr.print(output_mgr.step_completed("Shutting down Modal client."))
                    output_mgr.print(
                        "The detached app keeps running. You can track its progress at: "
                        f"[magenta]{running_app.app_page_url}[/magenta]"
                        ""
                    )
                if logs_loop:
                    logs_loop.cancel()
                await _status_based_disconnect(client, running_app.app_id, e)
            else:
                if output_mgr := _get_output_manager():
                    output_mgr.print(
                        "Disconnecting from Modal - This will terminate your Modal app in a few seconds.\n"
                    )
                await _status_based_disconnect(client, running_app.app_id, e)
                if logs_loop:
                    try:
                        await asyncio.wait_for(logs_loop, timeout=logs_timeout)
                    except asyncio.TimeoutError:
                        logger.warning("Timed out waiting for final app logs.")

                if output_mgr:
                    output_mgr.print(
                        output_mgr.step_completed(
                            "App aborted. "
                            f"[grey70]View run at [underline]{running_app.app_page_url}[/underline][/grey70]"
                        )
                    )
            return
        except BaseException as e:
            logger.info("Exception during app run")
            await _status_based_disconnect(client, running_app.app_id, e)
            raise

        # wait for logs gracefully, even though the task context would do the same
        # this allows us to log a more specific warning in case the app doesn't
        # provide all logs before exit
        if logs_loop:
            try:
                await asyncio.wait_for(logs_loop, timeout=logs_timeout)
            except asyncio.TimeoutError:
                logger.warning("Timed out waiting for final app logs.")

    if output_mgr := _get_output_manager():
        output_mgr.print(
            output_mgr.step_completed(
                f"App completed. [grey70]View run at [underline]{running_app.app_page_url}[/underline][/grey70]"
            )
        )


async def _serve_update(
    app: _App,
    existing_app_id: str,
    is_ready: Event,
    environment_name: str,
) -> None:
    """mdmd:hidden"""
    # Used by child process to reinitialize a served app
    client = await _Client.from_env()
    try:
        running_app: RunningApp = await _init_local_app_existing(client, existing_app_id, environment_name)

        # Create objects
        await _create_all_objects(
            client,
            running_app,
            app._functions,
            app._classes,
            environment_name,
        )

        # Publish the updated app
        await _publish_app(client, running_app, api_pb2.APP_STATE_UNSPECIFIED, app._functions, app._classes)

        # Communicate to the parent process
        is_ready.set()
    except asyncio.exceptions.CancelledError:
        # Stopped by parent process
        pass


@dataclasses.dataclass(frozen=True)
class DeployResult:
    """Dataclass representing the result of deploying an app."""

    app_id: str
    app_page_url: str
    app_logs_url: str
    warnings: list[str]


async def _deploy_app(
    app: _App,
    name: Optional[str] = None,
    namespace: Any = api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
    client: Optional[_Client] = None,
    environment_name: Optional[str] = None,
    tag: str = "",
) -> DeployResult:
    """Deploy an app and export its objects persistently.

    Typically, using the command-line tool `modal deploy <module or script>`
    should be used, instead of this method.

    **Usage:**

    ```python
    if __name__ == "__main__":
        deploy_app(app)
    ```

    Deployment has two primary purposes:

    * Persists all of the objects in the app, allowing them to live past the
      current app run. For schedules this enables headless "cron"-like
      functionality where scheduled functions continue to be invoked after
      the client has disconnected.
    * Allows for certain kinds of these objects, _deployment objects_, to be
      referred to and used by other apps.
    """
    if environment_name is None:
        environment_name = typing.cast(str, config.get("environment"))

    name = name or app.name
    if not name:
        raise InvalidError(
            "You need to either supply an explicit deployment name to the deploy command, "
            "or have a name set on the app.\n"
            "\n"
            "Examples:\n"
            'app.deploy("some_name")\n\n'
            "or\n"
            'app = App("some-name")'
        )
    else:
        check_object_name(name, "App")

    if tag and not is_valid_tag(tag):
        raise InvalidError(
            f"Deployment tag {tag!r} is invalid."
            "\n\nTags may only contain alphanumeric characters, dashes, periods, and underscores, "
            "and must be 50 characters or less"
        )

    if client is None:
        client = await _Client.from_env()

    t0 = time.time()

    # Get git information to track deployment history
    commit_info_task = asyncio.create_task(get_git_commit_info())

    running_app: RunningApp = await _init_local_app_from_name(
        client, name, namespace, environment_name=environment_name
    )

    async with TaskContext(0) as tc:
        # Start heartbeats loop to keep the client alive
        def heartbeat():
            return _heartbeat(client, running_app.app_id)

        tc.infinite_loop(heartbeat, sleep=HEARTBEAT_INTERVAL)

        try:
            # Create all members
            await _create_all_objects(
                client,
                running_app,
                app._functions,
                app._classes,
                environment_name=environment_name,
            )

            commit_info = None
            try:
                commit_info = await commit_info_task
            except Exception as e:
                logger.debug("Failed to get git commit info", exc_info=e)

            app_url, warnings = await _publish_app(
                client,
                running_app,
                api_pb2.APP_STATE_DEPLOYED,
                app._functions,
                app._classes,
                name,
                tag,
                commit_info,
            )
        except Exception as e:
            # Note that AppClientDisconnect only stops the app if it's still initializing, and is a no-op otherwise.
            await _disconnect(client, running_app.app_id, reason=api_pb2.APP_DISCONNECT_REASON_DEPLOYMENT_EXCEPTION)
            raise e

    if output_mgr := _get_output_manager():
        t = time.time() - t0
        output_mgr.print(output_mgr.step_completed(f"App deployed in {t:.3f}s! 🎉"))
        output_mgr.print(f"\nView Deployment: [magenta]{app_url}[/magenta]")
    return DeployResult(
        app_id=running_app.app_id,
        app_page_url=running_app.app_page_url,
        app_logs_url=running_app.app_logs_url,  # type: ignore
        warnings=[warning.message for warning in warnings],
    )


async def _interactive_shell(
    _app: _App, cmds: list[str], environment_name: str = "", pty: bool = True, **kwargs: Any
) -> None:
    """Run an interactive shell (like `bash`) within the image for this app.

    This is useful for online debugging and interactive exploration of the
    contents of this image. If `cmd` is optionally provided, it will be run
    instead of the default shell inside this image.

    **Example**

    ```python
    import modal

    app = modal.App(image=modal.Image.debian_slim().apt_install("vim"))
    ```

    You can now run this using

    ```
    modal shell script.py --cmd /bin/bash
    ```

    When calling programmatically, `kwargs` are passed to `Sandbox.create()`.
    """

    client = await _Client.from_env()
    async with _run_app(_app, client=client, environment_name=environment_name):
        sandbox_cmds = cmds if len(cmds) > 0 else ["/bin/bash"]
        sandbox_env = {
            "MODAL_TOKEN_ID": config["token_id"],
            "MODAL_TOKEN_SECRET": config["token_secret"],
            "MODAL_ENVIRONMENT": _get_environment_name(),
        }
        secrets = kwargs.pop("secrets", []) + [_Secret.from_dict(sandbox_env)]
        with enable_output():  # show any image build logs
            sandbox = await _Sandbox.create(
                "sleep",
                "100000",
                app=_app,
                secrets=secrets,
                **kwargs,
            )

        try:
            if pty:
                container_process = await sandbox.exec(
                    *sandbox_cmds, pty_info=get_pty_info(shell=True) if pty else None
                )
                await container_process.attach()
            else:
                container_process = await sandbox.exec(
                    *sandbox_cmds, stdout=StreamType.STDOUT, stderr=StreamType.STDOUT
                )
                await container_process.wait()
        except InteractiveTimeoutError:
            # Check on status of Sandbox. It may have crashed, causing connection failure.
            req = api_pb2.SandboxWaitRequest(sandbox_id=sandbox._object_id, timeout=0)
            resp = await retry_transient_errors(sandbox._client.stub.SandboxWait, req)
            if resp.result.exception:
                raise RemoteError(resp.result.exception)
            else:
                raise


def _run_stub(*args: Any, **kwargs: Any):
    """mdmd:hidden
    `run_stub` has been renamed to `run_app` and is deprecated. Please update your code.
    """
    deprecation_error(
        (2024, 5, 1), "`run_stub` has been renamed to `run_app` and is deprecated. Please update your code."
    )


def _deploy_stub(*args: Any, **kwargs: Any):
    """mdmd:hidden"""
    message = "`deploy_stub` has been renamed to `deploy_app`. Please update your code."
    deprecation_error((2024, 5, 1), message)


run_app = synchronize_api(_run_app)
serve_update = synchronize_api(_serve_update)
deploy_app = synchronize_api(_deploy_app)
interactive_shell = synchronize_api(_interactive_shell)
run_stub = synchronize_api(_run_stub)
deploy_stub = synchronize_api(_deploy_stub)
