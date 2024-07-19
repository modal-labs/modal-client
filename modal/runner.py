# Copyright Modal Labs 2022
import asyncio
import dataclasses
import os
from multiprocessing.synchronize import Event
from typing import TYPE_CHECKING, Any, AsyncGenerator, Coroutine, Dict, List, Optional, TypeVar

from grpclib import GRPCError, Status
from rich.console import Console
from synchronicity.async_wrap import asynccontextmanager

from modal_proto import api_pb2

from ._output import OutputManager, get_app_logs_loop, step_completed, step_progress
from ._pty import get_pty_info
from ._resolver import Resolver
from ._sandbox_shell import connect_to_sandbox
from ._utils.async_utils import TaskContext, synchronize_api
from ._utils.grpc_utils import retry_transient_errors
from ._utils.name_utils import check_object_name, is_valid_tag
from .client import HEARTBEAT_INTERVAL, HEARTBEAT_TIMEOUT, _Client
from .config import config, logger
from .exception import (
    ExecutionError,
    InteractiveTimeoutError,
    InvalidError,
    RemoteError,
    _CliUserExecutionError,
    deprecation_warning,
)
from .execution_context import is_local
from .object import _Object
from .running_app import RunningApp
from .sandbox import _Sandbox

if TYPE_CHECKING:
    from .app import _App
else:
    _App = TypeVar("_App")


async def _heartbeat(client: _Client, app_id: str) -> None:
    request = api_pb2.AppHeartbeatRequest(app_id=app_id)
    # TODO(erikbern): we should capture exceptions here
    # * if request fails: destroy the client
    # * if server says the app is gone: print a helpful warning about detaching
    await retry_transient_errors(client.stub.AppHeartbeat, request, attempt_timeout=HEARTBEAT_TIMEOUT)


async def _init_local_app_existing(client: _Client, existing_app_id: str) -> RunningApp:
    # Get all the objects first
    obj_req = api_pb2.AppGetObjectsRequest(app_id=existing_app_id)
    obj_resp = await retry_transient_errors(client.stub.AppGetObjects, obj_req)
    app_page_url = f"https://modal.com/apps/{existing_app_id}"  # TODO (elias): this should come from the backend
    object_ids = {item.tag: item.object.object_id for item in obj_resp.items}
    return RunningApp(existing_app_id, app_page_url=app_page_url, tag_to_object_id=object_ids)


async def _init_local_app_new(
    client: _Client,
    description: str,
    app_state: int,
    environment_name: str = "",
    interactive: bool = False,
) -> RunningApp:
    app_req = api_pb2.AppCreateRequest(
        description=description,
        environment_name=environment_name,
        app_state=app_state,
    )
    app_resp = await retry_transient_errors(client.stub.AppCreate, app_req)
    app_page_url = app_resp.app_logs_url
    logger.debug(f"Created new app with id {app_resp.app_id}")
    return RunningApp(
        app_resp.app_id, app_page_url=app_page_url, environment_name=environment_name, interactive=interactive
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
        return await _init_local_app_existing(client, existing_app_id)
    else:
        return await _init_local_app_new(
            client, name, api_pb2.APP_STATE_INITIALIZING, environment_name=environment_name
        )


async def _create_all_objects(
    client: _Client,
    running_app: RunningApp,
    indexed_objects: Dict[str, _Object],
    new_app_state: int,
    environment_name: str,
) -> None:
    """Create objects that have been defined but not created on the server."""
    if not client.authenticated:
        raise ExecutionError("Objects cannot be created with an unauthenticated client")

    resolver = Resolver(
        client,
        environment_name=environment_name,
        app_id=running_app.app_id,
    )
    with resolver.display():
        # Get current objects, and reset all objects
        tag_to_object_id = running_app.tag_to_object_id
        running_app.tag_to_object_id = {}

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
            if obj.object_id is not None:
                tag_to_object_id[tag] = obj.object_id

        await TaskContext.gather(*(_preload(tag, obj) for tag, obj in indexed_objects.items()))

        async def _load(tag, obj):
            existing_object_id = tag_to_object_id.get(tag)
            await resolver.load(obj, existing_object_id)
            running_app.tag_to_object_id[tag] = obj.object_id

        await TaskContext.gather(*(_load(tag, obj) for tag, obj in indexed_objects.items()))

    # Create the app (and send a list of all tagged obs)
    # TODO(erikbern): we should delete objects from a previous version that are no longer needed
    # We just delete them from the app, but the actual objects will stay around
    indexed_object_ids = running_app.tag_to_object_id
    assert indexed_object_ids == running_app.tag_to_object_id
    all_objects = resolver.objects()

    unindexed_object_ids = list(set(obj.object_id for obj in all_objects) - set(running_app.tag_to_object_id.values()))
    req_set = api_pb2.AppSetObjectsRequest(
        app_id=running_app.app_id,
        indexed_object_ids=indexed_object_ids,
        unindexed_object_ids=unindexed_object_ids,
        new_app_state=new_app_state,  # type: ignore
    )
    await retry_transient_errors(client.stub.AppSetObjects, req_set)


async def _disconnect(
    client: _Client,
    app_id: str,
    reason: "api_pb2.AppDisconnectReason.ValueType",
    exc_str: Optional[str] = None,
) -> None:
    """Tell the server the client has disconnected for this app. Terminates all running tasks
    for ephemeral apps."""

    if exc_str:
        exc_str = exc_str[:1000]  # Truncate to 1000 chars

    logger.debug("Sending app disconnect/stop request")
    req_disconnect = api_pb2.AppClientDisconnectRequest(app_id=app_id, reason=reason, exception=exc_str)
    await retry_transient_errors(client.stub.AppClientDisconnect, req_disconnect)
    logger.debug("App disconnected")


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
        environment_name = config.get("environment")

    if not is_local():
        raise InvalidError(
            "Can not run an app from within a container."
            " Are you calling app.run() directly?"
            " Consider using the `modal run` shell command."
        )
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
    running_app: RunningApp = await _init_local_app_new(
        client,
        app.description,
        environment_name=environment_name,
        app_state=app_state,
        interactive=interactive,
    )
    async with app._set_local_app(client, running_app), TaskContext(grace=config["logs_timeout"]) as tc:
        # Start heartbeats loop to keep the client alive
        # we don't log heartbeat exceptions in detached mode
        # as losing the local connection will not affect the running app
        tc.infinite_loop(
            lambda: _heartbeat(client, running_app.app_id), sleep=HEARTBEAT_INTERVAL, log_exception=not detach
        )

        if output_mgr := OutputManager.get():
            with output_mgr.make_live(step_progress("Initializing...")):
                initialized_msg = (
                    f"Initialized. [grey70]View run at [underline]{running_app.app_page_url}[/underline][/grey70]"
                )
                output_mgr.print(step_completed(initialized_msg))
                output_mgr.update_app_page_url(running_app.app_page_url)

            # Start logs loop
            logs_loop = tc.create_task(get_app_logs_loop(client, output_mgr, running_app.app_id))

        exc_info: Optional[BaseException] = None
        try:
            # Create all members
            await _create_all_objects(client, running_app, app._indexed_objects, app_state, environment_name)

            # Show logs from dynamically created images.
            # TODO: better way to do this
            if output_mgr := OutputManager.get():
                output_mgr.enable_image_logs()

            # Yield to context
            if output_mgr := OutputManager.get():
                with output_mgr.show_status_spinner():
                    yield app
            else:
                yield app
        except KeyboardInterrupt as e:
            exc_info = e
            # mute cancellation errors on all function handles to prevent exception spam
            for obj in app.registered_functions.values():
                obj._set_mute_cancellation(True)

            if detach:
                if output_mgr := OutputManager.get():
                    output_mgr.print(step_completed("Shutting down Modal client."))
                    output_mgr.print(
                        "The detached app keeps running. You can track its progress at: "
                        f"[magenta]{running_app.app_page_url}[/magenta]"
                        ""
                    )
                    logs_loop.cancel()
            else:
                if output_mgr := OutputManager.get():
                    output_mgr.print(
                        step_completed(
                            "App aborted. "
                            f"[grey70]View run at [underline]{running_app.app_page_url}[/underline][/grey70]"
                        )
                    )
                    output_mgr.print(
                        "Disconnecting from Modal - This will terminate your Modal app in a few seconds.\n"
                    )
        except BaseException as e:
            exc_info = e
            raise e
        finally:
            if isinstance(exc_info, KeyboardInterrupt):
                reason = api_pb2.APP_DISCONNECT_REASON_KEYBOARD_INTERRUPT
            elif exc_info is not None:
                reason = api_pb2.APP_DISCONNECT_REASON_LOCAL_EXCEPTION
            else:
                reason = api_pb2.APP_DISCONNECT_REASON_ENTRYPOINT_COMPLETED

            if isinstance(exc_info, _CliUserExecutionError):
                exc_str = repr(exc_info.__cause__)
            elif exc_info:
                exc_str = repr(exc_info)
            else:
                exc_str = ""

            await _disconnect(client, running_app.app_id, reason, exc_str)
            app._uncreate_all_objects()

    if output_mgr := OutputManager.get():
        output_mgr.print(
            step_completed(
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
        running_app: RunningApp = await _init_local_app_existing(client, existing_app_id)

        # Create objects
        await _create_all_objects(
            client,
            running_app,
            app._indexed_objects,
            api_pb2.APP_STATE_UNSPECIFIED,
            environment_name,
        )

        # Communicate to the parent process
        is_ready.set()
    except asyncio.exceptions.CancelledError:
        # Stopped by parent process
        pass


@dataclasses.dataclass(frozen=True)
class DeployResult:
    """Dataclass representing the result of deploying an app."""

    app_id: str


async def _deploy_app(
    app: _App,
    name: Optional[str] = None,
    namespace: Any = api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
    client: Optional[_Client] = None,
    environment_name: Optional[str] = None,
    tag: Optional[str] = None,
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
        environment_name = config.get("environment")

    if name is None:
        name = app.name
    if name is None:
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

    if tag is not None and not is_valid_tag(tag):
        raise InvalidError(
            f"Tag {tag} is invalid."
            "\n\nTags may only contain alphanumeric characters, dashes, periods, and underscores, "
            "and must be 50 characters or less"
        )

    if client is None:
        client = await _Client.from_env()

    running_app: RunningApp = await _init_local_app_from_name(
        client, name, namespace, environment_name=environment_name
    )

    async with TaskContext(0) as tc:
        # Start heartbeats loop to keep the client alive
        tc.infinite_loop(lambda: _heartbeat(client, running_app.app_id), sleep=HEARTBEAT_INTERVAL)

        # Don't change the app state - deploy state is set by AppDeploy
        post_init_state = api_pb2.APP_STATE_UNSPECIFIED

        try:
            # Create all members
            await _create_all_objects(
                client,
                running_app,
                app._indexed_objects,
                post_init_state,
                environment_name=environment_name,
            )

            # Deploy app
            # TODO(erikbern): not needed if the app already existed
            deploy_req = api_pb2.AppDeployRequest(
                app_id=running_app.app_id,
                name=name,
                tag=tag,
                namespace=namespace,
                object_entity="ap",
                visibility=api_pb2.APP_DEPLOY_VISIBILITY_WORKSPACE,
            )
            try:
                deploy_response = await retry_transient_errors(client.stub.AppDeploy, deploy_req)
            except GRPCError as exc:
                if exc.status == Status.INVALID_ARGUMENT:
                    raise InvalidError(exc.message)
                if exc.status == Status.FAILED_PRECONDITION:
                    raise InvalidError(exc.message)
                raise
            url = deploy_response.url
        except Exception as e:
            # Note that AppClientDisconnect only stops the app if it's still initializing, and is a no-op otherwise.
            await _disconnect(client, running_app.app_id, reason=api_pb2.APP_DISCONNECT_REASON_DEPLOYMENT_EXCEPTION)
            raise e

    if output_mgr := OutputManager.get():
        output_mgr.print(step_completed("App deployed! 🎉"))
        output_mgr.print(f"\nView Deployment: [magenta]{url}[/magenta]")
    return DeployResult(app_id=running_app.app_id)


async def _interactive_shell(_app: _App, cmds: List[str], environment_name: str = "", **kwargs: Any) -> None:
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

    ```bash
    modal shell script.py --cmd /bin/bash
    ```

    **kwargs will be passed into spawn_sandbox().
    """
    client = await _Client.from_env()
    async with _run_app(_app, client=client, environment_name=environment_name):
        console = Console()
        loading_status = console.status("Starting container...")
        loading_status.start()

        sandbox_cmds = cmds if len(cmds) > 0 else ["/bin/bash"]
        sb = await _Sandbox.create(*sandbox_cmds, pty_info=get_pty_info(shell=True), app=_app, **kwargs)
        for _ in range(40):
            await asyncio.sleep(0.5)
            resp = await sb._client.stub.SandboxGetTaskId(api_pb2.SandboxGetTaskIdRequest(sandbox_id=sb._object_id))
            if resp.task_id != "":
                break
            # else: sandbox hasn't been assigned a task yet
        else:
            loading_status.stop()
            raise InteractiveTimeoutError("Timed out while waiting for sandbox to start")

        loading_status.stop()
        try:
            await connect_to_sandbox(sb)
        except InteractiveTimeoutError:
            # Check on status of Sandbox. It may have crashed, causing connection failure.
            req = api_pb2.SandboxWaitRequest(sandbox_id=sb._object_id, timeout=0)
            resp = await retry_transient_errors(sb._client.stub.SandboxWait, req)
            if resp.result.exception:
                raise RemoteError(resp.result.exception)
            else:
                raise


def _run_stub(*args: Any, **kwargs: Any) -> AsyncGenerator[_App, None]:
    """mdmd:hidden
    `run_stub` has been renamed to `run_app` and is deprecated. Please update your code.
    """
    deprecation_warning(
        (2024, 5, 1), "`run_stub` has been renamed to `run_app` and is deprecated. Please update your code."
    )
    return _run_app(*args, **kwargs)


def _deploy_stub(*args: Any, **kwargs: Any) -> Coroutine[Any, Any, DeployResult]:
    """`deploy_stub` has been renamed to `deploy_app` and is deprecated. Please update your code."""
    deprecation_warning((2024, 5, 1), str(_deploy_stub.__doc__))
    return _deploy_app(*args, **kwargs)


run_app = synchronize_api(_run_app)
serve_update = synchronize_api(_serve_update)
deploy_app = synchronize_api(_deploy_app)
interactive_shell = synchronize_api(_interactive_shell)
run_stub = synchronize_api(_run_stub)
deploy_stub = synchronize_api(_deploy_stub)
