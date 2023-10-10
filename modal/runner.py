# Copyright Modal Labs 2022
import asyncio
import contextlib
import dataclasses
from multiprocessing.synchronize import Event
from typing import TYPE_CHECKING, AsyncGenerator, Optional, TypeVar

from modal_proto import api_pb2
from modal_utils.app_utils import is_valid_app_name
from modal_utils.async_utils import TaskContext, synchronize_api
from modal_utils.grpc_utils import retry_transient_errors

from . import _pty
from ._output import OutputManager, get_app_logs_loop, step_completed, step_progress
from .app import _LocalApp, is_local
from .client import HEARTBEAT_INTERVAL, HEARTBEAT_TIMEOUT, _Client
from .config import config
from .exception import InvalidError
from .functions import _Function

if TYPE_CHECKING:
    from .stub import _Stub
else:
    _Stub = TypeVar("_Stub")


async def _heartbeat(client, app_id):
    request = api_pb2.AppHeartbeatRequest(app_id=app_id)
    # TODO(erikbern): we should capture exceptions here
    # * if request fails: destroy the client
    # * if server says the app is gone: print a helpful warning about detaching
    await retry_transient_errors(client.stub.AppHeartbeat, request, attempt_timeout=HEARTBEAT_TIMEOUT)


@contextlib.asynccontextmanager
async def _run_stub(
    stub: _Stub,
    client: Optional[_Client] = None,
    stdout=None,
    show_progress: bool = True,
    detach: bool = False,
    output_mgr: Optional[OutputManager] = None,
    environment_name: Optional[str] = None,
    shell=False,
) -> AsyncGenerator[_Stub, None]:
    """mdmd:hidden"""
    if environment_name is None:
        environment_name = config.get("environment")

    if not is_local():
        raise InvalidError(
            "Can not run an app from within a container."
            " Are you calling stub.run() directly?"
            " Consider using the `modal run` shell command."
        )
    if stub._local_app:
        raise InvalidError(
            "App is already running and can't be started again.\n"
            "You should not use `stub.run` or `run_stub` within a Modal local_entrypoint"
        )

    if client is None:
        client = await _Client.from_env()
    if output_mgr is None:
        output_mgr = OutputManager(stdout, show_progress, "Running app...")
    post_init_state = api_pb2.APP_STATE_DETACHED if detach else api_pb2.APP_STATE_EPHEMERAL
    app = await _LocalApp._init_new(
        client,
        stub.description,
        detach=detach,
        deploying=False,
        environment_name=environment_name,
    )
    async with stub._set_local_app(app), TaskContext(grace=config["logs_timeout"]) as tc:
        # Start heartbeats loop to keep the client alive
        tc.infinite_loop(lambda: _heartbeat(client, app.app_id), sleep=HEARTBEAT_INTERVAL)

        with output_mgr.ctx_if_visible(output_mgr.make_live(step_progress("Initializing..."))):
            initialized_msg = f"Initialized. [grey70]View app at [underline]{app._app_page_url}[/underline][/grey70]"
            output_mgr.print_if_visible(step_completed(initialized_msg))
            output_mgr.update_app_page_url(app._app_page_url)

        # Start logs loop
        logs_loop = tc.create_task(get_app_logs_loop(app.app_id, client, output_mgr))

        try:
            # Create all members
            await app._create_all_objects(
                stub._blueprint, post_init_state, environment_name, shell=shell, output_mgr=output_mgr
            )

            # Update all functions client-side to have the output mgr
            for obj in stub.registered_functions.values():
                obj._set_output_mgr(output_mgr)

            # Update all the classes client-side to propagate output manager to their methods.
            for obj in stub.registered_classes.values():
                obj._set_output_mgr(output_mgr)

            # Show logs from dynamically created images.
            # TODO: better way to do this
            output_mgr.enable_image_logs()

            # Yield to context
            if stub._pty_input_stream:
                output_mgr._visible_progress = False
                async with _pty.write_stdin_to_pty_stream(stub._pty_input_stream):
                    yield stub
                output_mgr._visible_progress = True
            else:
                with output_mgr.show_status_spinner():
                    yield stub
        except KeyboardInterrupt:
            # mute cancellation errors on all function handles to prevent exception spam
            for obj in stub.registered_functions.values():
                obj._set_mute_cancellation(True)

            if detach:
                output_mgr.print_if_visible(step_completed("Shutting down Modal client."))
                output_mgr.print_if_visible(
                    f"""The detached app keeps running. You can track its progress at: [magenta]{app.log_url()}[/magenta]"""
                )
                logs_loop.cancel()
            else:
                output_mgr.print_if_visible(step_completed("App aborted."))
                output_mgr.print_if_visible(
                    "Disconnecting from Modal - This will terminate your Modal app in a few seconds.\n"
                )
        finally:
            await app.disconnect()
            stub._uncreate_all_objects()

    output_mgr.print_if_visible(step_completed("App completed."))


async def _serve_update(
    stub,
    existing_app_id: str,
    is_ready: Event,
    environment_name: str,
) -> None:
    """mdmd:hidden"""
    # Used by child process to reinitialize a served app
    client = await _Client.from_env()
    try:
        app = await _LocalApp._init_existing(client, existing_app_id)

        # Create objects
        output_mgr = OutputManager(None, True)
        await app._create_all_objects(
            stub._blueprint, api_pb2.APP_STATE_UNSPECIFIED, environment_name, output_mgr=output_mgr
        )

        # Communicate to the parent process
        is_ready.set()
    except asyncio.exceptions.CancelledError:
        # Stopped by parent process
        pass


@dataclasses.dataclass
class DeployResult:
    app_id: str


async def _deploy_stub(
    stub: _Stub,
    name: str = None,
    namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
    client=None,
    stdout=None,
    show_progress=True,
    object_entity="ap",
    environment_name: Optional[str] = None,
) -> DeployResult:
    """Deploy an app and export its objects persistently.

    Typically, using the command-line tool `modal deploy <module or script>`
    should be used, instead of this method.

    **Usage:**

    ```python
    if __name__ == "__main__":
        deploy_stub(stub)
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

    if not is_local():
        raise InvalidError("Cannot run a deploy from within a container.")
    if name is None:
        name = stub.name
    if name is None:
        raise InvalidError(
            "You need to either supply an explicit deployment name to the deploy command, or have a name set on the app.\n"
            "\n"
            "Examples:\n"
            'stub.deploy("some_name")\n\n'
            "or\n"
            'stub = Stub("some-name")'
        )

    if not is_valid_app_name(name):
        raise InvalidError(
            f"Invalid app name {name}. App names may only contain alphanumeric characters, dashes, periods, and underscores, and must be less than 64 characters in length. "
        )

    if client is None:
        client = await _Client.from_env()

    output_mgr = OutputManager(stdout, show_progress)

    app = await _LocalApp._init_from_name(client, name, namespace, environment_name=environment_name)

    async with TaskContext(0) as tc:
        # Start heartbeats loop to keep the client alive
        tc.infinite_loop(lambda: _heartbeat(client, app.app_id), sleep=HEARTBEAT_INTERVAL)

        # Don't change the app state - deploy state is set by AppDeploy
        post_init_state = api_pb2.APP_STATE_UNSPECIFIED

        # Create all members
        await app._create_all_objects(
            stub._blueprint, post_init_state, environment_name=environment_name, output_mgr=output_mgr
        )

        # Deploy app
        # TODO(erikbern): not needed if the app already existed
        url = await app.deploy(name, namespace, object_entity)

    output_mgr.print_if_visible(step_completed("App deployed! 🎉"))
    output_mgr.print_if_visible(f"\nView Deployment: [magenta]{url}[/magenta]")
    return DeployResult(app_id=app.app_id)


async def _interactive_shell(_function: _Function, cmd: str, environment_name: str = ""):
    """Run an interactive shell (like `bash`) within the image for this app.

    This is useful for online debugging and interactive exploration of the
    contents of this image. If `cmd` is optionally provided, it will be run
    instead of the default shell inside this image.

    **Example**

    ```python
    import modal

    stub = modal.Stub(image=modal.Image.debian_slim().apt_install("vim"))
    ```

    You can now run this using

    ```bash
    modal shell script.py --cmd /bin/bash
    ```
    """
    _stub = _function._stub
    _stub._add_pty_input_stream()  # TOOD(erikbern): slightly hacky
    async with _run_stub(_stub, environment_name=environment_name, shell=True):
        await _function.shell(cmd)


run_stub = synchronize_api(_run_stub)
serve_update = synchronize_api(_serve_update)
deploy_stub = synchronize_api(_deploy_stub)
interactive_shell = synchronize_api(_interactive_shell)
