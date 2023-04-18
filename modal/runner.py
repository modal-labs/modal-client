# Copyright Modal Labs 2022
import asyncio
import contextlib
from multiprocessing.synchronize import Event
from typing import AsyncGenerator, Optional

from modal_proto import api_pb2
from modal_utils.app_utils import is_valid_app_name
from modal_utils.async_utils import TaskContext
from modal_utils.grpc_utils import retry_transient_errors

from . import _pty
from ._output import OutputManager, step_completed, step_progress, get_app_logs_loop
from .app import _App, is_local
from .client import HEARTBEAT_INTERVAL, HEARTBEAT_TIMEOUT, _Client
from .config import config
from .exception import InvalidError
from .queue import _QueueHandle


async def _heartbeat(client, app_id):
    request = api_pb2.AppHeartbeatRequest(app_id=app_id)
    # TODO(erikbern): we should capture exceptions here
    # * if request fails: destroy the client
    # * if server says the app is gone: print a helpful warning about detaching
    await retry_transient_errors(client.stub.AppHeartbeat, request, attempt_timeout=HEARTBEAT_TIMEOUT)


@contextlib.asynccontextmanager
async def run_stub(
    stub,
    client: Optional[_Client] = None,
    stdout=None,
    show_progress: Optional[bool] = None,
    detach: bool = False,
    output_mgr: Optional[OutputManager] = None,
) -> AsyncGenerator[_App, None]:
    if not is_local():
        raise InvalidError(
            "Can not run an app from within a container."
            " Are you calling stub.run() directly?"
            " Consider using the `modal run` shell command."
        )
    if client is None:
        client = await _Client.from_env()
    if output_mgr is None:
        output_mgr = OutputManager(stdout, show_progress, "Running app...")
    post_init_state = api_pb2.APP_STATE_DETACHED if detach else api_pb2.APP_STATE_EPHEMERAL
    app = await _App._init_new(client, stub.description, detach=detach, deploying=False)
    async with stub._set_app(app), TaskContext(grace=config["logs_timeout"]) as tc:
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
            await app._create_all_objects(stub._blueprint, output_mgr, post_init_state)

            # Update all functions client-side to have the output mgr
            for tag, obj in stub._function_handles.items():
                obj._set_output_mgr(output_mgr)

            # Yield to context
            if stub._pty_input_stream:
                output_mgr._visible_progress = False
                handle = app._pty_input_stream
                assert isinstance(handle, _QueueHandle)
                async with _pty.write_stdin_to_pty_stream(handle):
                    yield app
                output_mgr._visible_progress = True
            else:
                with output_mgr.show_status_spinner():
                    yield app
        except KeyboardInterrupt:
            # mute cancellation errors on all function handles to prevent exception spam
            for tag, obj in stub._function_handles.items():
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

    output_mgr.print_if_visible(step_completed("App completed."))


async def serve_update(
    stub,
    existing_app_id: str,
    is_ready: Event,
) -> None:
    # Used by child process to reinitialize a served app
    client = await _Client.from_env()
    try:
        output_mgr = OutputManager(None, None)
        app = await _App._init_existing(client, existing_app_id)

        # Create objects
        await app._create_all_objects(stub._blueprint, output_mgr, api_pb2.APP_STATE_UNSPECIFIED)

        # Communicate to the parent process
        is_ready.set()
    except asyncio.exceptions.CancelledError:
        # Stopped by parent process
        pass


async def deploy_stub(
    stub,
    name: str = None,
    namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
    client=None,
    stdout=None,
    show_progress=None,
    object_entity="ap",
) -> _App:
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

    app = await _App._init_from_name(client, name, namespace)

    output_mgr = OutputManager(stdout, show_progress)

    async with TaskContext(0) as tc:
        # Start heartbeats loop to keep the client alive
        tc.infinite_loop(lambda: _heartbeat(client, app.app_id), sleep=HEARTBEAT_INTERVAL)

        # Don't change the app state - deploy state is set by AppDeploy
        post_init_state = api_pb2.APP_STATE_UNSPECIFIED

        # Create all members
        await app._create_all_objects(stub._blueprint, output_mgr, post_init_state)

        # Deploy app
        # TODO(erikbern): not needed if the app already existed
        url = await app.deploy(name, namespace, object_entity)

    output_mgr.print_if_visible(step_completed("App deployed! 🎉"))
    output_mgr.print_if_visible(f"\nView Deployment: [magenta]{url}[/magenta]")
    return app
