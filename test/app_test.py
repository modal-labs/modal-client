# Copyright Modal Labs 2022
import asyncio
import logging
import pytest
import time

from google.protobuf.empty_pb2 import Empty
from grpclib import GRPCError, Status

from modal import App, Dict, Image, Mount, Secret, Stub, Volume, enable_output, web_endpoint
from modal._output import OutputManager
from modal.app import list_apps  # type: ignore
from modal.exception import DeprecationError, ExecutionError, InvalidError, NotFoundError
from modal.partial_function import _parse_custom_domains
from modal.runner import deploy_app, deploy_stub
from modal_proto import api_pb2

from .supports import module_1, module_2


@pytest.mark.asyncio
async def test_attrs(servicer, client):
    app = App()
    with pytest.raises(DeprecationError):
        app.d = Dict.from_name("xyz")


def square(x):
    return x**2


@pytest.mark.asyncio
async def test_redeploy(servicer, client):
    app = App(image=Image.debian_slim().pip_install("pandas"))
    app.function()(square)

    # Deploy app
    res = await deploy_app.aio(app, "my-app", client=client)
    assert res.app_id == "ap-1"
    assert servicer.app_objects["ap-1"]["square"] == "fu-1"
    assert servicer.app_state_history[res.app_id] == [api_pb2.APP_STATE_INITIALIZING, api_pb2.APP_STATE_DEPLOYED]

    # Redeploy, make sure all ids are the same
    res = await deploy_app.aio(app, "my-app", client=client)
    assert res.app_id == "ap-1"
    assert servicer.app_objects["ap-1"]["square"] == "fu-1"
    assert servicer.app_state_history[res.app_id] == [
        api_pb2.APP_STATE_INITIALIZING,
        api_pb2.APP_STATE_DEPLOYED,
        api_pb2.APP_STATE_DEPLOYED,
    ]

    # Deploy to a different name, ids should change
    res = await deploy_app.aio(app, "my-app-xyz", client=client)
    assert res.app_id == "ap-2"
    assert servicer.app_objects["ap-2"]["square"] == "fu-2"
    assert servicer.app_state_history[res.app_id] == [api_pb2.APP_STATE_INITIALIZING, api_pb2.APP_STATE_DEPLOYED]


def dummy():
    pass


# Should exit without waiting for the "logs_timeout" grace period.
@pytest.mark.timeout(5)
def test_create_object_exception(servicer, client):
    servicer.function_create_error = True

    app = App()
    app.function()(dummy)

    with pytest.raises(GRPCError) as excinfo:
        with app.run(client=client):
            pass

    assert excinfo.value.status == Status.INTERNAL


def test_deploy_falls_back_to_app_name(servicer, client):
    named_app = App(name="foo_app")
    deploy_app(named_app, client=client)
    assert "foo_app" in servicer.deployed_apps


def test_deploy_uses_deployment_name_if_specified(servicer, client):
    named_app = App(name="foo_app")
    deploy_app(named_app, "bar_app", client=client)
    assert "bar_app" in servicer.deployed_apps
    assert "foo_app" not in servicer.deployed_apps


def test_run_function_without_app_error():
    app = App()
    dummy_modal = app.function()(dummy)

    with pytest.raises(ExecutionError) as excinfo:
        dummy_modal.remote()

    assert "hydrated" in str(excinfo.value)


def test_is_inside_basic():
    app = App()
    with pytest.raises(DeprecationError, match="imports()"):
        app.is_inside()


def test_missing_attr():
    """Trying to call a non-existent function on the App should produce
    an understandable error message."""

    app = App()
    with pytest.raises(AttributeError):
        app.fun()  # type: ignore


def test_same_function_name(caplog):
    app = App()

    # Add first function
    with caplog.at_level(logging.WARNING):
        app.function()(module_1.square)
    assert len(caplog.records) == 0

    # Add second function: check warning
    with caplog.at_level(logging.WARNING):
        app.function()(module_2.square)
    assert len(caplog.records) == 1
    assert "module_1" in caplog.text
    assert "module_2" in caplog.text
    assert "square" in caplog.text


def test_run_state(client, servicer):
    app = App()
    with app.run(client=client):
        assert servicer.app_state_history[app.app_id] == [api_pb2.APP_STATE_INITIALIZING, api_pb2.APP_STATE_EPHEMERAL]


def test_deploy_state(client, servicer):
    app = App()
    res = deploy_app(app, "foobar", client=client)
    assert servicer.app_state_history[res.app_id] == [api_pb2.APP_STATE_INITIALIZING, api_pb2.APP_STATE_DEPLOYED]


def test_detach_state(client, servicer):
    app = App()
    with app.run(client=client, detach=True):
        assert servicer.app_state_history[app.app_id] == [api_pb2.APP_STATE_INITIALIZING, api_pb2.APP_STATE_DETACHED]


@pytest.mark.asyncio
async def test_grpc_protocol(client, servicer):
    app = App()
    async with app.run(client=client):
        await asyncio.sleep(0.01)  # wait for heartbeat
    assert len(servicer.requests) == 4
    assert isinstance(servicer.requests[0], Empty)  # ClientHello
    assert isinstance(servicer.requests[1], api_pb2.AppCreateRequest)
    assert isinstance(servicer.requests[2], api_pb2.AppHeartbeatRequest)
    assert isinstance(servicer.requests[3], api_pb2.AppClientDisconnectRequest)


async def web1(x):
    return {"square": x**2}


async def web2(x):
    return {"cube": x**3}


def test_registered_web_endpoints(client, servicer):
    app = App()
    app.function()(square)
    app.function()(web_endpoint()(web1))
    app.function()(web_endpoint()(web2))

    @app.cls(serialized=True)
    class Cls:
        @web_endpoint()
        def cls_web_endpoint(self):
            pass

    assert app.registered_web_endpoints == ["web1", "web2", "Cls.cls_web_endpoint"]


def test_init_types():
    with pytest.raises(InvalidError):
        # singular secret to plural argument
        App(secrets=Secret.from_dict())  # type: ignore
    with pytest.raises(InvalidError):
        # not a Secret Object
        App(secrets=[{"foo": "bar"}])  # type: ignore
    with pytest.raises(InvalidError):
        # should be an Image
        App(image=Secret.from_dict())  # type: ignore

    App(
        image=Image.debian_slim().pip_install("pandas"),
        secrets=[Secret.from_dict()],
        mounts=[Mount.from_local_file(__file__)],
    )


def test_set_image_on_app_as_attribute():
    # TODO: do we want to deprecate this syntax? It's kind of random for image to
    #     have a reserved name in the blueprint, and being the only of the construction
    #     arguments that can be set on the instance after construction
    custom_img = Image.debian_slim().apt_install("emacs")
    app = App(image=custom_img)
    assert app._get_default_image() == custom_img


def test_redeploy_delete_objects(servicer, client):
    # Deploy an app with objects d1 and d2
    app = App()
    app.function(name="d1")(dummy)
    app.function(name="d2")(dummy)
    res = deploy_app(app, "xyz", client=client)

    # Check objects
    assert set(servicer.app_objects[res.app_id].keys()) == set(["d1", "d2"])

    # Deploy an app with objects d2 and d3
    app = App()
    app.function(name="d2")(dummy)
    app.function(name="d3")(dummy)
    res = deploy_app(app, "xyz", client=client)

    # Make sure d1 is deleted
    assert set(servicer.app_objects[res.app_id].keys()) == set(["d2", "d3"])


@pytest.mark.asyncio
async def test_unhydrate(servicer, client):
    app = App()

    f = app.function()(dummy)

    assert not f.is_hydrated
    async with app.run(client=client):
        assert f.is_hydrated

    # After app finishes, it should unhydrate
    assert not f.is_hydrated


def test_keyboard_interrupt(servicer, client):
    app = App()
    app.function()(square)
    with app.run(client=client):
        # The exit handler should catch this interrupt and exit gracefully
        raise KeyboardInterrupt()


def test_function_image_positional():
    app = App()
    image = Image.debian_slim()

    with pytest.raises(InvalidError) as excinfo:

        @app.function(image)  # type: ignore
        def f():
            pass

    assert "function(image=image)" in str(excinfo.value)


def test_function_decorator_on_class():
    app = App()
    with pytest.raises(TypeError, match="cannot be used on a class"):

        @app.function()
        class Foo:
            pass


@pytest.mark.asyncio
async def test_deploy_disconnect(servicer, client):
    app = App()
    app.function(secrets=[Secret.from_name("nonexistent-secret")])(square)

    with pytest.raises(NotFoundError):
        await deploy_app.aio(app, "my-app", client=client)

    assert servicer.app_state_history["ap-1"] == [
        api_pb2.APP_STATE_INITIALIZING,
        api_pb2.APP_STATE_STOPPED,
    ]


def test_parse_custom_domains():
    assert len(_parse_custom_domains(None)) == 0
    assert len(_parse_custom_domains(["foo.com", "bar.com"])) == 2
    with pytest.raises(AssertionError):
        assert _parse_custom_domains("foo.com")


def test_hydrated_other_app_object_gets_referenced(servicer, client):
    app = App("my-app")
    with servicer.intercept() as ctx:
        with Volume.ephemeral(client=client) as vol:
            app.function(volumes={"/vol": vol})(dummy)  # implicitly load vol
            deploy_app(app, client=client)
            app_set_objects_req = ctx.pop_request("AppSetObjects")
            assert vol.object_id in app_set_objects_req.unindexed_object_ids


def test_hasattr():
    app = App()
    assert not hasattr(app, "xyz")


def test_app(client):
    app = App()
    square_modal = app.function()(square)

    with app.run(client=client):
        square_modal.remote(42)


def test_list_apps(client):
    apps_0 = [app.name for app in list_apps(client=client)]
    app = App()
    deploy_app(app, "foobar", client=client)
    apps_1 = [app.name for app in list_apps(client=client)]

    assert len(apps_1) == len(apps_0) + 1
    assert set(apps_1) - set(apps_0) == set(["foobar"])


def test_non_string_app_name():
    with pytest.raises(InvalidError, match="Must be string"):
        App(Image.debian_slim())  # type: ignore


def test_function_named_app():
    # Make sure we have a helpful warning when a user's function is named "app"
    # as it might collide with the App variable name (in particular if people
    # find & replace "stub" with "app").
    app = App()

    with pytest.warns(match="app"):

        @app.function(serialized=True)
        def app():
            ...


def test_stub():
    with pytest.warns(match="App"):
        Stub()


def test_deploy_stub(servicer, client):
    app = App("xyz")
    deploy_app(app, client=client)
    with pytest.warns(match="deploy_app"):
        deploy_stub(app, client=client)


def test_app_logs(servicer, client):
    app = App()
    f = app.function()(dummy)

    with app.run(client=client):
        f.remote()

    logs = [data for data in app._logs(client=client)]
    assert logs == ["hello, world (1)\n"]


def test_app_interactive(servicer, client, capsys):
    app = App()

    async def app_logs_pty(servicer, stream):
        await stream.recv_message()

        # Enable PTY
        await stream.send_message(api_pb2.TaskLogsBatch(pty_exec_id="ta-123"))

        # Send some data (should be written raw to stdout)
        log = api_pb2.TaskLogs(data="some data\n", file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT)
        await stream.send_message(api_pb2.TaskLogsBatch(entry_id="xyz", items=[log]))

        # Send an EOF
        await stream.send_message(api_pb2.TaskLogsBatch(eof=True, task_id="ta-123"))

        # Terminate app
        await stream.send_message(api_pb2.TaskLogsBatch(app_done=True))

    with servicer.intercept() as ctx:
        ctx.set_responder("AppGetLogs", app_logs_pty)

        with OutputManager.enable_output():
            with app.run(client=client):
                time.sleep(0.1)

    captured = capsys.readouterr()
    assert captured.out.endswith("\nsome data\n\r")


def test_show_progress_deprecations(client, monkeypatch):
    # Unset env used to disable warning
    monkeypatch.delenv("MODAL_DISABLE_APP_RUN_OUTPUT_WARNING")

    app = App()

    # If show_progress is not provided, and output is not enabled, warn
    with pytest.warns(DeprecationError, match="enable_output"):
        with app.run(client=client):
            assert OutputManager.get() is not None  # Should be auto-enabled

    # If show_progress is not provided, and output is enabled, no warning
    with enable_output():
        with app.run(client=client):
            pass

    # If show_progress is set to True, and output is not enabled, warn
    with pytest.warns(DeprecationError, match="enable_output"):
        with app.run(client=client, show_progress=True):
            assert OutputManager.get() is not None  # Should be auto-enabled

    # If show_progress is set to True, and output is enabled, warn the flag is superfluous
    with pytest.warns(DeprecationError, match="`show_progress=True` is deprecated"):
        with enable_output():
            with app.run(client=client, show_progress=True):
                pass

    # If show_progress is set to False, and output is not enabled, no warning
    # This mode is currently used to suppress deprecation warnings, but will in itself be deprecated later.
    with app.run(client=client, show_progress=False):
        assert OutputManager.get() is None

    # If show_progress is set to False, and output is enabled, warn that it has no effect
    with pytest.warns(DeprecationError, match="no effect"):
        with enable_output():
            with app.run(client=client, show_progress=False):
                pass
