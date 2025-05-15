# Copyright Modal Labs 2022
import asyncio
import logging
import pytest
import time

from grpclib import GRPCError, Status

from modal import App, Image, Mount, Secret, Volume, enable_output, fastapi_endpoint, web_endpoint
from modal._partial_function import _parse_custom_domains
from modal._utils.async_utils import synchronizer
from modal.exception import DeprecationError, ExecutionError, InvalidError, NotFoundError
from modal.runner import run_app
from modal_proto import api_pb2

from .supports import module_1, module_2


def square(x):
    return x**2


@pytest.mark.asyncio
async def test_redeploy(servicer, client):
    app = App(image=Image.debian_slim().pip_install("pandas"))
    app.function()(square)

    # Deploy app
    await app.deploy.aio(name="my-app", client=client)
    assert app.app_id == "ap-1"
    assert servicer.app_objects["ap-1"]["square"] == "fu-1"
    assert servicer.app_state_history[app.app_id] == [api_pb2.APP_STATE_INITIALIZING, api_pb2.APP_STATE_DEPLOYED]

    # Redeploy, make sure all ids are the same
    await app.deploy.aio(name="my-app", client=client)
    assert app.app_id == "ap-1"
    assert servicer.app_objects["ap-1"]["square"] == "fu-1"
    assert servicer.app_state_history[app.app_id] == [
        api_pb2.APP_STATE_INITIALIZING,
        api_pb2.APP_STATE_DEPLOYED,
        api_pb2.APP_STATE_DEPLOYED,
    ]

    # Deploy to a different name, ids should change
    await app.deploy.aio(name="my-app-xyz", client=client)
    assert app.app_id == "ap-2"
    assert servicer.app_objects["ap-2"]["square"] == "fu-2"
    assert servicer.app_state_history[app.app_id] == [api_pb2.APP_STATE_INITIALIZING, api_pb2.APP_STATE_DEPLOYED]


def dummy():
    pass


# Should exit without waiting for the "logs_timeout" grace period.
@pytest.mark.timeout(5)
def test_create_object_internal_exception(servicer, client):
    servicer.function_create_error = GRPCError(Status.INTERNAL, "Function create failed")

    app = App()
    app.function()(dummy)

    with servicer.intercept() as ctx:
        with pytest.raises(GRPCError) as excinfo:
            with enable_output():  # this activates the log streaming loop, which could potentially hold up context exit
                with app.run(client=client):
                    pass

    assert len(ctx.get_requests("FunctionCreate")) == 4  # some retries are applied to internal errors
    assert excinfo.value.status == Status.INTERNAL
    assert len(ctx.get_requests("AppClientDisconnect")) == 1


@pytest.mark.timeout(5)
def test_create_object_invalid_exception(servicer, client):
    servicer.function_create_error = GRPCError(Status.INVALID_ARGUMENT, "something was invalid")

    app = App()
    app.function()(dummy)

    with servicer.intercept() as ctx:
        with pytest.raises(InvalidError, match="something was invalid"):  # error should be converted
            with enable_output():  # this activates the log streaming loop, which could potentially hold up context exit
                with app.run(client=client):
                    pass
    assert len(ctx.get_requests("FunctionCreate")) == 1  # no retries on an invalid request
    assert len(ctx.get_requests("AppClientDisconnect")) == 1


def test_deploy_falls_back_to_app_name(servicer, client):
    named_app = App(name="foo_app")
    named_app.deploy(client=client)
    assert "foo_app" in servicer.deployed_apps


def test_deploy_uses_deployment_name_if_specified(servicer, client):
    named_app = App(name="foo_app")
    named_app.deploy(name="bar_app", client=client)
    assert "bar_app" in servicer.deployed_apps
    assert "foo_app" not in servicer.deployed_apps


def test_run_function_without_app_error():
    app = App()
    dummy_modal = app.function()(dummy)

    with pytest.raises(ExecutionError) as excinfo:
        dummy_modal.remote()

    assert "hydrated" in str(excinfo.value)


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
    app.deploy(name="foobar", client=client)
    assert servicer.app_state_history[app.app_id] == [api_pb2.APP_STATE_INITIALIZING, api_pb2.APP_STATE_DEPLOYED]


def test_detach_state(client, servicer):
    app = App()
    with app.run(client=client, detach=True):
        assert servicer.app_state_history[app.app_id] == [api_pb2.APP_STATE_INITIALIZING, api_pb2.APP_STATE_DETACHED]


@pytest.mark.asyncio
async def test_grpc_protocol(client, servicer):
    app = App()
    async with app.run(client=client):
        await asyncio.sleep(0.01)  # wait for heartbeat
    assert len(servicer.requests) == 3
    assert isinstance(servicer.requests[0], api_pb2.AppCreateRequest)
    assert isinstance(servicer.requests[1], api_pb2.AppHeartbeatRequest)
    assert isinstance(servicer.requests[2], api_pb2.AppClientDisconnectRequest)


async def web1(x):
    return {"square": x**2}


async def web2(x):
    return {"cube": x**3}


def test_registered_fastapi_endpoints(client, servicer):
    app = App()
    app.function()(square)
    app.function()(fastapi_endpoint()(web1))
    app.function()(fastapi_endpoint()(web2))

    @app.cls(serialized=True)
    class Cls:
        @fastapi_endpoint()
        def web3(self):
            pass

    assert app.registered_web_endpoints == ["web1", "web2", "Cls.web3"]


def test_registered_legacy_web_endpoints(client, servicer):
    with pytest.warns(DeprecationError, match="fastapi_endpoint"):
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
        mounts=[Mount._from_local_file(__file__)],  # TODO: remove
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
    app.function(name="d1", serialized=True)(dummy)
    app.function(name="d2", serialized=True)(dummy)
    app.deploy(name="xyz", client=client)

    # Check objects
    assert set(servicer.app_objects[app.app_id].keys()) == {"d1", "d2"}

    # Deploy an app with objects d2 and d3
    app = App()
    app.function(name="d2", serialized=True)(dummy)
    app.function(name="d3", serialized=True)(dummy)
    app.deploy(name="xyz", client=client)

    # Make sure d1 is deleted
    assert set(servicer.app_objects[app.app_id].keys()) == {"d2", "d3"}


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
        await app.deploy.aio(name="my-app", client=client)

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
            app.deploy(client=client)
            function_create_req: api_pb2.FunctionCreateRequest = ctx.pop_request("FunctionCreate")
            assert vol.object_id in {obj.object_id for obj in function_create_req.function.object_dependencies}


def test_hasattr():
    app = App()
    assert not hasattr(app, "xyz")


def test_app(client):
    app = App()
    square_modal = app.function()(square)

    with app.run(client=client):
        square_modal.remote(42)


def test_non_string_app_name():
    with pytest.raises(InvalidError, match="Must be string"):
        App(Image.debian_slim())  # type: ignore


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

        with enable_output():
            with app.run(client=client):
                time.sleep(0.1)

    captured = capsys.readouterr()
    assert captured.out.endswith("\nsome data\n\r")


def test_app_interactive_no_output(servicer, client):
    app = App()

    with pytest.warns(match="Interactive mode is disabled because no output manager is active"):
        with app.run(client=client, interactive=True):
            # Verify that interactive mode was disabled
            assert not app.is_interactive


@pytest.mark.asyncio
async def test_deploy_from_container(servicer, container_client):
    app = App(image=Image.debian_slim().pip_install("pandas"))
    app.function()(square)

    # Deploy app
    await app.deploy.aio(name="my-app", client=container_client)
    assert servicer.app_objects["ap-1"]["square"] == "fu-1"
    assert servicer.app_state_history[app.app_id] == [api_pb2.APP_STATE_INITIALIZING, api_pb2.APP_STATE_DEPLOYED]


def test_app_create_bad_environment_name_error(client):
    environment_name = "this=is@not.allowed"
    app = App()
    with pytest.raises(InvalidError, match="Invalid Environment name"):
        with run_app(
            app, environment_name=environment_name, client=client
        ):  # TODO: why isn't environment_name an argument to app.run?
            pass

    assert len(asyncio.all_tasks(synchronizer._loop)) == 1  # no trailing tasks, except the `loop_inner` ever-task


def test_overriding_function_warning(caplog):
    app = App()

    @app.function(serialized=True)
    def func():  # type: ignore
        return 1

    assert len(caplog.messages) == 0

    app_2 = App()
    app_2.include(app)

    assert len(caplog.messages) == 0

    app_3 = App()

    app_3.include(app)
    app_3.include(app_2)

    assert len(caplog.messages) == 0

    app_4 = App()

    @app_4.function(serialized=True)  # type: ignore
    def func():  # noqa: F811
        return 2

    assert len(caplog.messages) == 0

    app_3.include(app_4)
    assert "Overriding existing function" in caplog.messages[0]


@pytest.mark.parametrize("name", ["", " ", "no way", "my-app!", "a" * 65])
def test_lookup_invalid_name(name):
    with pytest.raises(InvalidError, match="Invalid App name"):
        App.lookup(name)
