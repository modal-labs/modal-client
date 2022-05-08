import pytest

import modal.exception
from modal import App
from modal._app_state import AppState
from modal.aio import AioApp, AioQueue
from modal.app import _App
from modal.exception import NotFoundError


def test_app(reset_global_apps):
    app_a = _App()
    app_b = _App()
    assert app_a != app_b


def test_common_app(reset_global_apps):
    _App._initialize_container_app()
    app_a = _App()
    app_a.state = AppState.RUNNING  # Dummy to make sure constructor isn't run twice
    app_b = _App()
    assert app_a == app_b
    assert app_b.state == AppState.RUNNING


@pytest.mark.asyncio
async def test_create_object(servicer, aio_client):
    app = AioApp()
    async with app.run(client=aio_client):
        q = await AioQueue.create(app=app)
        await q.put("foo")
        await q.put("bar")
        assert await q.get() == "foo"
        assert await q.get() == "bar"


@pytest.mark.asyncio
async def test_persistent_object(servicer, aio_client):
    app_1 = AioApp()
    app_1["q_1"] = AioQueue(app=app_1)
    await app_1.deploy("my-queue", client=aio_client)

    app_2 = AioApp()
    async with app_2.run(client=aio_client):
        q_2 = await app_2.include("my-queue")
        assert isinstance(q_2, AioQueue)
        assert q_2.object_id == "qu-1"

        with pytest.raises(NotFoundError):
            await app_2.include("bazbazbaz")


def square(x):
    return x**2


@pytest.mark.asyncio
async def test_redeploy(servicer, aio_client):
    app = AioApp()
    app.function(square)
    f_name = "client_test.app_test.square"

    # Deploy app
    await app.deploy("my-app", client=aio_client)
    assert app.app_id == "ap-1"
    assert servicer.app_objects["ap-1"][f_name] == "fu-1"

    # Redeploy, make sure all ids are the same
    await app.deploy("my-app", client=aio_client)
    assert app.app_id == "ap-1"
    assert servicer.app_objects["ap-1"][f_name] == "fu-1"

    # Deploy to a different name, ids should change
    await app.deploy("my-app-xyz", client=aio_client)
    assert app.app_id == "ap-2"
    assert servicer.app_objects["ap-2"][f_name] == "fu-2"


# Should exit without waiting for the logs grace period.
@pytest.mark.timeout(1)
def test_create_object_exception(servicer, client):
    servicer.function_create_error = True

    app = App()

    @app.function
    def f():
        pass

    with pytest.raises(Exception):
        with app.run(client=client):
            pass


def test_deploy_falls_back_to_app_name(servicer, client):
    named_app = App(name="foo_app")
    named_app.deploy(client=client)
    assert "foo_app" in servicer.deployed_apps


def test_deploy_uses_deployment_name_if_specified(servicer, client):
    named_app = App(name="foo_app")
    named_app.deploy("bar_app", client=client)
    assert "bar_app" in servicer.deployed_apps
    assert "foo_app" not in servicer.deployed_apps


def test_deploy_running_app_fails(servicer, client):
    app = App()
    with app.run(client=client):
        with pytest.raises(modal.exception.InvalidError):
            app.deploy(name="my_deployment", client=client)


@pytest.mark.skip(reason="revisit in a sec once the app state stuff is fixed")
def test_run_function_without_app_error():
    app = App()

    @app.function()
    def foo():
        pass

    with pytest.raises(modal.exception.InvalidError):
        foo()
