import pytest

import modal.exception
from modal import App
from modal._app_state import AppState
from modal.aio import AioApp, AioQueue
from modal.exception import ExecutionError, NotFoundError


def test_app(reset_global_apps):
    app_a = App()
    app_b = App()
    assert app_a != app_b


def test_common_app(reset_global_apps):
    App._initialize_container_app()
    app_a = App()
    app_a.state = AppState.RUNNING  # Dummy to make sure constructor isn't run twice
    app_b = App()
    assert app_a == app_b
    assert app_b.state == AppState.RUNNING


@pytest.mark.asyncio
async def test_create_object(servicer, client):
    app = AioApp()
    async with app.run(client=client):
        q = await AioQueue.create(app=app)
        await q.put("foo")
        await q.put("bar")
        assert await q.get() == "foo"
        assert await q.get() == "bar"


@pytest.mark.asyncio
async def test_persistent_object(servicer, client):
    app_1 = AioApp()
    async with app_1.run(client=client):
        q_1 = await AioQueue.create(app=app_1)
        assert q_1.object_id == "qu-1"
        await app_1.deploy("my-queue", q_1)

    app_2 = AioApp()
    async with app_2.run(client=client):
        q_2 = await app_2.include("my-queue")
        assert isinstance(q_2, AioQueue)
        assert q_2.object_id == "qu-1"

        with pytest.raises(NotFoundError):
            await app_2.include("bazbazbaz")


@pytest.mark.skip("TODO: how should this behave when we don't have global run?")
def test_run_inside_container(reset_global_apps, servicer, client):
    App._initialize_container_app()
    app = App()
    with pytest.raises(ExecutionError):
        with app.run(client=client):
            pass


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
    with named_app.run(client=client):
        named_app.deploy()
    assert "foo_app" in servicer.deployments


def test_deploy_uses_deployment_name_if_specified(servicer, client):
    named_app = App(name="foo_app")
    with named_app.run(client=client):
        named_app.deploy("bar_app")
    assert "bar_app" in servicer.deployments
    assert "foo_app" not in servicer.deployments


def test_deploy_without_run_fails(servicer, client):
    app = App()
    with pytest.raises(modal.exception.InvalidError):
        app.deploy(name="my_deployment")
