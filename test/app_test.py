import pytest

from modal import App, Queue, run
from modal._app_singleton import get_default_app
from modal._app_state import AppState
from modal.exception import ExecutionError, NotFoundError


def test_app(reset_global_apps):
    app_a = App()
    app_b = App()
    assert app_a != app_b
    app_default = get_default_app()
    assert app_default != app_a
    assert app_default != app_b


def test_common_app(reset_global_apps):
    App.initialize_container_app()
    app_a = App()
    app_a.state = AppState.RUNNING  # Dummy to make sure constructor isn't run twice
    app_b = App()
    assert app_a == app_b
    assert app_b.state == AppState.RUNNING
    app_default = get_default_app()
    assert app_default == app_a


@pytest.mark.asyncio
async def test_create_object(servicer, client):
    app = App()
    async with app.run(client=client):
        q = await Queue.create(app=app)
        await q.put("foo")
        await q.put("bar")
        assert await q.get() == "foo"
        assert await q.get() == "bar"


@pytest.mark.asyncio
async def test_persistent_object(servicer, client):
    app_1 = App()
    async with app_1.run(client=client):
        q_1 = await Queue.create(app=app_1)
        assert q_1.object_id == "qu-1"
        await app_1.deploy("my-queue", q_1)

    app_2 = App()
    async with app_2.run(client=client):
        q_2 = await app_2.include("my-queue")
        assert isinstance(q_2, Queue)
        assert q_2.object_id == "qu-1"

        with pytest.raises(NotFoundError):
            await app_2.include("bazbazbaz")


def test_global_run(reset_global_apps, servicer, client):
    with run(client=client):
        q = Queue.create()
        assert q.object_id == "qu-1"


def test_run_inside_container(reset_global_apps, servicer, client):
    App.initialize_container_app()
    with pytest.raises(ExecutionError):
        with run(client=client):
            pass
