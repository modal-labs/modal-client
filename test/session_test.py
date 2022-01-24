import pytest

from modal import Queue, Session, run
from modal._session_singleton import get_default_session
from modal._session_state import SessionState
from modal.exception import ExecutionError, NotFoundError


def test_session(reset_global_sessions):
    session_a = Session()
    session_b = Session()
    assert session_a != session_b
    session_default = get_default_session()
    assert session_default != session_a
    assert session_default != session_b


def test_common_session(reset_global_sessions):
    Session.initialize_container_session()
    session_a = Session()
    session_a.state = SessionState.RUNNING  # Dummy to make sure constructor isn't run twice
    session_b = Session()
    assert session_a == session_b
    assert session_b.state == SessionState.RUNNING
    session_default = get_default_session()
    assert session_default == session_a


@pytest.mark.asyncio
async def test_create_object(servicer, client):
    session = Session()
    async with session.run(client=client):
        q = await Queue.create(session=session)
        await q.put("foo")
        await q.put("bar")
        assert await q.get() == "foo"
        assert await q.get() == "bar"


@pytest.mark.asyncio
async def test_persistent_object(servicer, client):
    session_1 = Session()
    async with session_1.run(client=client):
        q_1 = await Queue.create(session=session_1)
        assert q_1.object_id == "qu-1"
        await session_1.share(q_1, "my-queue")

    session_2 = Session()
    async with session_2.run(client=client):
        q_2 = await session_2.use("my-queue")
        assert isinstance(q_2, Queue)
        assert q_2.object_id == "qu-1"

        with pytest.raises(NotFoundError):
            await session_2.use("bazbazbaz")


@pytest.mark.asyncio
async def test_persistent_object_2(servicer, client):
    # .deploy supersedes .share, so this test will take over the previous one
    session_1 = Session()
    async with session_1.run(client=client):
        q_1 = await Queue.create(session=session_1)
        assert q_1.object_id == "qu-1"
        await session_1.deploy("my-queue", q_1)


def test_global_run(reset_global_sessions, servicer, client):
    with run(client=client):
        q = Queue.create()
        assert q.object_id == "qu-1"


def test_run_inside_container(reset_global_sessions, servicer, client):
    Session.initialize_container_session()
    with pytest.raises(ExecutionError):
        with run(client=client):
            pass
