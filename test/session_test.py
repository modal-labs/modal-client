import pytest

from modal import Queue, Session
from modal._session_singleton import get_default_session
from modal._session_state import SessionState


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
        q = Queue(session=session)
        await q.put("foo")
        await q.put("bar")
        assert await q.get() == "foo"
        assert await q.get() == "bar"
