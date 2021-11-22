import pytest

from polyester import Session
from polyester.session_state import SessionState


@pytest.fixture
def common_session():
    Session.initialize_common()
    yield
    Session.initialize_common(unset=True)


def test_session():
    session_a = Session()
    session_b = Session()
    assert session_a != session_b


def test_common_session(common_session):
    session_a = Session()
    session_a.state = SessionState.RUNNING  # Dummy to make sure constructor isn't run twice
    session_b = Session()
    assert session_a == session_b
    assert session_b.state == SessionState.RUNNING
