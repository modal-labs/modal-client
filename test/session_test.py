import pytest
from modal import Session
from modal.session_state import SessionState


def test_session():
    session_a = Session()
    session_b = Session()
    assert session_a != session_b


def test_common_session(reset_session_singleton):
    Session.initialize_singleton()
    session_a = Session()
    session_a.state = SessionState.RUNNING  # Dummy to make sure constructor isn't run twice
    session_b = Session()
    assert session_a == session_b
    assert session_b.state == SessionState.RUNNING
