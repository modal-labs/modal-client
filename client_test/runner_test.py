import modal
from modal.runner import run_stub


def test_run_stub(servicer, client):
    dummy_stub = modal.Stub()
    with servicer.intercept() as ctx:
        with run_stub(dummy_stub, client=client):
            pass
    rpc_methods_called = [name for name, _ in ctx.calls]
    assert "AppCreate" in rpc_methods_called
    assert "AppSetObjects" in rpc_methods_called
    assert "AppClientDisconnect" in rpc_methods_called
