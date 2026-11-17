# Copyright Modal Labs 2022
import sys
import types

from modal._ipython import is_interactive_ipython
from modal_proto import api_pb2


def test_is_interactive_ipython(monkeypatch):
    monkeypatch.setitem(sys.modules, "IPython", types.SimpleNamespace(get_ipython=lambda: object()))
    assert is_interactive_ipython() is True
    monkeypatch.setitem(sys.modules, "IPython", types.SimpleNamespace(get_ipython=lambda: None))
    assert is_interactive_ipython() is False
    monkeypatch.delitem(sys.modules, "IPython", raising=False)
    assert is_interactive_ipython() is False


def test_notebook_function(client, servicer, monkeypatch):
    module = types.ModuleType("__main__")
    monkeypatch.setitem(sys.modules, "__main__", module)
    exec(
        """
import modal

app = modal.App()

@app.function()
def f():
    return 42
""",
        module.__dict__,
    )
    with module.app.run(client=client):
        module.f.remote()
        definition = servicer.app_functions[module.f.object_id]
    assert definition.definition_type == api_pb2.Function.DEFINITION_TYPE_SERIALIZED
    assert definition.function_serialized
