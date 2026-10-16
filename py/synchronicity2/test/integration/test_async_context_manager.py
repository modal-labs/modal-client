# Copyright Modal Labs 2026
"""Integration tests for async_context_manager_impl.py support file."""

import pytest
from pathlib import Path

from test.integration.test_utils import check_pyright


@pytest.mark.usefixtures("generated_wrappers")
def test_runtime_class_sync():
    import async_context_manager

    r = async_context_manager.AsyncResource("test")
    assert r.state == "init"
    with r as r2:
        assert r2.name == "test"
        assert r.state == "entered"
        assert r is r2
    assert r.state == "exited"


@pytest.mark.usefixtures("generated_wrappers")
@pytest.mark.asyncio
async def test_runtime_class_async():
    import async_context_manager

    r = async_context_manager.AsyncResource("test")
    assert r.state == "init"
    async with r as r2:
        assert r2.name == "test"
        assert r.state == "entered"
        assert r is r2
    assert r.state == "exited"


@pytest.mark.usefixtures("generated_wrappers")
def test_runtime_function_sync():
    import async_context_manager

    with async_context_manager.managed_value() as v:
        assert isinstance(v, async_context_manager.Connection)
        assert v.value == 42


@pytest.mark.usefixtures("generated_wrappers")
@pytest.mark.asyncio
async def test_runtime_function_async():
    import async_context_manager

    async with async_context_manager.managed_value() as v:
        assert isinstance(v, async_context_manager.Connection)
        assert v.value == 42


@pytest.mark.usefixtures("generated_wrappers")
def test_runtime_method_sync():
    import async_context_manager

    svc = async_context_manager.ServiceWithContextMethod()
    with svc.connect() as v:
        assert isinstance(v, async_context_manager.Connection)
        assert v.value == 99


@pytest.mark.usefixtures("generated_wrappers")
def test_runtime_method_sync_no_value():
    import async_context_manager

    svc = async_context_manager.ServiceWithContextMethod()
    with svc.no_value() as v:
        assert v is None


@pytest.mark.usefixtures("generated_wrappers")
@pytest.mark.asyncio
async def test_runtime_method_async_no_value():
    import async_context_manager

    svc = async_context_manager.ServiceWithContextMethod()
    async with svc.no_value() as v:
        assert v is None


@pytest.mark.usefixtures("generated_wrappers")
def test_runtime_classmethod_context_manager_sync():
    import async_context_manager

    with async_context_manager.ServiceWithFactoryContexts.connect_class() as v:
        assert isinstance(v, async_context_manager.Connection)
        assert v.value == 123


@pytest.mark.usefixtures("generated_wrappers")
def test_runtime_staticmethod_context_manager_sync():
    import async_context_manager

    with async_context_manager.ServiceWithFactoryContexts.connect_static() as v:
        assert isinstance(v, async_context_manager.Connection)
        assert v.value == 321


@pytest.mark.usefixtures("generated_wrappers")
@pytest.mark.asyncio
async def test_runtime_method_async():
    import async_context_manager

    svc = async_context_manager.ServiceWithContextMethod()
    async with svc.connect() as v:
        assert isinstance(v, async_context_manager.Connection)
        assert v.value == 99


@pytest.mark.usefixtures("generated_wrappers")
@pytest.mark.asyncio
async def test_runtime_classmethod_context_manager_async():
    import async_context_manager

    async with async_context_manager.ServiceWithFactoryContexts.connect_class() as v:
        assert isinstance(v, async_context_manager.Connection)
        assert v.value == 123


@pytest.mark.usefixtures("generated_wrappers")
@pytest.mark.asyncio
async def test_runtime_staticmethod_context_manager_async():
    import async_context_manager

    async with async_context_manager.ServiceWithFactoryContexts.connect_static() as v:
        assert isinstance(v, async_context_manager.Connection)
        assert v.value == 321


def test_pyright_implementation():
    import async_context_manager_impl

    check_pyright([Path(async_context_manager_impl.__file__)])


def test_pyright_wrapper():
    import async_context_manager

    check_pyright([Path(async_context_manager.__file__)])


def test_pyright_usage():
    from importlib.util import find_spec

    spec = find_spec("async_context_manager_typecheck")
    assert spec and spec.origin
    check_pyright([Path(spec.origin)])
