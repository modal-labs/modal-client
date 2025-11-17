# Copyright Modal Labs 2025
"""Unit tests for sync-in-async warning message rewriting logic."""

from modal._utils.async_utils import rewrite_sync_to_async


def test_rewrite_simple_call():
    """Test rewriting a simple method call."""
    code = "obj.method()"
    success, result = rewrite_sync_to_async(code, "method")
    assert success is True
    assert result == "await obj.method.aio()"


def test_rewrite_chained_call():
    """Test rewriting a chained method call."""
    code = "modal.Dict.objects.list(client=client)"
    success, result = rewrite_sync_to_async(code, "list")
    assert success is True
    assert result == "await modal.Dict.objects.list.aio(client=client)"


def test_rewrite_assignment():
    """Test rewriting an assignment statement."""
    code = "result = obj.method(arg)"
    success, result = rewrite_sync_to_async(code, "method")
    assert success is True
    assert result == "result = await obj.method.aio(arg)"


def test_rewrite_return_statement():
    """Test rewriting a return statement."""
    code = "return q.get()"
    success, result = rewrite_sync_to_async(code, "get")
    assert success is True
    assert result == "return await q.get.aio()"


def test_rewrite_return_with_args():
    """Test rewriting a return statement with arguments."""
    code = "return obj.fetch(key, default=None)"
    success, result = rewrite_sync_to_async(code, "fetch")
    assert success is True
    assert result == "return await obj.fetch.aio(key, default=None)"


def test_rewrite_yield_statement():
    """Test rewriting a yield statement."""
    code = "yield obj.next_item()"
    success, result = rewrite_sync_to_async(code, "next_item")
    assert success is True
    assert result == "yield await obj.next_item.aio()"


def test_rewrite_raise_statement():
    """Test rewriting a raise statement."""
    code = "raise obj.create_error()"
    success, result = rewrite_sync_to_async(code, "create_error")
    assert success is True
    assert result == "raise await obj.create_error.aio()"


def test_rewrite_with_whitespace():
    """Test rewriting with various whitespace."""
    code = "result  =  obj.method  (  )"
    success, result = rewrite_sync_to_async(code, "method")
    assert success is True
    # Note: whitespace before ( is not fully preserved, which is acceptable
    assert result == "result  =  await obj.method.aio(  )"


def test_rewrite_multiline_args():
    """Test rewriting with arguments on the same line."""
    code = "obj.method(arg1, arg2, kwarg=value)"
    success, result = rewrite_sync_to_async(code, "method")
    assert success is True
    assert result == "await obj.method.aio(arg1, arg2, kwarg=value)"


def test_rewrite_function_not_in_line():
    """Test fallback when function name not found in line."""
    code = "some_other_call()"
    success, result = rewrite_sync_to_async(code, "missing_func")
    assert success is False
    assert result == "await ...missing_func.aio(...)"


def test_rewrite_function_name_ambiguity():
    """Test handling when function name appears but not as a method call."""
    code = "# call list() here"
    success, result = rewrite_sync_to_async(code, "list")
    assert success is False
    assert result == "await ...list.aio(...)"


def test_rewrite_nested_call():
    """Test rewriting nested method calls (only rewrites first match)."""
    code = "obj.outer(obj.inner())"
    success, result = rewrite_sync_to_async(code, "outer")
    assert success is True
    assert result == "await obj.outer.aio(obj.inner())"


def test_rewrite_with_complex_args():
    """Test rewriting with complex arguments."""
    code = "result = db.query(f'SELECT * FROM {table}', timeout=30)"
    success, result = rewrite_sync_to_async(code, "query")
    assert success is True
    assert result == "result = await db.query.aio(f'SELECT * FROM {table}', timeout=30)"


def test_rewrite_indented():
    """Test rewriting with indentation."""
    code = "    result = obj.method()"
    success, result = rewrite_sync_to_async(code, "method")
    assert success is True
    # Note: lstrip() is called, so leading whitespace is removed from expression part
    assert result == "    result = await obj.method.aio()"


def test_rewrite_iterator_pattern():
    """Test __aiter__ pattern for 'for' loops."""
    code = "for x in obj.iterate():"
    success, result = rewrite_sync_to_async(code, "__aiter__")
    assert success is True
    assert result == "async for x in obj.iterate():"


def test_rewrite_context_manager_pattern():
    """Test __aenter__ pattern for 'with' statements."""
    code = "with obj.open() as f:"
    success, result = rewrite_sync_to_async(code, "__aenter__")
    assert success is True
    assert result == "async with obj.open() as f:"


def test_rewrite_property_access():
    """Test rewriting property access (no parentheses)."""
    code = "f.web_url"
    success, result = rewrite_sync_to_async(code, "web_url")
    assert success is True
    assert result == "await f.web_url"


def test_rewrite_property_with_comment():
    """Test rewriting property access with trailing comment."""
    code = "f.web_url  # expected to raise due to failing hydration"
    success, result = rewrite_sync_to_async(code, "web_url")
    assert success is True
    assert result == "await f.web_url  # expected to raise due to failing hydration"


def test_rewrite_property_in_assignment():
    """Test rewriting property access in an assignment."""
    code = "url = obj.web_url"
    success, result = rewrite_sync_to_async(code, "web_url")
    assert success is True
    assert result == "url = await obj.web_url"


def test_rewrite_fallback_on_complex_expression():
    """Test fallback when the expression is too complex to rewrite safely."""
    code = "if obj.check() and other.method():"
    success, result = rewrite_sync_to_async(code, "check")
    assert success is False
    # Should fall back to generic suggestion when 'if' keyword is present
    assert result == "await ...check.aio(...)"


def test_rewrite_fallback_with_and_keyword():
    """Test fallback with 'and' keyword."""
    code = "result = obj.check() and other.verify()"
    success, result = rewrite_sync_to_async(code, "check")
    assert success is False
    # Should fall back due to 'and' keyword
    assert result == "await ...check.aio(...)"


def test_rewrite_fallback_with_for_keyword():
    """Test fallback with 'for' keyword in regular method call."""
    code = "for item in obj.iterate():"
    success, result = rewrite_sync_to_async(code, "iterate")
    assert success is False
    # Should fall back due to 'for' keyword (handled separately as __aiter__)
    assert result == "await ...iterate.aio(...)"
