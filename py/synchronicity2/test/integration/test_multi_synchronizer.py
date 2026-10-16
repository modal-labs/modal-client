# Copyright Modal Labs 2026
"""Two named synchronizers use distinct worker threads and event loops."""

from pathlib import Path

from test.integration.test_utils import check_pyright, closing_synchronizers


def test_wrappers_from_one_codegen_invocation_share_synchronizer():
    import integration_synchronizer
    import simple_class
    import simple_function

    assert simple_function._synchronizer is integration_synchronizer.synchronizer
    assert simple_class._synchronizer is integration_synchronizer.synchronizer


def test_runtime():
    with closing_synchronizers("multi_sync._synchronizer_a", "multi_sync._synchronizer_b"):
        import multi_sync.a
        import multi_sync.b
        from multi_sync._synchronizer_a import synchronizer as sync_a
        from multi_sync._synchronizer_b import synchronizer as sync_b

        tid_a, lid_a = multi_sync.a.thread_and_loop_a()
        tid_b, lid_b = multi_sync.b.thread_and_loop_b()

        assert tid_a != tid_b
        assert lid_a != lid_b

        assert sync_a._thread is not None and sync_b._thread is not None
        assert sync_a._thread.ident == tid_a
        assert sync_b._thread.ident == tid_b
        assert sync_a._loop is not None and sync_b._loop is not None
        assert sync_a._loop is not sync_b._loop


def test_pyright_implementation():
    import multi_synchronizer_a_impl
    import multi_synchronizer_b_impl

    check_pyright([Path(multi_synchronizer_a_impl.__file__), Path(multi_synchronizer_b_impl.__file__)])


def test_pyright_wrapper():
    import multi_sync.a
    import multi_sync.b

    check_pyright([Path(multi_sync.a.__file__), Path(multi_sync.b.__file__)])


def test_pyright_usage():
    from importlib.util import find_spec

    spec = find_spec("multi_synchronizer_typecheck")
    assert spec and spec.origin
    check_pyright([Path(spec.origin)])
