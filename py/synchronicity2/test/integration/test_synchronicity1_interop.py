# Copyright Modal Labs 2026
import importlib
import sys
from pathlib import Path

from test.integration.test_utils import check_pyright


def test_synchronicity1_wrapped_type_runtime(generated_wrappers):
    from synchronicity1_interop_runtime import synchronicity1_synchronizer

    sys.modules.pop("synchronicity1_legacy", None)
    sys.modules.pop("synchronicity1_interop", None)
    assert "synchronicity1_legacy" not in sys.modules

    try:
        module = importlib.import_module("synchronicity1_interop")
        synchronizer_module = importlib.import_module("synchronicity1_interop_synchronizer")
        from synchronicity1_legacy import LegacyValue

        value = LegacyValue(42)
        result = module.echo_legacy(value)
        results = module.echo_legacy_list([value])

        assert "synchronicity1_legacy" in sys.modules
        assert isinstance(result, LegacyValue)
        assert result is value
        assert result.value == 42
        assert results == [value]
        assert synchronizer_module.synchronizer._synchronicity1 is synchronicity1_synchronizer
    finally:
        synchronicity1_synchronizer._close_loop()
        sys.modules.pop("synchronicity1_interop", None)
        sys.modules.pop("synchronicity1_interop_synchronizer", None)


def test_synchronicity1_wrapped_type_codegen(generated_wrappers):
    source = (generated_wrappers / "synchronicity1_interop.py").read_text()
    synchronizer_source = (generated_wrappers / "synchronicity1_interop_synchronizer.py").read_text()

    assert "import synchronicity1_legacy\n" in source
    assert "import synchronicity1_legacy_impl\n" in source
    assert "import synchronicity1_interop_synchronizer\n" in source
    assert "_synchronizer = synchronicity1_interop_synchronizer.synchronizer" in source
    assert "_synchronicity1._translate_in(value)" in source
    assert "_synchronicity1._translate_out(result)" in source
    assert "_unwrap_wrapper" not in source
    assert "import synchronicity1_interop_runtime\n" in synchronizer_source
    assert "from synchronicity2.synchronizer import Synchronizer" in synchronizer_source
    assert (
        "synchronizer = Synchronizer("
        "synchronicity1_synchronizer=synchronicity1_interop_runtime.synchronicity1_synchronizer)" in synchronizer_source
    )


def test_synchronicity1_wrapped_type_pyright(generated_wrappers):
    check_pyright(
        [
            generated_wrappers / "synchronicity1_interop.py",
            Path(__file__).parent.parent / "support_files" / "synchronicity1_interop_typecheck.py",
        ]
    )
