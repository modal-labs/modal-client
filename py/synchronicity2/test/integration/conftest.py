# Copyright Modal Labs 2026
"""Pytest configuration for integration tests.

Provides session-scoped fixtures that generate wrapper code once and reuse it across all tests.
"""

import asyncio
import gc
import pytest
import shutil
import subprocess
import sys
import warnings

from test.integration.test_utils import GENERATED_DIR, PROJECT_ROOT, SUPPORT_FILES, codegen_command, codegen_env

try:
    import pytest_asyncio.plugin as pytest_asyncio_plugin
except ImportError:  # pragma: no cover - only relevant when pytest-asyncio is installed
    pytest_asyncio_plugin = None


def _get_existing_event_loop_no_warn(policy=None) -> asyncio.AbstractEventLoop:
    """Return the current event loop without implicitly creating a new one."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            pass

        if policy is None:
            policy = asyncio.get_event_loop_policy()

        local = getattr(policy, "_local", None)
        loop = getattr(local, "_loop", None) if local is not None else None
        if loop is None:
            raise RuntimeError("There is no current event loop")
        return loop


if pytest_asyncio_plugin is not None:
    pytest_asyncio_plugin._get_event_loop_no_warn = _get_existing_event_loop_no_warn


def _iter_loaded_synchronizers():
    seen_ids: set[int] = set()
    for module_name, module in tuple(sys.modules.items()):
        if module is None:
            continue
        synchronizer = getattr(module, "synchronizer", None)
        if synchronizer is None or not hasattr(synchronizer, "_wrapper_classes"):
            continue
        synchronizer_id = id(synchronizer)
        if synchronizer_id in seen_ids:
            continue
        seen_ids.add(synchronizer_id)
        yield (module_name, synchronizer)


def _collect_active_synchronizers() -> list[tuple[str, object]]:
    active: list[tuple[str, object]] = []
    for name, synchronizer in _iter_loaded_synchronizers():
        thread = getattr(synchronizer, "_thread", None)
        loop = getattr(synchronizer, "_loop", None)
        thread_alive = thread is not None and thread.is_alive()
        loop_open = loop is not None and not loop.is_closed()
        if thread_alive or loop_open:
            active.append((name, synchronizer))
    return active


def _collect_unclosed_event_loops(ignored_loop_ids: set[int]) -> list[asyncio.BaseEventLoop]:
    loops: list[asyncio.BaseEventLoop] = []
    for obj in gc.get_objects():
        if not isinstance(obj, asyncio.BaseEventLoop):
            continue
        if obj.is_closed() or id(obj) in ignored_loop_ids:
            continue
        loops.append(obj)
    return loops


@pytest.fixture(autouse=True)
def close_generated_synchronizers():
    """Close generated synchronizers after each test if they were started."""
    yield

    for _module_name, synchronizer in _iter_loaded_synchronizers():
        close_loop = getattr(synchronizer, "_close_loop", None)
        if callable(close_loop):
            close_loop()


def _fail_if_runtime_resources_leaked() -> None:
    gc.collect()

    active_synchronizers = _collect_active_synchronizers()
    synchronizer_loop_ids = {
        id(loop)
        for _, synchronizer in active_synchronizers
        if (loop := getattr(synchronizer, "_loop", None)) is not None
    }
    stray_loops = _collect_unclosed_event_loops(synchronizer_loop_ids)

    if not active_synchronizers and not stray_loops:
        return

    details: list[str] = []
    if active_synchronizers:
        details.append("Active synchronizers:")
        for name, synchronizer in active_synchronizers:
            thread = getattr(synchronizer, "_thread", None)
            loop = getattr(synchronizer, "_loop", None)
            details.append(
                f"- {name}: thread_alive={thread.is_alive() if thread else False}, "
                f"loop_closed={loop.is_closed() if loop else None}"
            )
            close_loop = getattr(synchronizer, "_close_loop", None)
            if callable(close_loop):
                close_loop()

    if stray_loops:
        details.append("Unclosed event loops:")
        for loop in stray_loops:
            details.append(f"- {loop!r}")
            try:
                loop.close()
            except ValueError:
                pass

    asyncio.set_event_loop(None)
    pytest.fail("Test leaked event loops or synchronizer threads:\n" + "\n".join(details))


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_teardown(item, nextitem):
    yield
    _fail_if_runtime_resources_leaked()


@pytest.fixture(scope="session", autouse=True)
def generated_wrappers():
    """Generate all wrapper modules once using the CLI and make them available to all tests.

    This fixture:
    1. Vendors ``mylib.synchronicity2`` under ``generated/`` and copies ``mylib/_weather_impl.py`` there
    2. Uses the synchronicity2 CLI to generate wrapper code for all support modules
    3. Runs separate CLI passes for wrappers that use distinct synchronizer modules
    4. Adds ``generated/`` and ``support_files`` to ``sys.path``
    5. Keeps files after tests complete for manual inspection

    Returns:
        SimpleNamespace with attributes for each generated module
    """
    # Get paths
    project_root = PROJECT_ROOT
    support_files_path = SUPPORT_FILES
    generated_dir = GENERATED_DIR

    # Clean out the generated directory if it exists
    if generated_dir.exists():
        shutil.rmtree(generated_dir)

    # Create fresh generated directory
    generated_dir.mkdir(exist_ok=True)

    from synchronicity2.codegen.runtime_vendor import vendor_runtime

    vendor_runtime(target_package="mylib.synchronicity2", output_base=generated_dir)

    shutil.copy2(
        support_files_path / "mylib" / "_weather_impl.py",
        generated_dir / "mylib" / "_weather_impl.py",
    )
    shutil.copytree(
        support_files_path / "sandboxlib",
        generated_dir / "sandboxlib",
    )

    # Set up environment with support_files on PYTHONPATH

    env = codegen_env(str(support_files_path), str(project_root))

    def run_codegen(module_specs: list[str], synchronizer_module: str, *extra_args: str) -> None:
        module_args = [arg for module_name in module_specs for arg in ("-m", module_name)]
        result = subprocess.run(
            [
                *codegen_command(),
                "wrappers",
                "--synchronizer-module",
                synchronizer_module,
                *extra_args,
                *module_args,
                "-o",
                str(generated_dir),
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            print(f"CLI failed: {result.stderr}")
            print(f"CLI stdout: {result.stdout}")
            raise RuntimeError(f"Failed to generate wrapper code: {result.stderr}")

    module_specs = [
        "simple_function_impl",
        "simple_class_impl",
        "callback_translation_impl",
        "class_with_translation_impl",
        "class_with_inheritance_impl",
        "class_with_self_references_impl",
        "same_object_two_types_impl",
        "cross_module_wrapper_impl._base",
        "cross_module_wrapper_impl._sub",
        "generic_class_impl",
        "event_loop_check_impl",
        "nested_generators_impl",
        "two_way_generator_impl",
        "functions_with_typevars_impl",
        "overloads_impl",
        "union_translation_impl",
        "multifile_impl._a",
        "multifile_impl._b",
        "classmethod_staticmethod_impl",
        "custom_iterators_impl",
        "manual_nowrap_impl",
        "async_context_manager_impl",
        "property_class_impl",
        "dynamic_attr_impl",
        "renamed_exports_impl",
        "staticmethod_method_local_typevar_impl",
        "classproperty_class_impl",
        "sequence_callable_annotations_impl",
        "decorator_factory_impl",
        "descriptor_dunder_impl",
        "variadic_translation_impl",
        "include_underscored_methods_impl",
        "sandboxlib._sandbox",
    ]
    run_codegen(module_specs, "integration_synchronizer")
    run_codegen(
        ["synchronicity1_interop_impl"],
        "synchronicity1_interop_synchronizer",
        "--preload-module",
        "synchronicity1_interop_preload",
        "--synchronicity1-synchronizer",
        "synchronicity1_interop_runtime:synchronicity1_synchronizer",
    )
    run_codegen(["multi_synchronizer_a_impl"], "multi_sync._synchronizer_a")
    run_codegen(["multi_synchronizer_b_impl"], "multi_sync._synchronizer_b")

    # README-style mylib package (vendored runtime + mylib.weather wrappers)
    env_weather = codegen_env(str(generated_dir), str(support_files_path), str(project_root))
    weather_result = subprocess.run(
        [
            *codegen_command(),
            "wrappers",
            "-m",
            "mylib._weather_impl",
            "--synchronizer-module",
            "mylib._synchronizer",
            "--runtime-package",
            "mylib.synchronicity2",
            "-o",
            str(generated_dir),
        ],
        capture_output=True,
        text=True,
        env=env_weather,
        cwd=str(project_root),
    )
    if weather_result.returncode != 0:
        print(f"Weather CLI failed: {weather_result.stderr}")
        print(f"Weather CLI stdout: {weather_result.stdout}")
        raise RuntimeError(f"Failed to generate mylib.weather: {weather_result.stderr}")

    # Add paths to sys.path:
    # - support_files_path: so generated code can import impl modules (e.g., multifile_impl._a)
    # - generated_dir: so multifile.* can be imported directly
    sys.path.insert(0, str(support_files_path))
    sys.path.insert(0, str(generated_dir))

    active_synchronizers = _collect_active_synchronizers()
    stray_loops = _collect_unclosed_event_loops(set())
    if active_synchronizers or stray_loops:
        details: list[str] = []
        if active_synchronizers:
            details.append(
                "generated_wrappers unexpectedly activated synchronizers: "
                + ", ".join(name for name, _ in active_synchronizers)
            )
        if stray_loops:
            details.append(
                "generated_wrappers unexpectedly created event loops: " + ", ".join(repr(loop) for loop in stray_loops)
            )
        pytest.fail("\n".join(details))

    yield generated_dir
    # Note: We do NOT delete the generated/ directory - keep files for manual inspection


@pytest.fixture(scope="session")
def support_files():
    return SUPPORT_FILES
