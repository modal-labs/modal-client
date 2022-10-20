# Copyright Modal Labs 2022
import contextlib
import logging
from typing import Dict

from .config import config, logger

if config.get("tracing_enabled"):
    try:
        from ddtrace import tracer
        from ddtrace.propagation.http import HTTPPropagator

        TRACING_ENABLED = True
    except ImportError:
        TRACING_ENABLED = False
else:
    TRACING_ENABLED = False

if TRACING_ENABLED:
    logging.getLogger("ddtrace").setLevel(logging.CRITICAL)
    tracer.configure(hostname="172.19.0.1")

if config.get("profiling_enabled"):
    try:
        import ddtrace.profiling.auto  # noqa: F401
    except ImportError:
        pass


def extract_tracing_context(headers: Dict[str, str]):
    if not TRACING_ENABLED:
        return

    try:
        tracing_context = HTTPPropagator.extract(headers)
        tracer.context_provider.activate(tracing_context)
    except Exception:
        logger.exception("Failed to extract tracing context")


def inject_tracing_context(metadata: Dict[str, str]):
    if not TRACING_ENABLED:
        return

    try:
        context = tracer.current_trace_context()
        if context:
            HTTPPropagator.inject(context, metadata)
    except Exception:
        logger.exception("Failed to inject tracing context")


def wrap(*args, **kwargs):
    if not TRACING_ENABLED:
        return lambda f: f

    return tracer.wrap(*args, **kwargs, service="modal-runtime-client")


def trace(*args, **kwargs):
    if not TRACING_ENABLED:
        return contextlib.nullcontext()

    return tracer.trace(*args, **kwargs, service="modal-runtime-client")


def set_span_tag(key, value):
    if not TRACING_ENABLED:
        return

    span = tracer.current_span()
    if span:
        span.set_tag(key, value)
