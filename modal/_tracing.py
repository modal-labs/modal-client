# Copyright Modal Labs 2022
import contextlib
import logging
import os
from typing import Dict

from .config import config, logger

if config.get("tracing_enabled") or os.environ.get("DD_TRACE_ENABLED") == "true":
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
    if any(os.environ.get(k) for k in ("DD_TRACE_AGENT_URL", "DD_AGENT_HOST")):
        tracer.configure()
    else:
        tracer.configure(hostname="127.0.0.1")

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


TRACING_KWARGS = {} if os.environ.get("DD_SERVICE") else {"service": "modal-runtime-client"}


def wrap(*args, **kwargs):
    if not TRACING_ENABLED:
        return lambda f: f

    return tracer.wrap(*args, **kwargs, **TRACING_KWARGS)


def trace(*args, **kwargs):
    if not TRACING_ENABLED:
        return contextlib.nullcontext()

    return tracer.trace(*args, **kwargs, **TRACING_KWARGS)


def set_span_tag(key, value):
    if not TRACING_ENABLED:
        return

    span = tracer.current_span()
    if span:
        span.set_tag(key, value)
