import logging
from typing import Dict

from .config import config, logger

TRACING_ENABLED = config.get("tracing_enabled")
if TRACING_ENABLED:
    from ddtrace import tracer
    from ddtrace.propagation.http import HTTPPropagator

    logging.getLogger("ddtrace").setLevel(logging.CRITICAL)
    tracer.configure(hostname="172.19.0.1")


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
        HTTPPropagator.inject(context, metadata)
    except Exception:
        logger.exception("Failed to inject tracing context")


def wrap(*args, **kwargs):
    if not TRACING_ENABLED:
        return lambda f: f

    return tracer.wrap(*args, **kwargs, service="modal-runtime-client")
