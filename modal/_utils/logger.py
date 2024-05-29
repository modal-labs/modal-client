# Copyright Modal Labs 2022
import logging
import os


def configure_logger(logger: logging.Logger, log_level: str, log_format: str):
    ch = logging.StreamHandler()
    log_level_numeric = logging.getLevelName(log_level.upper())
    logger.setLevel(log_level_numeric)
    ch.setLevel(log_level_numeric)

    if log_format.upper() == "JSON":
        # This is primarily for modal internal use.
        # pythonjsonlogger is already installed in the environment.
        from pythonjsonlogger import jsonlogger

        json_formatter = jsonlogger.JsonFormatter(
            fmt=(
                "%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] "
                "[dd.service=%(dd.service)s dd.env=%(dd.env)s dd.version=%(dd.version)s dd.trace_id=%(dd.trace_id)s "
                "dd.span_id=%(dd.span_id)s] "
                "- %(message)s"
            ),
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )

        ch.setFormatter(json_formatter)
    else:
        ch.setFormatter(logging.Formatter("[%(threadName)s] %(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z"))

    logger.addHandler(ch)


log_level = os.environ.get("MODAL_LOGLEVEL", "WARNING")
log_format = os.environ.get("MODAL_LOG_FORMAT", "STRING")

logger = logging.getLogger("modal-utils")
configure_logger(logger, log_level, log_format)
