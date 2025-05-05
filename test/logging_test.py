# Copyright Modal Labs 2025
import importlib
import json
import pytest
import re

import modal.config


@pytest.fixture
def clean_logger(modal_config):
    # this makes sure a new logger is set up for every test using this,
    # using a specific the modal.toml config
    # Since the logging module is based on a bunch of side effects, this
    # has to do some hacky module reloading and resetting of log handlers

    def f(conf):
        with modal_config(conf):
            modal.config.logger.handlers.clear()
            importlib.reload(modal.config)  # necessary since loggers are configured in global scope
        return modal.config.logger

    yield f
    modal.config.logger.handlers.clear()
    importlib.reload(modal.config)  # reset to normal log config


def test_log_level_configuration(clean_logger, capsys):
    conf = """[main]
active = true
loglevel = "INFO"
"""
    logger = clean_logger(conf)
    logger.info("dummy")
    log_output = capsys.readouterr().err
    assert re.match(r"^\[modal-client] [^ ]+ dummy$", log_output)


def test_log_format_json(clean_logger, capsys):
    conf = """[main]
active = true
log_format = "JSON"
"""
    logger = clean_logger(conf)
    logger.warning("dummy")
    json_line = capsys.readouterr().err
    log_struct = json.loads(json_line)
    assert log_struct["message"] == "dummy"
    assert log_struct["levelname"] == "WARNING"


def test_custom_log_pattern(clean_logger, capsys):
    conf = """[main]
active = true
log_pattern = "custom %(message)s"
"""
    logger = clean_logger(conf)
    logger.warning("dummy")
    assert capsys.readouterr().err == "custom dummy\n"
