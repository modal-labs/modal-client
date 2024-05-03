# Copyright Modal Labs 2022
import atexit
import logging
import sys
from typing import Any

from modal import App
from modal._utils.async_utils import run_coro_blocking
from modal.config import config, logger

app_ctx: Any


def load_ipython_extension(ipython):
    global app_ctx

    # Set logger for notebook sys.stdout
    logger.addHandler(logging.StreamHandler(stream=sys.stdout))
    logger.setLevel(config["loglevel"])

    # Create an app and provide it in the IPython app
    app = App()
    ipython.push({"app": app})

    app_ctx = app.run()

    # Notebooks have an event loop present, but we want this function
    # to be blocking. This is fairly hacky.
    run_coro_blocking(app_ctx.__aenter__())

    def exit_app():
        print("Exiting modal app")
        run_coro_blocking(app_ctx.__aexit__(None, None, None))

    atexit.register(exit_app)


def unload_ipython_extension(ipython):
    global app_ctx

    run_coro_blocking(app_ctx.__aexit__(None, None, None))
