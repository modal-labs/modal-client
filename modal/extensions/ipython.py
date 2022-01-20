import atexit
import logging
import sys

from IPython.core.magic import register_cell_magic

from modal import Session
from modal._async_utils import run_coro_blocking
from modal._session_singleton import set_default_session
from modal.config import config, logger


def load_ipython_extension(ipython):
    global session_ctx

    # Set logger for notebook sys.stdout
    logger.addHandler(logging.StreamHandler(stream=sys.stdout))
    logger.setLevel(config["loglevel"])

    # Create a session and provide it in the IPython session
    session = Session(blocking_late_creation_ok=True)
    ipython.push({"session": session})
    set_default_session(session)

    session_ctx = session.run()

    # Notebooks have an event loop present, but we want this function
    # to be blocking. This is fairly hacky.
    run_coro_blocking(session_ctx.__aenter__())

    def exit_session(self):
        print("Exiting modal session")
        run_coro_blocking(session_ctx.__aexit__(None, None, None))

    atexit.register(exit_session)


def unload_ipython_extension(ipython):
    global session_ctx

    run_coro_blocking(session_ctx.__aexit__(None, None, None))
