import atexit
import logging
import sys

from IPython.core.magic import register_cell_magic

from polyester import Session
from polyester.config import config, logger


def load_ipython_extension(ipython):
    global session_ctx

    # Set logger for notebook sys.stdout
    logger.addHandler(logging.StreamHandler(stream=sys.stdout))
    logger.setLevel(config["loglevel"])

    # Create a session and provide it in the IPython session
    session = Session()
    ipython.push({"session": session})

    session_ctx = session.run()
    session_ctx.__enter__()

    def exit_session(self):
        print("Exiting polyester session")
        session_ctx.__exit__(None, None, None)

    atexit.register(exit_session)


def unload_ipython_extension(ipython):
    global session_ctx

    session_ctx.__exit__(None, None, None)
