# Copyright Modal Labs 2022
import logging
import os

# TODO: set this from env.
logger = logging.getLogger("modal-utils")

ch = logging.StreamHandler()

log_level_numeric = logging.getLevelName(os.environ.get("MODAL_LOGLEVEL", "WARNING").upper())
logger.setLevel(log_level_numeric)
ch.setLevel(log_level_numeric)
ch.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z"))
logger.addHandler(ch)
