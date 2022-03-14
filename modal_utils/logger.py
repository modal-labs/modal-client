import logging

# TODO: set this from env.
logging.basicConfig(level="WARNING", format="%(threadName)s %(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z")
logger = logging.getLogger()
