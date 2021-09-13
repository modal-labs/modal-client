import logging
import os

config = {
    'loglevel': os.environ.get('LOGLEVEL', 'WARNING').upper(),
    'server.url': os.environ.get('POLYESTER_SERVER_URL', 'https://api.polyester.cloud'),
    'token.id': os.environ.get('POLYESTER_TOKEN_ID'),
    'token.secret': os.environ.get('POLYESTER_TOKEN_SECRET'),
    'task.id': os.environ.get('POLYESTER_TASK_ID'),  # Maybe redundant since it's also passed as a cmd arg
    'task.secret': os.environ.get('POLYESTER_TASK_SECRET'),
    'sync_entrypoint': os.environ.get('POLYESTER_SYNC_ENTRYPOINT'),
}

logging.basicConfig(
    level=config['loglevel'],
    format='%(threadName)s %(asctime)s %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S%z'
)
logger = logging.getLogger()
