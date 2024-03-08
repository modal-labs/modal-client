# Copyright Modal Labs 2022
import sys

if sys.version_info[:2] < (3, 8):
    raise RuntimeError("This version of Modal requires at least Python 3.8")
if sys.version_info[:2] >= (3, 13):
    raise RuntimeError("This version of Modal does not support Python 3.13+")

from modal_version import __version__

try:
    from ._tunnel import Tunnel, forward
    from .app import container_app, interact, is_local
    from .client import Client
    from .cloud_bucket_mount import CloudBucketMount
    from .cls import Cls
    from .dict import Dict
    from .exception import Error
    from .functions import Function, current_function_call_id, current_input_id
    from .image import Image
    from .mount import Mount, create_package_mounts
    from .network_file_system import NetworkFileSystem
    from .partial_function import asgi_app, build, enter, exit, method, web_endpoint, wsgi_app
    from .proxy import Proxy
    from .queue import Queue
    from .retries import Retries
    from .sandbox import Sandbox
    from .schedule import Cron, Period
    from .scheduler_placement import SchedulerPlacement
    from .secret import Secret
    from .shared_volume import SharedVolume
    from .stub import Stub
    from .volume import Volume
except Exception:
    print()
    print("#" * 80)
    print("#" + "Something with the Modal installation seems broken.".center(78) + "#")
    print("#" + "Please email support@modal.com and we will try to help!".center(78) + "#")
    print("#" * 80)
    print()
    raise

__all__ = [
    "__version__",
    "Client",
    "Cls",
    "Cron",
    "Dict",
    "Error",
    "Function",
    "Image",
    "Mount",
    "NetworkFileSystem",
    "Period",
    "Proxy",
    "Queue",
    "Retries",
    "CloudBucketMount",
    "Sandbox",
    "SchedulerPlacement",
    "Secret",
    "SharedVolume",
    "Stub",
    "Tunnel",
    "Volume",
    "asgi_app",
    "build",
    "container_app",
    "create_package_mounts",
    "current_function_call_id",
    "current_input_id",
    "enter",
    "exit",
    "forward",
    "is_local",
    "method",
    "web_endpoint",
    "wsgi_app",
    "interact",
]
