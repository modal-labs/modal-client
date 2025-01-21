# Copyright Modal Labs 2022
import sys

if sys.version_info[:2] < (3, 9):
    raise RuntimeError("This version of Modal requires at least Python 3.9")
if sys.version_info[:2] >= (3, 14):
    raise RuntimeError("This version of Modal does not support Python 3.14+")

from modal_version import __version__

try:
    from ._runtime.execution_context import current_function_call_id, current_input_id, interact, is_local
    from ._tunnel import Tunnel, forward
    from .app import App, Stub
    from .client import Client
    from .cloud_bucket_mount import CloudBucketMount
    from .cls import Cls, parameter
    from .dict import Dict
    from .exception import Error
    from .file_pattern_matcher import FilePatternMatcher
    from .functions import Function
    from .image import Image
    from .mount import Mount
    from .network_file_system import NetworkFileSystem
    from .output import enable_output
    from .partial_function import asgi_app, batched, build, enter, exit, method, web_endpoint, web_server, wsgi_app
    from .proxy import Proxy
    from .queue import Queue
    from .retries import Retries
    from .sandbox import Sandbox
    from .sandbox_snapshot import SandboxSnapshot
    from .schedule import Cron, Period
    from .scheduler_placement import SchedulerPlacement
    from .secret import Secret
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
    "App",
    "Client",
    "Cls",
    "Cron",
    "Dict",
    "Error",
    "FilePatternMatcher",
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
    "SandboxSnapshot",
    "SchedulerPlacement",
    "Secret",
    "Stub",
    "Tunnel",
    "Volume",
    "asgi_app",
    "batched",
    "build",
    "current_function_call_id",
    "current_input_id",
    "enable_output",
    "enter",
    "exit",
    "forward",
    "is_local",
    "interact",
    "method",
    "parameter",
    "web_endpoint",
    "web_server",
    "wsgi_app",
]
