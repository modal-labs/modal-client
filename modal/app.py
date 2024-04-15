# Copyright Modal Labs 2022
from dataclasses import dataclass, field
from typing import Dict, Optional

from google.protobuf.message import Message

from .app_utils import _list_apps, list_apps  # noqa: F401


@dataclass
class RunningApp:
    app_id: str
    environment_name: Optional[str] = None
    app_page_url: Optional[str] = None
    tag_to_object_id: Dict[str, str] = field(default_factory=dict)
    object_handle_metadata: Dict[str, Optional[Message]] = field(default_factory=dict)
    interactive: bool = False
