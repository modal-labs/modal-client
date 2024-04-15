# Copyright Modal Labs 2022
from dataclasses import dataclass, field
from typing import Dict, Optional

from google.protobuf.message import Message

from .app_utils import _list_apps, list_apps  # noqa: F401


@dataclass
class _LocalApp:
    app_id: str
    app_page_url: str
    tag_to_object_id: Dict[str, str] = field(default_factory=dict)
    environment_name: Optional[str] = None
    interactive: bool = False


@dataclass
class _ContainerApp:
    app_id: str
    environment_name: Optional[str] = None
    tag_to_object_id: Dict[str, str] = field(default_factory=dict)
    object_handle_metadata: Dict[str, Optional[Message]] = field(default_factory=dict)
