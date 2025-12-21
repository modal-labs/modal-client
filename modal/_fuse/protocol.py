# Copyright Modal Labs 2025
"""Protocol definitions for FUSE daemon <-> file server communication.

This module defines the message format used to communicate between the
FUSE daemon running in the container and the local file server.
"""

import base64
import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union


class FuseOp(str, Enum):
    """FUSE operations supported by the protocol."""

    GETATTR = "getattr"
    READDIR = "readdir"
    READ = "read"
    READLINK = "readlink"
    STATFS = "statfs"
    OPEN = "open"
    RELEASE = "release"
    # Handshake operations
    INIT = "init"
    PING = "ping"


@dataclass
class FuseRequest:
    """A request from the FUSE daemon to the file server."""

    op: FuseOp
    path: str
    # Operation-specific parameters
    offset: int = 0
    size: int = 0
    flags: int = 0
    fh: int = 0
    request_id: int = 0

    def to_json(self) -> str:
        return json.dumps(
            {
                "op": self.op.value,
                "path": self.path,
                "offset": self.offset,
                "size": self.size,
                "flags": self.flags,
                "fh": self.fh,
                "request_id": self.request_id,
            }
        )

    @classmethod
    def from_json(cls, data: str) -> "FuseRequest":
        d = json.loads(data)
        return cls(
            op=FuseOp(d["op"]),
            path=d["path"],
            offset=d.get("offset", 0),
            size=d.get("size", 0),
            flags=d.get("flags", 0),
            fh=d.get("fh", 0),
            request_id=d.get("request_id", 0),
        )


@dataclass
class FuseResponse:
    """A response from the file server to the FUSE daemon."""

    request_id: int
    error: Optional[int] = None  # errno value, None means success
    # Response data varies by operation
    data: Optional[bytes] = None  # For read operations
    attrs: Optional[dict[str, Any]] = None  # For getattr
    entries: Optional[list[str]] = None  # For readdir
    link_target: Optional[str] = None  # For readlink
    statfs: Optional[dict[str, int]] = None  # For statfs
    fh: int = 0  # File handle for open

    def to_json(self) -> str:
        result: dict[str, Any] = {"request_id": self.request_id}
        if self.error is not None:
            result["error"] = self.error
        if self.data is not None:
            result["data"] = base64.b64encode(self.data).decode("ascii")
        if self.attrs is not None:
            result["attrs"] = self.attrs
        if self.entries is not None:
            result["entries"] = self.entries
        if self.link_target is not None:
            result["link_target"] = self.link_target
        if self.statfs is not None:
            result["statfs"] = self.statfs
        if self.fh != 0:
            result["fh"] = self.fh
        return json.dumps(result)

    @classmethod
    def from_json(cls, data: str) -> "FuseResponse":
        d = json.loads(data)
        return cls(
            request_id=d["request_id"],
            error=d.get("error"),
            data=base64.b64decode(d["data"]) if d.get("data") else None,
            attrs=d.get("attrs"),
            entries=d.get("entries"),
            link_target=d.get("link_target"),
            statfs=d.get("statfs"),
            fh=d.get("fh", 0),
        )


def stat_to_dict(st: Union[os.stat_result, Path]) -> dict[str, Any]:
    """Convert stat result to a dictionary for JSON serialization."""
    if isinstance(st, Path):
        st = os.stat(st)

    return {
        "st_mode": st.st_mode,
        "st_ino": st.st_ino,
        "st_dev": st.st_dev,
        "st_nlink": st.st_nlink,
        "st_uid": st.st_uid,
        "st_gid": st.st_gid,
        "st_size": st.st_size,
        "st_atime": st.st_atime,
        "st_mtime": st.st_mtime,
        "st_ctime": st.st_ctime,
    }


def lstat_to_dict(path: Path) -> dict[str, Any]:
    """Like stat_to_dict but doesn't follow symlinks."""
    import os

    st = os.lstat(path)
    return {
        "st_mode": st.st_mode,
        "st_ino": st.st_ino,
        "st_dev": st.st_dev,
        "st_nlink": st.st_nlink,
        "st_uid": st.st_uid,
        "st_gid": st.st_gid,
        "st_size": st.st_size,
        "st_atime": st.st_atime,
        "st_mtime": st.st_mtime,
        "st_ctime": st.st_ctime,
    }


def error_response(request_id: int, err: int) -> FuseResponse:
    """Create an error response with the given errno."""
    return FuseResponse(request_id=request_id, error=err)


def success_response(request_id: int, **kwargs: Any) -> FuseResponse:
    """Create a success response with the given data."""
    return FuseResponse(request_id=request_id, **kwargs)
