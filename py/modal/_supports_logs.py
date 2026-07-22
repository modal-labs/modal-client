# Copyright Modal Labs 2026

import dataclasses
from typing import Protocol

from modal._logs import LogsFilters
from modal.client import _Client


@dataclasses.dataclass(frozen=True)
class _LogQueryData:
    """Resolved data needed to query logs for a Modal object.

    Log managers lazily ask their source object for this data before the first
    query.
    """

    client: _Client
    app_id: str
    filters: LogsFilters


class _SupportsLogs(Protocol):
    """Protocol for objects that support log streaming and fetching.
    Need to define this in this file because otherwise we will get a mypy
    error caused by how synchronicity wraps the _logs_manager.LogsManager class.
    """

    async def _get_log_query_data(self) -> _LogQueryData: ...

    @property
    def object_id(self) -> str: ...
