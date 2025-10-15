# Copyright Modal Labs 2025
from dataclasses import dataclass
from typing import Optional

from .client import _Client
from .config import config


@dataclass
class LoadMetadata:
    """Encapsulates optional metadata values used during object loading.

    This metadata is set during object construction and propagated through
    parent-child relationships (e.g., App -> Function, Cls -> Obj -> bound methods).
    """

    client: Optional[_Client] = None
    environment_name: Optional[str] = None
    app_id: Optional[str] = None

    _apply_defaults: bool = True

    @classmethod
    def empty(cls) -> "LoadMetadata":
        """Create an empty LoadMetadata with all fields set to None.

        Used when loading objects that don't have a parent context.
        """
        return cls(client=None, environment_name=None, app_id=None)

    @classmethod
    def no_defaults(cls) -> "LoadMetadata":
        """Create a LoadMetadata that will never apply defaults

        Use sparingly, only in places where we know the client and environment will not
        be used.
        """
        return cls(client=None, environment_name=None, app_id=None, _apply_defaults=False)

    def merge_from(self, other: "LoadMetadata") -> None:
        """Merge non-None values from other into this metadata."""
        if other.client is not None:
            self.client = other.client
        if other.environment_name is not None:
            self.environment_name = other.environment_name
        if other.app_id is not None:
            self.app_id = other.app_id

    def copy(self) -> "LoadMetadata":
        """Create a shallow copy of this metadata."""
        return LoadMetadata(
            client=self.client,
            environment_name=self.environment_name,
            app_id=self.app_id,
        )

    def merged_with(self, parent: "LoadMetadata") -> "LoadMetadata":
        """Create a new LoadMetadata with parent values filling in None fields.

        Returns a new LoadMetadata without mutating self or parent.
        Values from self take precedence over values from parent.
        """
        return LoadMetadata(
            client=self.client if self.client is not None else parent.client,
            environment_name=self.environment_name if self.environment_name is not None else parent.environment_name,
            app_id=self.app_id if self.app_id is not None else parent.app_id,
            _apply_defaults=self._apply_defaults,
        )

    async def apply_defaults(self) -> "LoadMetadata":
        """Infer default client and environment_name if not present

        Returns a new instance (no in place mutation)"""
        if not self._apply_defaults:
            return self

        return LoadMetadata(
            client=await _Client.from_env() if self.client is None else self.client,
            environment_name=config.get("environment") if self.environment_name is None else self.environment_name,
            app_id=self.app_id,
        )
