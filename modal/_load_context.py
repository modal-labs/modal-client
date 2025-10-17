# Copyright Modal Labs 2025
from typing import Optional

from .client import _Client
from .config import config


class LoadContext:
    """Encapsulates optional metadata values used during object loading.

    This metadata is set during object construction and propagated through
    parent-child relationships (e.g., App -> Function, Cls -> Obj -> bound methods).
    """

    _client: Optional[_Client] = None
    _environment_name: Optional[str] = None
    _app_id: Optional[str] = None

    def __init__(
        self,
        *,
        client: Optional[_Client] = None,
        environment_name: Optional[str] = None,
        app_id: Optional[str] = None,
        _apply_defaults: bool = True,
    ):
        self._client = client
        self._environment_name = environment_name
        self._app_id = app_id
        self._apply_defaults = _apply_defaults

    @property
    def client(self) -> _Client:
        assert self._client is not None
        return self._client

    @property
    def environment_name(self) -> str:
        assert self._environment_name is not None
        return self._environment_name

    @property
    def app_id(self) -> Optional[str]:
        return self._app_id

    _apply_defaults: bool = True

    @classmethod
    def empty(cls) -> "LoadContext":
        """Create an empty LoadContext with all fields set to None.

        Used when loading objects that don't have a parent context.
        """
        return cls(client=None, environment_name=None, app_id=None)

    @classmethod
    def no_defaults(cls) -> "LoadContext":
        """Create a LoadContext that will never apply defaults

        Use sparingly, only in places where we know the client and environment will not
        be used.
        """
        return cls(client=None, environment_name=None, app_id=None, _apply_defaults=False)

    def copy(self) -> "LoadContext":
        """Create a shallow copy of this metadata."""
        return LoadContext(
            client=self.client,
            environment_name=self.environment_name,
            app_id=self.app_id,
        )

    def merged_with(self, parent: "LoadContext") -> "LoadContext":
        """Create a new LoadContext with parent values filling in None fields.

        Returns a new LoadContext without mutating self or parent.
        Values from self take precedence over values from parent.
        """
        return LoadContext(
            client=self._client if self._client is not None else parent._client,
            environment_name=self._environment_name if self._environment_name is not None else parent._environment_name,
            app_id=self._app_id if self._app_id is not None else parent._app_id,
            _apply_defaults=self._apply_defaults,
        )

    async def apply_defaults(self) -> "LoadContext":
        """Infer default client and environment_name if not present

        Returns a new instance (no in place mutation)"""
        if not self._apply_defaults:
            return self

        return LoadContext(
            client=await _Client.from_env() if self._client is None else self.client,
            environment_name=config.get("environment") if self._environment_name is None else self._environment_name,
            app_id=self._app_id,
        )
