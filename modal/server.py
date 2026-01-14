# Copyright Modal Labs 2025
import inspect
import typing
from typing import Any, Optional

from google.protobuf.message import Message

from ._functions import _Function
from ._load_context import LoadContext
from ._object import live_method
from ._partial_function import (
    _find_callables_for_obj,
    _find_partial_methods_for_user_cls,
    _PartialFunction,
    _PartialFunctionFlags,
)
from ._utils.async_utils import synchronize_api, synchronizer
from ._utils.deprecation import warn_if_passing_namespace
from .client import _Client
from .cls import is_parameter
from .exception import InvalidError

if typing.TYPE_CHECKING:
    import modal.app


class _Server:
    """Server runs an HTTP server started in an @enter method.

    See [lifecycle hooks](https://modal.com/docs/guide/lifecycle-functions) for more information.

    Generally, you will not construct a Server directly.
    Instead, use the [`@app.server()`](https://modal.com/docs/reference/modal.App#server) decorator.

    TODO(claudia): Add examples
    """

    # Maps 1-1 w function
    _type_prefix = "fu"

    _app: Optional["modal.app._App"] = None
    _name: Optional[str] = None
    # Raw user defined class
    _user_cls: Optional[type] = None
    # Instantiated raw user class
    _user_cls_instance: Optional[Any] = None
    # Function interface with server backend
    _service_function: Optional[_Function] = None
    _has_entered: bool = False

    def __init__(self):
        self._initialize_from_empty()

    def _initialize_from_empty(self):
        self._app = None
        self._name = None
        self._user_cls = None
        self._user_cls_instance = None
        self._service_function = None
        self._has_entered = False

    def _initialize_from_other(self, other: "_Server"):
        self._app = other._app
        self._name = other._name
        self._user_cls = other._user_cls
        self._user_cls_instance = other._user_cls_instance
        self._service_function = other._service_function
        self._has_entered = other._has_entered

    def _get_user_cls(self) -> type:
        assert self._user_cls is not None
        return self._user_cls

    def _get_name(self) -> str:
        assert self._name is not None
        return self._name

    def _get_app(self) -> "modal.app._App":
        assert self._app is not None
        return self._app

    @property
    def __name__(self) -> str:
        """Return the name of the server class for compatibility with code expecting class-like objects."""
        return self._name or ""

    def _get_service_function(self) -> _Function:
        assert self._service_function is not None
        return self._service_function

    @staticmethod
    def _extract_user_cls(wrapped_user_cls: "type | _PartialFunction") -> type:
        if isinstance(wrapped_user_cls, _PartialFunction):
            return wrapped_user_cls.user_cls
        else:
            return wrapped_user_cls

    # ============ Lifecycle Management ============

    def _get_or_create_user_cls_instance(self) -> Any:
        """Get or construct the local server instance."""
        if self._user_cls_instance is None:
            assert self._user_cls is not None
            self._user_cls_instance = object.__new__(self._user_cls)
        return self._user_cls_instance

    def _enter(self):
        """Run @enter lifecycle hooks (sync version)."""
        assert self._user_cls is not None
        if self._has_entered:
            return

        user_cls_instance = self._get_or_create_user_cls_instance()

        # Support __enter__ context manager protocol
        enter_method = getattr(user_cls_instance, "__enter__", None)
        if enter_method is not None:
            enter_method()

        # Run @modal.enter() decorated methods
        for method_flag in (
            _PartialFunctionFlags.ENTER_PRE_SNAPSHOT,
            _PartialFunctionFlags.ENTER_POST_SNAPSHOT,
        ):
            for enter_method in _find_callables_for_obj(user_cls_instance, method_flag).values():
                enter_method()

        self._has_entered = True

    @synchronizer.nowrap
    async def _aenter(self):
        """Run @enter lifecycle hooks (async version)."""
        assert self._user_cls is not None
        if self._has_entered:
            return

        user_cls_instance = self._get_or_create_user_cls_instance()

        aenter_method = getattr(user_cls_instance, "__aenter__", None)
        enter_method = getattr(user_cls_instance, "__enter__", None)
        if aenter_method is not None:
            await aenter_method()
        elif enter_method is not None:
            enter_method()

        # Run @modal.enter() decorated methods
        for method_flag in (
            _PartialFunctionFlags.ENTER_PRE_SNAPSHOT,
            _PartialFunctionFlags.ENTER_POST_SNAPSHOT,
        ):
            for enter_method in _find_callables_for_obj(user_cls_instance, method_flag).values():
                res = enter_method()
                if inspect.iscoroutine(res):
                    await res

        self._has_entered = True

    @property
    def _entered(self) -> bool:
        return self._has_entered

    @_entered.setter
    def _entered(self, val: bool):
        self._has_entered = val

    # ============ Live Methods ============

    @live_method
    async def get_urls(self) -> Optional[list[str]]:
        """Get the URL(s) of this server."""
        return await self._get_service_function()._experimental_get_flash_urls()

    @live_method
    async def update_autoscaler(
        self,
        *,
        min_containers: Optional[int] = None,
        max_containers: Optional[int] = None,
        scaledown_window: Optional[int] = None,
        buffer_containers: Optional[int] = None,
    ) -> None:
        """Override the current autoscaler behavior for this Server.

        Unspecified parameters will retain their current value.

        Examples:
        TODO(claudia): Add examples

        """
        return await self._get_service_function().update_autoscaler(
            min_containers=min_containers,
            max_containers=max_containers,
            scaledown_window=scaledown_window,
            buffer_containers=buffer_containers,
        )

    # ============ Hydration ============

    def _hydrate_metadata(self, metadata: Optional[Message]):
        service_function = self._get_service_function()
        assert service_function.is_hydrated

    async def hydrate(self, client: Optional[_Client] = None) -> "_Server":
        """Hydrate the server by hydrating its underlying service function."""
        # This is required since we want to support @livemethod() decorated methods
        # and is normally handled by the _Object.hydrate() method
        # but we only want to hydrate the service function.
        service_function = self._get_service_function()
        await service_function.hydrate(client)
        return self

    # ============ Construction ============

    @staticmethod
    def _validate_wrapped_user_cls_decorators(
        wrapped_user_cls: "type | _PartialFunction", enable_memory_snapshot: bool
    ):
        user_cls = _Server._extract_user_cls(wrapped_user_cls)

        if not inspect.isclass(user_cls):
            raise TypeError("The @app.server() decorator must be used on a class.")

        # Check for modal.parameter() - not allowed on server classes
        params = {k: v for k, v in user_cls.__dict__.items() if is_parameter(v)}
        if params:
            raise InvalidError(
                f"Server class {user_cls.__name__} cannot use modal.parameter(). "
                "Servers do not support parameterization."
            )

        if not _find_partial_methods_for_user_cls(
            user_cls, _PartialFunctionFlags.ENTER_PRE_SNAPSHOT
        ) and not _find_partial_methods_for_user_cls(user_cls, _PartialFunctionFlags.ENTER_POST_SNAPSHOT):
            raise InvalidError("Server class must have an @modal.enter() to setup the server.")

        # Check for disallowed decorators
        # @modal.method() not allowed
        if _find_partial_methods_for_user_cls(user_cls, _PartialFunctionFlags.CALLABLE_INTERFACE).values():
            raise InvalidError(
                f"Server class {user_cls.__name__} cannot have @method() decorated functions. "
                "Servers only expose HTTP endpoints."
            )
        # @enter with snap=True without enable_memory_snapshot
        if (
            _find_partial_methods_for_user_cls(user_cls, _PartialFunctionFlags.ENTER_PRE_SNAPSHOT)
            and not enable_memory_snapshot
        ):
            raise InvalidError("Server must have `enable_memory_snapshot=True` to use `snap=True` on @enter methods.")

        if isinstance(wrapped_user_cls, _PartialFunction):
            # @modal.concurrent not allowed on server classes
            if wrapped_user_cls.flags & _PartialFunctionFlags.CONCURRENT:
                raise InvalidError(
                    f"Server class {user_cls.__name__} cannot have @concurrent() decorated functions. "
                    "Please use `target_concurrency` param instead."
                )
            # @modal.http_server not allowed on server classes
            if wrapped_user_cls.flags & _PartialFunctionFlags.HTTP_WEB_INTERFACE:
                raise InvalidError(
                    f"Server class {user_cls.__name__} cannot have @modal.http_server() decorator. "
                    "Servers already expose HTTP endpoints."
                )

    @staticmethod
    def validate_construction_mechanism(wrapped_user_cls: "type | _PartialFunction"):
        """Validate that the server class doesn't have a custom constructor."""
        # Extract the underlying class if wrapped in a _PartialFunction (e.g., from @modal.clustered())
        user_cls = _Server._extract_user_cls(wrapped_user_cls)

        if user_cls.__init__ != object.__init__:
            raise InvalidError(
                f"Server class {user_cls.__name__} cannot have a custom __init__ method. "
                "Use @modal.enter() for initialization logic instead."
            )

    @staticmethod
    def from_local(
        wrapped_user_cls: "type | _PartialFunction",
        app: "modal.app._App",
        service_function: _Function,
    ) -> "_Server":
        """Create a Server from a local class definition.

        Note: Validation should be done by the caller (app.server()) BEFORE creating
        the service function, so we don't repeat it here.
        """
        # Extract the underlying class if wrapped in a _PartialFunction (e.g., from @modal.clustered())
        user_cls = _Server._extract_user_cls(wrapped_user_cls)

        # Mark lifecycle methods as registered to avoid warnings
        lifecycle_flags = ~_PartialFunctionFlags.interface_flags()
        lifecycle_partials = _find_partial_methods_for_user_cls(user_cls, lifecycle_flags)
        for partial_function in lifecycle_partials.values():
            partial_function.registered = True

        server = _Server()
        server._app = app
        server._user_cls = user_cls
        server._service_function = service_function
        server._name = user_cls.__name__
        return server

    @classmethod
    def from_name(
        cls: type["_Server"],
        app_name: str,
        name: str,
        *,
        namespace: Any = None,  # Deprecated, hidden
        environment_name: Optional[str] = None,
        client: Optional[_Client] = None,
    ) -> "_Server":
        """Reference a Server from a deployed App by its name.

        This is a lazy method that defers hydrating the local
        object with metadata from Modal servers until the first
        time it is actually used.

        TODO(claudia): Add examples
        """
        warn_if_passing_namespace(namespace, "modal.Server.from_name")

        load_context_overrides = LoadContext(client=client, environment_name=environment_name)

        # The service function uses a special naming convention
        service_function_name = f"{name}.*"

        server = _Server()
        server._service_function = _Function._from_name(
            app_name,
            service_function_name,
            load_context_overrides=load_context_overrides,
        )
        server._name = name
        return server

    def _is_local(self) -> bool:
        """Returns True if this Server has local source code available."""
        return self._user_cls is not None


Server = synchronize_api(_Server)
