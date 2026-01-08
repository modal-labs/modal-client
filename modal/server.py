# Copyright Modal Labs 2025
import inspect
import typing
from collections.abc import Collection
from typing import Any, Optional

from google.protobuf.message import Message

from modal_proto import api_pb2

from ._functions import _Function
from ._load_context import LoadContext
from ._object import _Object, live_method
from ._partial_function import (
    _find_callables_for_obj,
    _find_partial_methods_for_user_cls,
    _PartialFunction,
    _PartialFunctionFlags,
)
from ._resolver import Resolver
from ._traceback import print_server_warnings
from ._utils.async_utils import synchronize_api, synchronizer
from ._utils.deprecation import warn_if_passing_namespace
from .client import _Client
from .exception import InvalidError, NotFoundError

if typing.TYPE_CHECKING:
    import modal.app


class _Server(_Object):
    """Server runs an HTTP server started in an @enter method.

    See [lifecycle hooks](https://modal.com/docs/guide/lifecycle-functions) for more information.

    Generally, you will not construct a Server directly.
    Instead, use the [`@app.server()`](https://modal.com/docs/reference/modal.App#server) decorator.

    Example:

    ```python
        @app.server(port=8000, proxy_regions=["us-east"], min_containers=1)
        class SimpleServer:
            @modal.enter()
            def start(self):
                import subprocess
                self.proc = subprocess.Popen(["python3", "-m", "http.server", "8000"])

            @modal.exit()
            def stop(self):
                self.proc.terminate()
    ```
    """

    # Backend returns "cs-" IDs for servers (since it is defined on classes).
    # Set directly to avoid overwriting _Cls's registration in _Object._prefix_to_type.
    _type_prefix = "cs"

    _app: Optional["modal.app._App"] = None
    _name: Optional[str] = None
    _user_server: Optional[type] = None
    _user_server_function: Optional[_Function] = None
    # Runtime state for lifecycle management
    _user_server_instance: Optional[Any] = None
    _has_entered: bool = False

    def _initialize_from_empty(self):
        self._user_server = None
        self._user_server_function = None
        self._name = None
        self._app = None
        self._user_server_instance = None
        self._has_entered = False

    def _initialize_from_other(self, other: "_Server"):
        super()._initialize_from_other(other)
        self._app = other._app
        self._user_server = other._user_server
        self._user_server_function = other._user_server_function
        self._name = other._name
        self._load_context_overrides = other._load_context_overrides
        self._user_server_instance = other._user_server_instance
        self._has_entered = other._has_entered

    def _get_app(self) -> "modal.app._App":
        assert self._app is not None
        return self._app

    def _get_user_server(self) -> type:
        assert self._user_server is not None
        return self._user_server

    def _get_name(self) -> str:
        assert self._name is not None
        return self._name

    @property
    def __name__(self) -> str:
        """Return the name of the server class for compatibility with code expecting class-like objects."""
        return self._name or ""

    def _get_method_names(self) -> Collection[str]:
        """Servers don't have methods - return empty collection for compatibility."""
        return []

    def _get_service_function(self) -> _Function:
        assert self._user_server_function is not None
        return self._user_server_function

    # ============ Lifecycle Management ============

    def _get_or_create_instance(self) -> Any:
        """Get or construct the local server instance."""
        if self._user_server_instance is None:
            assert self._user_server is not None
            self._user_server_instance = object.__new__(self._user_server)
        return self._user_server_instance

    def _enter(self):
        """Run @enter lifecycle hooks (sync version)."""
        assert self._user_server is not None
        if self._has_entered:
            return

        user_server_instance = self._get_or_create_instance()

        # Support __enter__ context manager protocol
        enter_method = getattr(user_server_instance, "__enter__", None)
        if enter_method is not None:
            enter_method()

        # Run @modal.enter() decorated methods
        for method_flag in (
            _PartialFunctionFlags.ENTER_PRE_SNAPSHOT,
            _PartialFunctionFlags.ENTER_POST_SNAPSHOT,
        ):
            for enter_method in _find_callables_for_obj(user_server_instance, method_flag).values():
                enter_method()

        self._has_entered = True

    @synchronizer.nowrap
    async def _aenter(self):
        """Run @enter lifecycle hooks (async version)."""
        if self._has_entered:
            return

        user_server_instance = self._get_or_create_instance()

        aenter_method = getattr(user_server_instance, "__aenter__", None)
        enter_method = getattr(user_server_instance, "__enter__", None)
        if aenter_method is not None:
            await aenter_method()
        elif enter_method is not None:
            enter_method()

        # Run @modal.enter() decorated methods
        for method_flag in (
            _PartialFunctionFlags.ENTER_PRE_SNAPSHOT,
            _PartialFunctionFlags.ENTER_POST_SNAPSHOT,
        ):
            for enter_method in _find_callables_for_obj(user_server_instance, method_flag).values():
                enter_method()

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
        # if target_concurrency is not None:
        #     await self._get_service_function().update_autoscaler(target_concurrency=target_concurrency)
        return await self._get_service_function().update_autoscaler(
            min_containers=min_containers,
            max_containers=max_containers,
            scaledown_window=scaledown_window,
            buffer_containers=buffer_containers,
        )

    # ============ Hydration ============

    def _hydrate_metadata(self, metadata: Optional[Message]):
        # Servers don't have method metadata like Cls does
        if metadata is not None:
            assert isinstance(metadata, api_pb2.ClassHandleMetadata)
        # Just verify the service function is hydrated
        service_function = self._get_service_function()
        assert service_function.is_hydrated

    # ============ Construction ============

    @staticmethod
    def validate_wrapped_server_decorators(wrapped_server: type, enable_memory_snapshot: bool):
        if not inspect.isclass(wrapped_server):
            raise TypeError("The @app.server() decorator must be used on a class.")
        if not _find_partial_methods_for_user_cls(
            wrapped_server, _PartialFunctionFlags.ENTER_PRE_SNAPSHOT
        ) and not _find_partial_methods_for_user_cls(wrapped_server, _PartialFunctionFlags.ENTER_POST_SNAPSHOT):
            raise InvalidError("Server class must have an @modal.enter() to setup the server.")

        # Check for disallowed decorators
        # @modal.method() not allowed
        if _find_partial_methods_for_user_cls(wrapped_server, _PartialFunctionFlags.CALLABLE_INTERFACE).values():
            raise InvalidError(
                f"Server class {wrapped_server.__name__} cannot have @method() decorated functions. "
                "Servers only expose HTTP endpoints."
            )
        # @enter with snap=True without enable_memory_snapshot
        if (
            _find_partial_methods_for_user_cls(wrapped_server, _PartialFunctionFlags.ENTER_PRE_SNAPSHOT)
            and not enable_memory_snapshot
        ):
            raise InvalidError("Server must have `enable_memory_snapshot=True` to use `snap=True` on @enter methods.")

        if isinstance(wrapped_server, _PartialFunction):
            # @modal.concurrent not allowed on server classes
            if wrapped_server.flags & _PartialFunctionFlags.CONCURRENT:
                raise InvalidError(
                    f"Server class {wrapped_server.__name__} cannot have @concurrent() decorated functions. "
                    "Please use `target_concurrency` param instead."
                )
            # @modal.http_server not allowed on server classes
            if wrapped_server.flags & _PartialFunctionFlags.HTTP_WEB_INTERFACE:
                raise InvalidError(
                    f"Server class {wrapped_server.__name__} cannot have @modal.experimental.http_server() decorator. "
                    "Servers already expose HTTP endpoints."
                )

    @staticmethod
    def validate_construction_mechanism(user_server: type):
        """Validate that the server class doesn't have a custom constructor."""
        if user_server.__init__ != object.__init__:
            raise InvalidError(
                f"Server class {user_server.__name__} cannot have a custom __init__ method. "
                "Use @modal.enter() for initialization logic instead."
            )

    @staticmethod
    def from_local(
        user_server: type,
        app: "modal.app._App",
        service_function: _Function,
    ) -> "_Server":
        """Create a Server from a local class definition."""
        _Server.validate_wrapped_server_decorators(user_server, enable_memory_snapshot=False)
        # Validate - no custom constructors allowed
        _Server.validate_construction_mechanism(user_server)

        # Mark lifecycle methods as registered to avoid warnings
        lifecycle_flags = ~_PartialFunctionFlags.interface_flags()
        lifecycle_partials = _find_partial_methods_for_user_cls(user_server, lifecycle_flags)
        for partial_function in lifecycle_partials.values():
            partial_function.registered = True

        def _deps() -> list[_Function]:
            return [service_function]

        async def _load(
            self: "_Server",
            resolver: Resolver,
            load_context: LoadContext,
            existing_object_id: Optional[str],
        ):
            assert load_context.app_id is not None
            req = api_pb2.ClassCreateRequest(
                app_id=load_context.app_id,
                existing_class_id=existing_object_id or "",
                only_class_function=True,
            )
            resp = await load_context.client.stub.ClassCreate(req)
            self._hydrate(resp.class_id, load_context.client, resp.handle_metadata)

        rep = f"Server({user_server.__name__})"
        server: _Server = _Server._from_loader(
            _load,
            rep,
            deps=_deps,
            load_context_overrides=app._root_load_context,
        )
        server._app = app
        server._user_server = user_server
        server._user_server_function = service_function
        server._name = user_server.__name__
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

        ```python
        TODO(claudia): Add examples
        ```
        """
        warn_if_passing_namespace(namespace, "modal.Server.from_name")

        async def _load_remote(
            self: _Server,
            resolver: Resolver,
            load_context: LoadContext,
            existing_object_id: Optional[str],
        ):
            request = api_pb2.ClassGetRequest(
                app_name=app_name,
                object_tag=name,
                environment_name=load_context.environment_name,
                only_class_function=True,
            )
            try:
                response = await load_context.client.stub.ClassGet(request)
            except NotFoundError as exc:
                env_context = (
                    f" (in the '{load_context.environment_name}' environment)" if load_context.environment_name else ""
                )
                raise NotFoundError(
                    f"Lookup failed for Server '{name}' from the '{app_name}' app{env_context}: {exc}."
                ) from None

            print_server_warnings(response.server_warnings)
            assert self._user_server_function is not None
            await resolver.load(self._user_server_function, load_context)
            self._hydrate(response.class_id, load_context.client, response.handle_metadata)

        environment_rep = f", environment_name={environment_name!r}" if environment_name else ""
        rep = f"Server.from_name({app_name!r}, {name!r}{environment_rep})"

        load_context_overrides = LoadContext(client=client, environment_name=environment_name)
        server = _Server._from_loader(
            _load_remote,
            rep,
            is_another_app=True,
            hydrate_lazily=True,
            load_context_overrides=load_context_overrides,
        )

        # The service function uses a special naming convention
        service_function_name = f"{name}.*"
        server._user_server_function = _Function._from_name(
            app_name,
            service_function_name,
            load_context_overrides=load_context_overrides,
        )
        server._name = name
        return server

    def _is_local(self) -> bool:
        """Returns True if this Server has local source code available."""
        return self._user_server is not None


Server = synchronize_api(_Server)
