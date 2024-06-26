# Copyright Modal Labs 2022
import asyncio
import inspect
import os
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, AsyncIterator, Callable, Dict, List, Literal, Optional, Type

from grpclib import GRPCError
from grpclib.exceptions import StreamTerminatedError
from synchronicity.exceptions import UserCodeException

from modal_proto import api_pb2

from .._serialization import deserialize, deserialize_data_format, serialize
from .._traceback import append_modal_tb
from ..config import config, logger
from ..exception import ExecutionError, FunctionTimeoutError, InvalidError, RemoteError
from ..mount import ROOT_DIR, _is_modal_path, _Mount
from .blob_utils import MAX_OBJECT_SIZE_BYTES, blob_download, blob_upload
from .grpc_utils import RETRYABLE_GRPC_STATUS_CODES, unary_stream


class FunctionInfoType(Enum):
    PACKAGE = "package"
    FILE = "file"
    SERIALIZED = "serialized"
    NOTEBOOK = "notebook"


class LocalFunctionError(InvalidError):
    """Raised if a function declared in a non-global scope is used in an impermissible way"""


def entrypoint_only_package_mount_condition(entrypoint_file):
    entrypoint_path = Path(entrypoint_file)

    def inner(filename):
        path = Path(filename)
        if path == entrypoint_path:
            return True
        if path.name == "__init__.py" and path.parent in entrypoint_path.parents:
            # ancestor __init__.py are included
            return True
        return False

    return inner


def is_global_object(object_qual_name):
    return "<locals>" not in object_qual_name.split(".")


def is_async(function):
    # TODO: this is somewhat hacky. We need to know whether the function is async or not in order to
    # coerce the input arguments to the right type. The proper way to do is to call the function and
    # see if you get a coroutine (or async generator) back. However at this point, it's too late to
    # coerce the type. For now let's make a determination based on inspecting the function definition.
    # This sometimes isn't correct, since a "vanilla" Python function can return a coroutine if it
    # wraps async code or similar. Let's revisit this shortly.
    if inspect.ismethod(function):
        function = function.__func__  # inspect the underlying function
    if inspect.iscoroutinefunction(function) or inspect.isasyncgenfunction(function):
        return True
    elif inspect.isfunction(function) or inspect.isgeneratorfunction(function):
        return False
    else:
        raise RuntimeError(f"Function {function} is a strange type {type(function)}")


class FunctionInfo:
    """Class that helps us extract a bunch of information about a function."""

    raw_f: Optional[Callable[..., Any]]  # if None - this is a "class service function"
    function_name: str
    cls: Optional[Type[Any]]
    definition_type: "api_pb2.Function.DefinitionType.ValueType"
    module_name: Optional[str]

    _type: FunctionInfoType
    _file: Optional[str]
    _base_dir: str
    _remote_dir: Optional[PurePosixPath] = None

    def is_service_class(self):
        if self.raw_f is None:
            assert self.cls
            return True
        return False

    # TODO: we should have a bunch of unit tests for this
    def __init__(
        self,
        f: Optional[Callable[..., Any]],
        serialized=False,
        name_override: Optional[str] = None,
        cls: Optional[Type] = None,
    ):
        self.raw_f = f
        self.cls = cls

        if name_override is not None:
            self.function_name = name_override
        elif f is None and cls:
            # "service function" for running all methods of a class
            self.function_name = f"{cls.__name__}.*"
        elif f.__qualname__ != f.__name__ and not serialized:
            # single method of a class - should be only @build-methods at this point
            if len(f.__qualname__.split(".")) > 2:
                raise InvalidError(
                    f"Cannot wrap `{f.__qualname__}`:"
                    " functions and classes used in Modal must be defined in global scope."
                    " If trying to apply additional decorators, they may need to use `functools.wraps`."
                )
            self.function_name = f"{cls.__name__}.{f.__name__}"
        else:
            self.function_name = f.__qualname__

        # If it's a cls, the @method could be defined in a base class in a different file.
        if cls is not None:
            module = inspect.getmodule(cls)
        else:
            module = inspect.getmodule(f)

        if getattr(module, "__package__", None) and not serialized:
            # This is a "real" module, eg. examples.logs.f
            # Get the package path
            # Note: __import__ always returns the top-level package.
            self._file = os.path.abspath(module.__file__)
            package_paths = set([os.path.abspath(p) for p in __import__(module.__package__).__path__])
            # There might be multiple package paths in some weird cases
            base_dirs = [
                base_dir for base_dir in package_paths if os.path.commonpath((base_dir, self._file)) == base_dir
            ]

            if not base_dirs:
                logger.info(f"Module files: {self._file}")
                logger.info(f"Package paths: {package_paths}")
                logger.info(f"Base dirs: {base_dirs}")
                raise Exception("Wasn't able to find the package directory!")
            elif len(base_dirs) > 1:
                # Base_dirs should all be prefixes of each other since they all contain `module_file`.
                base_dirs.sort(key=len)
            self._base_dir = base_dirs[0]
            self.module_name = module.__spec__.name
            self._remote_dir = ROOT_DIR / PurePosixPath(module.__package__.split(".")[0])
            self.definition_type = api_pb2.Function.DEFINITION_TYPE_FILE
            self._type = FunctionInfoType.PACKAGE
        elif hasattr(module, "__file__") and not serialized:
            # This generally covers the case where it's invoked with
            # python foo/bar/baz.py

            # If it's a cls, the @method could be defined in a base class in a different file.
            self._file = os.path.abspath(inspect.getfile(module))
            self.module_name = inspect.getmodulename(self._file)
            self._base_dir = os.path.dirname(self._file)
            self.definition_type = api_pb2.Function.DEFINITION_TYPE_FILE
            self._type = FunctionInfoType.FILE
        else:
            self.module_name = None
            self._base_dir = os.path.abspath("")  # get current dir
            self.definition_type = api_pb2.Function.DEFINITION_TYPE_SERIALIZED
            if serialized:
                self._type = FunctionInfoType.SERIALIZED
            else:
                self._type = FunctionInfoType.NOTEBOOK

        if self.definition_type == api_pb2.Function.DEFINITION_TYPE_FILE:
            # Sanity check that this function is defined in global scope
            # Unfortunately, there's no "clean" way to do this in Python
            qualname = f.__qualname__ if f else cls.__qualname__
            if not is_global_object(qualname):
                raise LocalFunctionError(
                    "Modal can only import functions defined in global scope unless they are `serialized=True`"
                )

    def is_serialized(self) -> bool:
        return self.definition_type == api_pb2.Function.DEFINITION_TYPE_SERIALIZED

    def serialized_function(self) -> bytes:
        # Note: this should only be called from .load() and not at function decoration time
        #       otherwise the serialized function won't have access to variables/side effect
        #        defined after it in the same file
        assert self.is_serialized()
        if self.raw_f:
            serialized_bytes = serialize(self.raw_f)
            logger.debug(f"Serializing {self.raw_f.__qualname__}, size is {len(serialized_bytes)}")
            return serialized_bytes
        else:
            logger.debug(f"Serializing function for class service function {self.cls.__qualname__} as empty")
            return b""

    def get_globals(self) -> Dict[str, Any]:
        from .._vendor.cloudpickle import _extract_code_globals

        func = self.raw_f
        f_globals_ref = _extract_code_globals(func.__code__)
        f_globals = {k: func.__globals__[k] for k in f_globals_ref if k in func.__globals__}
        return f_globals

    def get_entrypoint_mount(self) -> List[_Mount]:
        """
        Includes:
        * Implicit mount of the function itself (the module or package that the function is part of)

        Does not include:
        * Client mount
        * Explicit mounts added to the stub or function declaration
        * "Auto mounted" mounts, i.e. all mounts in sys.modules that are *not* installed in site-packages.
            These are typically local modules which are imported but not part of the running package

        """
        if self._type == FunctionInfoType.NOTEBOOK:
            # Don't auto-mount anything for notebooks.
            return []

        # make sure the function's own entrypoint is included:
        if self._type == FunctionInfoType.PACKAGE:
            if config.get("automount"):
                return [_Mount.from_local_python_packages(self.module_name)]
            elif self.definition_type == api_pb2.Function.DEFINITION_TYPE_FILE:
                # mount only relevant file and __init__.py:s
                return [
                    _Mount.from_local_dir(
                        self._base_dir,
                        remote_path=self._remote_dir,
                        recursive=True,
                        condition=entrypoint_only_package_mount_condition(self._file),
                    )
                ]
        elif self.definition_type == api_pb2.Function.DEFINITION_TYPE_FILE:
            remote_path = ROOT_DIR / Path(self._file).name
            if not _is_modal_path(remote_path):
                return [
                    _Mount.from_local_file(
                        self._file,
                        remote_path=remote_path,
                    )
                ]
        return []

    def get_tag(self):
        return self.function_name

    def is_nullary(self):
        signature = inspect.signature(self.raw_f)
        for param in signature.parameters.values():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                # variadic parameters are nullary
                continue
            if param.default is param.empty:
                return False
        return True


def method_has_params(f: Callable) -> bool:
    """Return True if a method (bound or unbound) has parameters other than self.

    Used for deprecation of @exit() parameters.
    """
    num_params = len(inspect.signature(f).parameters)
    if hasattr(f, "__self__"):
        return num_params > 0
    else:
        return num_params > 1


async def _stream_function_call_data(
    client, function_call_id: str, variant: Literal["data_in", "data_out"]
) -> AsyncIterator[Any]:
    """Read from the `data_in` or `data_out` stream of a function call."""
    last_index = 0
    retries_remaining = 10

    if variant == "data_in":
        stub_fn = client.stub.FunctionCallGetDataIn
    elif variant == "data_out":
        stub_fn = client.stub.FunctionCallGetDataOut
    else:
        raise ValueError(f"Invalid variant {variant}")

    while True:
        req = api_pb2.FunctionCallGetDataRequest(function_call_id=function_call_id, last_index=last_index)
        try:
            async for chunk in unary_stream(stub_fn, req):
                if chunk.index <= last_index:
                    continue
                last_index = chunk.index
                if chunk.data_blob_id:
                    message_bytes = await blob_download(chunk.data_blob_id, client.stub)
                else:
                    message_bytes = chunk.data
                message = deserialize_data_format(message_bytes, chunk.data_format, client)
                yield message
        except (GRPCError, StreamTerminatedError) as exc:
            if retries_remaining > 0:
                retries_remaining -= 1
                if isinstance(exc, GRPCError):
                    if exc.status in RETRYABLE_GRPC_STATUS_CODES:
                        await asyncio.sleep(1.0)
                        continue
                elif isinstance(exc, StreamTerminatedError):
                    continue
            raise


OUTPUTS_TIMEOUT = 55.0  # seconds
ATTEMPT_TIMEOUT_GRACE_PERIOD = 5  # seconds


def exc_with_hints(exc: BaseException):
    """mdmd:hidden"""
    if isinstance(exc, ImportError) and exc.msg == "attempted relative import with no known parent package":
        exc.msg += """\n
HINT: For relative imports to work, you might need to run your modal app as a module. Try:
- `python -m my_pkg.my_app` instead of `python my_pkg/my_app.py`
- `modal deploy my_pkg.my_app` instead of `modal deploy my_pkg/my_app.py`
"""
    elif isinstance(
        exc, RuntimeError
    ) and "CUDA error: no kernel image is available for execution on the device" in str(exc):
        msg = (
            exc.args[0]
            + """\n
HINT: This error usually indicates an outdated CUDA version. Older versions of torch (<=1.12)
come with CUDA 10.2 by default. If pinning to an older torch version, you can specify a CUDA version
manually, for example:
-  image.pip_install("torch==1.12.1+cu116", find_links="https://download.pytorch.org/whl/torch_stable.html")
"""
        )
        exc.args = (msg,)

    return exc


async def _process_result(result: api_pb2.GenericResult, data_format: int, stub, client=None):
    if result.WhichOneof("data_oneof") == "data_blob_id":
        data = await blob_download(result.data_blob_id, stub)
    else:
        data = result.data

    if result.status == api_pb2.GenericResult.GENERIC_STATUS_TIMEOUT:
        raise FunctionTimeoutError(result.exception)
    elif result.status != api_pb2.GenericResult.GENERIC_STATUS_SUCCESS:
        if data:
            try:
                exc = deserialize(data, client)
            except Exception as deser_exc:
                raise ExecutionError(
                    "Could not deserialize remote exception due to local error:\n"
                    + f"{deser_exc}\n"
                    + "This can happen if your local environment does not have the remote exception definitions.\n"
                    + "Here is the remote traceback:\n"
                    + f"{result.traceback}"
                )
            if not isinstance(exc, BaseException):
                raise ExecutionError(f"Got remote exception of incorrect type {type(exc)}")

            if result.serialized_tb:
                try:
                    tb_dict = deserialize(result.serialized_tb, client)
                    line_cache = deserialize(result.tb_line_cache, client)
                    append_modal_tb(exc, tb_dict, line_cache)
                except Exception:
                    pass
            uc_exc = UserCodeException(exc_with_hints(exc))
            raise uc_exc
        raise RemoteError(result.exception)

    try:
        return deserialize_data_format(data, data_format, client)
    except ModuleNotFoundError as deser_exc:
        raise ExecutionError(
            "Could not deserialize result due to error:\n"
            f"{deser_exc}\n"
            "This can happen if your local environment does not have a module that was used to construct the result. \n"
        )


async def _create_input(
    args, kwargs, client, *, idx: Optional[int] = None, method_name: Optional[str] = None
) -> api_pb2.FunctionPutInputsItem:
    """Serialize function arguments and create a FunctionInput protobuf,
    uploading to blob storage if needed.
    """
    if idx is None:
        idx = 0
    if method_name is None:
        method_name = ""  # proto compatible

    args_serialized = serialize((args, kwargs))

    if len(args_serialized) > MAX_OBJECT_SIZE_BYTES:
        args_blob_id = await blob_upload(args_serialized, client.stub)

        return api_pb2.FunctionPutInputsItem(
            input=api_pb2.FunctionInput(
                args_blob_id=args_blob_id,
                data_format=api_pb2.DATA_FORMAT_PICKLE,
                method_name=method_name,
            ),
            idx=idx,
        )
    else:
        return api_pb2.FunctionPutInputsItem(
            input=api_pb2.FunctionInput(
                args=args_serialized,
                data_format=api_pb2.DATA_FORMAT_PICKLE,
                method_name=method_name,
            ),
            idx=idx,
        )
