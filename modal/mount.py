# Copyright Modal Labs 2022
import abc
import asyncio
import concurrent.futures
import dataclasses
import os
import site
import sys
import sysconfig
import time
import typing
import warnings
from collections.abc import AsyncGenerator
from pathlib import Path, PurePosixPath
from typing import Callable, Optional, Sequence, Union

from google.protobuf.message import Message

import modal.exception
import modal.file_pattern_matcher
from modal_proto import api_pb2
from modal_version import __version__

from ._object import _get_environment_name, _Object
from ._resolver import Resolver
from ._utils.async_utils import aclosing, async_map, synchronize_api
from ._utils.blob_utils import FileUploadSpec, blob_upload_file, get_file_upload_spec_from_path
from ._utils.deprecation import deprecation_warning, renamed_parameter
from ._utils.grpc_utils import retry_transient_errors
from ._utils.name_utils import check_object_name
from ._utils.package_utils import get_module_mount_info
from .client import _Client
from .config import config, logger
from .exception import InvalidError, ModuleNotMountable
from .file_pattern_matcher import FilePatternMatcher

ROOT_DIR: PurePosixPath = PurePosixPath("/root")
MOUNT_PUT_FILE_CLIENT_TIMEOUT = 10 * 60  # 10 min max for transferring files

# Supported releases and versions for python-build-standalone.
#
# These can be updated safely, but changes will trigger a rebuild for all images
# that rely on `add_python()` in their constructor.
PYTHON_STANDALONE_VERSIONS: dict[str, tuple[str, str]] = {
    "3.9": ("20230826", "3.9.18"),
    "3.10": ("20230826", "3.10.13"),
    "3.11": ("20230826", "3.11.5"),
    "3.12": ("20240107", "3.12.1"),
    "3.13": ("20241008", "3.13.0"),
}

MOUNT_DEPRECATION_MESSAGE_PATTERN = """modal.Mount usage will soon be deprecated.

Use {replacement} instead, which is functionally and performance-wise equivalent.

See https://modal.com/docs/guide/modal-1-0-migration for more details.
"""


def client_mount_name() -> str:
    """Get the deployed name of the client package mount."""
    return f"modal-client-mount-{__version__}"


def python_standalone_mount_name(version: str) -> str:
    """Get the deployed name of the python-build-standalone mount."""
    if "-" in version:  # default to glibc
        version, libc = version.split("-")
    else:
        libc = "gnu"
    if version not in PYTHON_STANDALONE_VERSIONS:
        raise modal.exception.InvalidError(
            f"Unsupported standalone python version: {version!r}, supported values are "
            f"{list(PYTHON_STANDALONE_VERSIONS)}"
        )
    if libc != "gnu":
        raise modal.exception.InvalidError(f"Unsupported libc identifier: {libc}")
    release, full_version = PYTHON_STANDALONE_VERSIONS[version]
    return f"python-build-standalone.{release}.{full_version}-{libc}"


class _MountEntry(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def description(self) -> str: ...

    @abc.abstractmethod
    def get_files_to_upload(self) -> typing.Iterator[tuple[Path, str]]: ...

    @abc.abstractmethod
    def watch_entry(self) -> tuple[Path, Path]: ...

    @abc.abstractmethod
    def top_level_paths(self) -> list[tuple[Path, PurePosixPath]]: ...


def _select_files(entries: list[_MountEntry]) -> list[tuple[Path, PurePosixPath]]:
    # TODO: make this async
    all_files: set[tuple[Path, PurePosixPath]] = set()
    for entry in entries:
        all_files |= set(entry.get_files_to_upload())
    return list(all_files)


@dataclasses.dataclass
class _MountFile(_MountEntry):
    local_file: Path
    remote_path: PurePosixPath

    def description(self) -> str:
        return str(self.local_file)

    def get_files_to_upload(self):
        local_file = self.local_file.resolve()
        if not local_file.exists():
            raise FileNotFoundError(local_file)

        rel_filename = self.remote_path
        yield local_file, rel_filename

    def watch_entry(self):
        safe_path = self.local_file.expanduser().absolute()
        return safe_path.parent, safe_path

    def top_level_paths(self) -> list[tuple[Path, PurePosixPath]]:
        return [(self.local_file, self.remote_path)]


@dataclasses.dataclass
class _MountDir(_MountEntry):
    local_dir: Path
    remote_path: PurePosixPath
    ignore: Callable[[Path], bool]
    recursive: bool

    def description(self):
        return str(self.local_dir.expanduser().absolute())

    def get_files_to_upload(self):
        # we can't use .resolve() eagerly here since that could end up "renaming" symlinked files
        # see test_mount_directory_with_symlinked_file
        local_dir = self.local_dir.expanduser().absolute()

        if not local_dir.exists():
            raise FileNotFoundError(local_dir)

        if not local_dir.is_dir():
            raise NotADirectoryError(local_dir)

        if self.recursive:
            gen = (os.path.join(root, name) for root, dirs, files in os.walk(local_dir) for name in files)
        else:
            gen = (dir_entry.path for dir_entry in os.scandir(local_dir) if dir_entry.is_file())

        for local_filename in gen:
            local_path = Path(local_filename)
            if not self.ignore(local_path):
                local_relpath = local_path.expanduser().absolute().relative_to(local_dir)
                mount_path = self.remote_path / local_relpath.as_posix()
                yield local_path.resolve(), mount_path

    def watch_entry(self):
        return self.local_dir.resolve().expanduser(), None

    def top_level_paths(self) -> list[tuple[Path, PurePosixPath]]:
        return [(self.local_dir, self.remote_path)]


def module_mount_condition(module_base: Path):
    SKIP_BYTECODE = True  # hard coded for now
    SKIP_DOT_PREFIXED = True

    def condition(f: str):
        path = Path(f)
        if SKIP_BYTECODE and path.suffix == ".pyc":
            return False

        # Check parent dir names to see if file should be included,
        # but ignore dir names above root of mounted module:
        # /a/.venv/site-packages/mymod/foo.py should be included by default
        # /a/my_mod/.config/foo.py should *not* be included by default
        while path != module_base and path != path.parent:
            if SKIP_BYTECODE and path.name == "__pycache__":
                return False

            if SKIP_DOT_PREFIXED and path.name.startswith("."):
                return False

            path = path.parent

        return True

    return condition


def module_mount_ignore_condition(module_base: Path):
    return lambda f: not module_mount_condition(module_base)(str(f))


@dataclasses.dataclass
class _MountedPythonModule(_MountEntry):
    # the purpose of this is to keep printable information about which Python package
    # was mounted. Functionality wise it's the same as mounting a dir or a file with
    # the Module

    module_name: str
    remote_dir: Union[PurePosixPath, str] = ROOT_DIR.as_posix()  # cast needed here for type stub generation...
    ignore: Optional[Callable[[Path], bool]] = None

    def description(self) -> str:
        return f"PythonPackage:{self.module_name}"

    def _proxy_entries(self) -> list[_MountEntry]:
        mount_infos = get_module_mount_info(self.module_name)
        entries = []
        for mount_info in mount_infos:
            is_package, base_path = mount_info
            if is_package:
                remote_dir = PurePosixPath(self.remote_dir, *self.module_name.split("."))
                entries.append(
                    _MountDir(
                        base_path,
                        remote_path=remote_dir,
                        ignore=self.ignore or module_mount_ignore_condition(base_path),
                        recursive=True,
                    )
                )
            else:
                path_segments = self.module_name.split(".")[:-1]
                remote_path = PurePosixPath(self.remote_dir, *path_segments, Path(base_path).name)
                entries.append(
                    _MountFile(
                        local_file=Path(base_path),
                        remote_path=remote_path,
                    )
                )
        return entries

    def get_files_to_upload(self) -> typing.Iterator[tuple[Path, str]]:
        for entry in self._proxy_entries():
            yield from entry.get_files_to_upload()

    def watch_entry(self) -> tuple[Path, Path]:
        for entry in self._proxy_entries():
            # TODO: fix watch for mounts of multi-path packages
            return entry.watch_entry()

    def top_level_paths(self) -> list[tuple[Path, PurePosixPath]]:
        paths = []
        for sub in self._proxy_entries():
            paths.extend(sub.top_level_paths())
        return paths


class NonLocalMountError(Exception):
    # used internally to signal an error when trying to access entries on a non-local mount definition
    pass


class _Mount(_Object, type_prefix="mo"):
    """
    **Deprecated**: Mounts should not be used explicitly anymore, use `Image.add_local_*` commands instead.

    Create a mount for a local directory or file that can be attached
    to one or more Modal functions.

    **Usage**

    ```python notest
    import modal
    import os
    app = modal.App()

    @app.function(mounts=[modal.Mount.from_local_dir("~/foo", remote_path="/root/foo")])
    def f():
        # `/root/foo` has the contents of `~/foo`.
        print(os.listdir("/root/foo/"))
    ```

    Modal syncs the contents of the local directory every time the app runs, but uses the hash of
    the file's contents to skip uploading files that have been uploaded before.
    """

    _entries: Optional[list[_MountEntry]] = None
    _deployment_name: Optional[str] = None
    _namespace: Optional[int] = None
    _environment_name: Optional[str] = None
    _content_checksum_sha256_hex: Optional[str] = None

    @staticmethod
    def _new(entries: list[_MountEntry] = []) -> "_Mount":
        rep = f"Mount({entries})"

        async def mount_content_deduplication_key():
            try:
                included_files = await asyncio.get_event_loop().run_in_executor(None, _select_files, entries)
            except NonLocalMountError:
                return None
            return (_Mount._type_prefix, "local", frozenset(included_files))

        obj = _Mount._from_loader(_Mount._load_mount, rep, deduplication_key=mount_content_deduplication_key)
        obj._entries = entries
        obj._is_local = True
        return obj

    def _extend(self, entry: _MountEntry) -> "_Mount":
        return _Mount._new(self._entries + [entry])

    @property
    def entries(self):
        """mdmd:hidden"""
        if self._entries is None:
            raise NonLocalMountError()
        return self._entries

    def _hydrate_metadata(self, handle_metadata: Optional[Message]):
        assert isinstance(handle_metadata, api_pb2.MountHandleMetadata)
        self._content_checksum_sha256_hex = handle_metadata.content_checksum_sha256_hex

    def _top_level_paths(self) -> list[tuple[Path, PurePosixPath]]:
        # Returns [(local_absolute_path, remote_path), ...] for all top level entries in the Mount
        # Used to determine if a package mount is installed in a sys directory or not
        res: list[tuple[Path, PurePosixPath]] = []
        for entry in self.entries:
            res.extend(entry.top_level_paths())
        return res

    def is_local(self) -> bool:
        """mdmd:hidden"""
        # TODO(erikbern): since any remote ref bypasses the constructor,
        # we can't rely on it to be set. Let's clean this up later.
        return getattr(self, "_is_local", False)

    @staticmethod
    def _add_local_dir(
        local_path: Path,
        remote_path: PurePosixPath,
        ignore: Callable[[Path], bool] = modal.file_pattern_matcher._NOTHING,
    ):
        return _Mount._new()._extend(
            _MountDir(
                local_dir=local_path,
                ignore=ignore,
                remote_path=remote_path,
                recursive=True,
            ),
        )

    def add_local_dir(
        self,
        local_path: Union[str, Path],
        *,
        # Where the directory is placed within in the mount
        remote_path: Union[str, PurePosixPath, None] = None,
        # Predicate filter function for file selection, which should accept a filepath and return `True` for inclusion.
        # Defaults to including all files.
        condition: Optional[Callable[[str], bool]] = None,
        # add files from subdirectories as well
        recursive: bool = True,
    ) -> "_Mount":
        """
        Add a local directory to the `Mount` object.
        """
        local_path = Path(local_path)
        if remote_path is None:
            remote_path = local_path.name
        remote_path = PurePosixPath("/", remote_path)
        if condition is None:

            def include_all(path):
                return True

            condition = include_all

        def converted_condition(path: Path) -> bool:
            return not condition(str(path))

        return self._extend(
            _MountDir(
                local_dir=local_path,
                ignore=converted_condition,
                remote_path=remote_path,
                recursive=recursive,
            ),
        )

    @staticmethod
    def from_local_dir(
        local_path: Union[str, Path],
        *,
        # Where the directory is placed within in the mount
        remote_path: Union[str, PurePosixPath, None] = None,
        # Predicate filter function for file selection, which should accept a filepath and return `True` for inclusion.
        # Defaults to including all files.
        condition: Optional[Callable[[str], bool]] = None,
        # add files from subdirectories as well
        recursive: bool = True,
    ) -> "_Mount":
        """
        **Deprecated:** Use image.add_local_dir() instead

        Create a `Mount` from a local directory.

        **Usage**

        ```python notest
        assets = modal.Mount.from_local_dir(
            "~/assets",
            condition=lambda pth: not ".venv" in pth,
            remote_path="/assets",
        )
        ```
        """
        deprecation_warning(
            (2025, 1, 8),
            MOUNT_DEPRECATION_MESSAGE_PATTERN.format(replacement="image.add_local_dir"),
        )
        return _Mount._from_local_dir(local_path, remote_path=remote_path, condition=condition, recursive=recursive)

    @staticmethod
    def _from_local_dir(
        local_path: Union[str, Path],
        *,
        # Where the directory is placed within in the mount
        remote_path: Union[str, PurePosixPath, None] = None,
        # Predicate filter function for file selection, which should accept a filepath and return `True` for inclusion.
        # Defaults to including all files.
        condition: Optional[Callable[[str], bool]] = None,
        # add files from subdirectories as well
        recursive: bool = True,
    ) -> "_Mount":
        return _Mount._new().add_local_dir(
            local_path, remote_path=remote_path, condition=condition, recursive=recursive
        )

    def add_local_file(
        self,
        local_path: Union[str, Path],
        remote_path: Union[str, PurePosixPath, None] = None,
    ) -> "_Mount":
        """
        Add a local file to the `Mount` object.
        """
        local_path = Path(local_path)
        if remote_path is None:
            remote_path = local_path.name
        remote_path = PurePosixPath("/", remote_path)
        return self._extend(
            _MountFile(
                local_file=local_path,
                remote_path=PurePosixPath(remote_path),
            ),
        )

    @staticmethod
    def from_local_file(local_path: Union[str, Path], remote_path: Union[str, PurePosixPath, None] = None) -> "_Mount":
        """
        **Deprecated**: Use image.add_local_file() instead

        Create a `Mount` mounting a single local file.

        **Usage**

        ```python notest
        # Mount the DBT profile in user's home directory into container.
        dbt_profiles = modal.Mount.from_local_file(
            local_path="~/profiles.yml",
            remote_path="/root/dbt_profile/profiles.yml",
        )
        ```
        """
        deprecation_warning(
            (2025, 1, 8),
            MOUNT_DEPRECATION_MESSAGE_PATTERN.format(replacement="image.add_local_file"),
        )
        return _Mount._from_local_file(local_path, remote_path)

    @staticmethod
    def _from_local_file(local_path: Union[str, Path], remote_path: Union[str, PurePosixPath, None] = None) -> "_Mount":
        return _Mount._new().add_local_file(local_path, remote_path=remote_path)

    @staticmethod
    def _description(entries: list[_MountEntry]) -> str:
        local_contents = [e.description() for e in entries]
        return ", ".join(local_contents)

    @staticmethod
    async def _get_files(entries: list[_MountEntry]) -> AsyncGenerator[FileUploadSpec, None]:
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as exe:
            all_files = await loop.run_in_executor(exe, _select_files, entries)

            futs = []
            for local_filename, remote_filename in all_files:
                logger.debug(f"Mounting {local_filename} as {remote_filename}")
                futs.append(loop.run_in_executor(exe, get_file_upload_spec_from_path, local_filename, remote_filename))

            logger.debug(f"Computing checksums for {len(futs)} files using {exe._max_workers} worker threads")
            for fut in asyncio.as_completed(futs):
                try:
                    yield await fut
                except FileNotFoundError as exc:
                    # Can happen with temporary files (e.g. emacs will write temp files and delete them quickly)
                    logger.info(f"Ignoring file not found: {exc}")

    async def _load_mount(
        self: "_Mount",
        resolver: Resolver,
        existing_object_id: Optional[str],
    ):
        t0 = time.monotonic()

        # Asynchronously list and checksum files with a thread pool, then upload them concurrently.
        n_seen, n_finished = 0, 0
        total_uploads, total_bytes = 0, 0
        accounted_hashes: set[str] = set()
        message_label = _Mount._description(self._entries)
        blob_upload_concurrency = asyncio.Semaphore(16)  # Limit uploads of large files.
        status_row = resolver.add_status_row()

        async def _put_file(file_spec: FileUploadSpec) -> api_pb2.MountFile:
            nonlocal n_seen, n_finished, total_uploads, total_bytes
            n_seen += 1
            status_row.message(f"Creating mount {message_label}: Uploaded {n_finished}/{n_seen} files")

            remote_filename = file_spec.mount_filename
            mount_file = api_pb2.MountFile(
                filename=remote_filename,
                sha256_hex=file_spec.sha256_hex,
                mode=file_spec.mode,
            )

            if file_spec.sha256_hex in accounted_hashes:
                n_finished += 1
                return mount_file

            # Try to catch cases where user modified their local files (e.g. changed git branches)
            # between triggering a build and Modal actually uploading the file
            if config.get("build_validation") != "ignore" and file_spec.source_is_path:
                mtime = os.stat(file_spec.source_description).st_mtime
                if mtime > resolver.build_start:
                    msg = f"{file_spec.source_description} was modified during build process."
                    if config.get("build_validation") == "error":
                        raise modal.exception.ExecutionError(msg)
                    elif config.get("build_validation") == "warn":
                        warnings.warn(msg)

            request = api_pb2.MountPutFileRequest(sha256_hex=file_spec.sha256_hex)
            accounted_hashes.add(file_spec.sha256_hex)
            response = await retry_transient_errors(resolver.client.stub.MountPutFile, request, base_delay=1)

            if response.exists:
                n_finished += 1
                return mount_file

            total_uploads += 1
            total_bytes += file_spec.size

            if file_spec.use_blob:
                logger.debug(f"Creating blob file for {file_spec.source_description} ({file_spec.size} bytes)")
                async with blob_upload_concurrency:
                    with file_spec.source() as fp:
                        blob_id = await blob_upload_file(
                            fp, resolver.client.stub, sha256_hex=file_spec.sha256_hex, md5_hex=file_spec.md5_hex
                        )
                logger.debug(f"Uploading blob file {file_spec.source_description} as {remote_filename}")
                request2 = api_pb2.MountPutFileRequest(data_blob_id=blob_id, sha256_hex=file_spec.sha256_hex)
            else:
                logger.debug(
                    f"Uploading file {file_spec.source_description} to {remote_filename} ({file_spec.size} bytes)"
                )
                request2 = api_pb2.MountPutFileRequest(data=file_spec.content, sha256_hex=file_spec.sha256_hex)

            start_time = time.monotonic()
            while time.monotonic() - start_time < MOUNT_PUT_FILE_CLIENT_TIMEOUT:
                response = await retry_transient_errors(resolver.client.stub.MountPutFile, request2, base_delay=1)
                if response.exists:
                    n_finished += 1
                    return mount_file

            raise modal.exception.MountUploadTimeoutError(f"Mounting of {file_spec.source_description} timed out")

        # Upload files, or check if they already exist.
        n_concurrent_uploads = 512
        files: list[api_pb2.MountFile] = []
        async with aclosing(
            async_map(_Mount._get_files(self._entries), _put_file, concurrency=n_concurrent_uploads)
        ) as stream:
            async for file in stream:
                files.append(file)

        if not files:
            logger.warning(f"Mount of '{message_label}' is empty.")

        # Build the mount.
        status_row.message(f"Creating mount {message_label}: Finalizing index of {len(files)} files")
        if self._deployment_name:
            req = api_pb2.MountGetOrCreateRequest(
                deployment_name=self._deployment_name,
                namespace=self._namespace,
                environment_name=self._environment_name,
                object_creation_type=api_pb2.OBJECT_CREATION_TYPE_CREATE_FAIL_IF_EXISTS,
                files=files,
            )
        elif resolver.app_id is not None:
            req = api_pb2.MountGetOrCreateRequest(
                object_creation_type=api_pb2.OBJECT_CREATION_TYPE_ANONYMOUS_OWNED_BY_APP,
                files=files,
                app_id=resolver.app_id,
            )
        else:
            req = api_pb2.MountGetOrCreateRequest(
                object_creation_type=api_pb2.OBJECT_CREATION_TYPE_EPHEMERAL,
                files=files,
                environment_name=resolver.environment_name,
            )

        resp = await retry_transient_errors(resolver.client.stub.MountGetOrCreate, req, base_delay=1)
        status_row.finish(f"Created mount {message_label}")

        logger.debug(f"Uploaded {total_uploads} new files and {total_bytes} bytes in {time.monotonic() - t0}s")
        self._hydrate(resp.mount_id, resolver.client, resp.handle_metadata)

    @staticmethod
    def from_local_python_packages(
        *module_names: str,
        remote_dir: Union[str, PurePosixPath] = ROOT_DIR.as_posix(),
        # Predicate filter function for file selection, which should accept a filepath and return `True` for inclusion.
        # Defaults to including all files.
        condition: Optional[Callable[[str], bool]] = None,
        ignore: Optional[Union[Sequence[str], Callable[[Path], bool]]] = None,
    ) -> "_Mount":
        """
        **Deprecated**: Use image.add_local_python_source instead

        Returns a `modal.Mount` that makes local modules listed in `module_names` available inside the container.
        This works by mounting the local path of each module's package to a directory inside the container
        that's on `PYTHONPATH`.

        **Usage**

        ```python notest
        import modal
        import my_local_module

        app = modal.App()

        @app.function(mounts=[
            modal.Mount.from_local_python_packages("my_local_module", "my_other_module"),
        ])
        def f():
            my_local_module.do_stuff()
        ```
        """
        deprecation_warning(
            (2025, 1, 8),
            MOUNT_DEPRECATION_MESSAGE_PATTERN.format(replacement="image.add_local_python_source"),
        )
        return _Mount._from_local_python_packages(
            *module_names, remote_dir=remote_dir, condition=condition, ignore=ignore
        )

    @staticmethod
    def _from_local_python_packages(
        *module_names: str,
        remote_dir: Union[str, PurePosixPath] = ROOT_DIR.as_posix(),
        # Predicate filter function for file selection, which should accept a filepath and return `True` for inclusion.
        # Defaults to including all files.
        condition: Optional[Callable[[str], bool]] = None,
        ignore: Optional[Union[Sequence[str], Callable[[Path], bool]]] = None,
    ) -> "_Mount":
        if condition is not None:
            if ignore is not None:
                raise InvalidError("Cannot specify both `ignore` and `condition`")

            def converted_condition(path: Path) -> bool:
                return not condition(str(path))

            ignore = converted_condition
        elif isinstance(ignore, list):
            ignore = FilePatternMatcher(*ignore)

        mount = _Mount._new()
        for module_name in module_names:
            mount = mount._extend(_MountedPythonModule(module_name, remote_dir, ignore))
        return mount

    @staticmethod
    @renamed_parameter((2024, 12, 18), "label", "name")
    def from_name(
        name: str,
        *,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        environment_name: Optional[str] = None,
    ) -> "_Mount":
        """mdmd:hidden"""

        async def _load(provider: _Mount, resolver: Resolver, existing_object_id: Optional[str]):
            req = api_pb2.MountGetOrCreateRequest(
                deployment_name=name,
                namespace=namespace,
                environment_name=_get_environment_name(environment_name, resolver),
            )
            response = await resolver.client.stub.MountGetOrCreate(req)
            provider._hydrate(response.mount_id, resolver.client, response.handle_metadata)

        return _Mount._from_loader(_load, "Mount()", hydrate_lazily=True)

    @classmethod
    @renamed_parameter((2024, 12, 18), "label", "name")
    async def lookup(
        cls: type["_Mount"],
        name: str,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        client: Optional[_Client] = None,
        environment_name: Optional[str] = None,
    ) -> "_Mount":
        """mdmd:hidden"""
        deprecation_warning(
            (2025, 1, 27),
            "`modal.Mount.lookup` is deprecated and will be removed in a future release."
            " It can be replaced with `modal.Mount.from_name`."
            "\n\nSee https://modal.com/docs/guide/modal-1-0-migration for more information.",
        )
        obj = _Mount.from_name(name, namespace=namespace, environment_name=environment_name)
        if client is None:
            client = await _Client.from_env()
        resolver = Resolver(client=client)
        await resolver.load(obj)
        return obj

    async def _deploy(
        self: "_Mount",
        deployment_name: Optional[str] = None,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        environment_name: Optional[str] = None,
        client: Optional[_Client] = None,
    ) -> None:
        check_object_name(deployment_name, "Mount")
        environment_name = _get_environment_name(environment_name, resolver=None)
        self._deployment_name = deployment_name
        self._namespace = namespace
        self._environment_name = environment_name
        if client is None:
            client = await _Client.from_env()
        resolver = Resolver(client=client, environment_name=environment_name)
        await resolver.load(self)

    def _get_metadata(self) -> api_pb2.MountHandleMetadata:
        if self._content_checksum_sha256_hex is None:
            raise ValueError("Trying to access checksum of unhydrated mount")

        return api_pb2.MountHandleMetadata(content_checksum_sha256_hex=self._content_checksum_sha256_hex)


Mount = synchronize_api(_Mount)


def _create_client_mount():
    # TODO(erikbern): make this a static method on the Mount class
    import synchronicity

    import modal

    # Get the base_path because it also contains `modal_proto`.
    modal_parent_dir, _ = os.path.split(modal.__path__[0])
    client_mount = _Mount._new()

    for pkg_name in MODAL_PACKAGES:
        package_base_path = Path(modal_parent_dir) / pkg_name
        client_mount = client_mount.add_local_dir(
            package_base_path,
            remote_path=f"/pkg/{pkg_name}",
            condition=module_mount_condition(package_base_path),
            recursive=True,
        )

    # Mount synchronicity, so version changes don't trigger image rebuilds for users.
    synchronicity_base_path = Path(synchronicity.__path__[0])
    client_mount = client_mount.add_local_dir(
        synchronicity_base_path,
        remote_path="/pkg/synchronicity",
        condition=module_mount_condition(synchronicity_base_path),
        recursive=True,
    )
    return client_mount


create_client_mount = synchronize_api(_create_client_mount)


def _get_client_mount():
    # TODO(erikbern): make this a static method on the Mount class
    if config["sync_entrypoint"]:
        return _create_client_mount()
    else:
        return _Mount.from_name(client_mount_name(), namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL)


SYS_PREFIXES = {
    Path(p)
    for p in (
        sys.prefix,
        sys.base_prefix,
        sys.exec_prefix,
        sys.base_exec_prefix,
        *sysconfig.get_paths().values(),
        *site.getsitepackages(),
        site.getusersitepackages(),
    )
}


SYS_PREFIXES |= {p.resolve() for p in SYS_PREFIXES}

MODAL_PACKAGES = ["modal", "modal_proto", "modal_version"]


def _is_modal_path(remote_path: PurePosixPath):
    path_prefix = remote_path.parts[:3]
    remote_python_paths = [("/", "root"), ("/", "pkg")]
    for base in remote_python_paths:
        is_modal_path = path_prefix in [base + (mod,) for mod in MODAL_PACKAGES] or path_prefix == base + (
            "synchronicity",
        )
        if is_modal_path:
            return True
    return False


def get_sys_modules_mounts() -> dict[str, _Mount]:
    """mdmd:hidden

    Auto-mount local modules that have been imported in global scope.
    This may or may not include the "entrypoint" of the function as well, depending on how modal is invoked
    Note: sys.modules may change during the iteration
    """
    auto_mounts = {}
    top_level_modules = []
    skip_prefixes = set()
    for name, module in sorted(sys.modules.items(), key=lambda kv: len(kv[0])):
        parent = name.rsplit(".")[0]
        if parent and parent in skip_prefixes:
            skip_prefixes.add(name)
            continue
        skip_prefixes.add(name)
        top_level_modules.append((name, module))

    for module_name, module in top_level_modules:
        if module_name.startswith("__"):
            # skip "built in" modules like __main__ and __mp_main__
            # the running function's main file should be included anyway
            continue

        try:
            # at this point we don't know if the sys.modules module should be mounted or not
            potential_mount = _Mount._from_local_python_packages(module_name)
            mount_paths = potential_mount._top_level_paths()
        except ModuleNotMountable:
            # this typically happens if the module is a built-in, has binary components or doesn't exist
            continue

        for local_path, remote_path in mount_paths:
            if any(local_path.is_relative_to(p) for p in SYS_PREFIXES) or _is_modal_path(remote_path):
                # skip any module that has paths in SYS_PREFIXES, or would overwrite the modal Package in the container
                break
        else:
            auto_mounts[module_name] = potential_mount

    return auto_mounts
