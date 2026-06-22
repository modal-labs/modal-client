# Copyright Modal Labs 2026
import asyncio
import importlib.metadata
import json
import os
import sys
import traceback
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import (
    ClassVar,
    Optional,
)

from synchronicity.async_wrap import asynccontextmanager

from modal._runtime import gpu_memory_snapshot
from modal._serialization import (
    pickle_exception,
    pickle_traceback,
)
from modal._traceback import print_exception
from modal._utils.async_utils import asyncify, synchronize_api
from modal._utils.blob_utils import format_blob_data
from modal._utils.grpc_utils import Retry
from modal._utils.package_utils import parse_major_minor_version
from modal.client import _Client
from modal.config import config, logger
from modal_proto import api_pb2


class UserException(Exception):
    """Used to shut down the task gracefully."""


class _TaskLifecycleManager:
    _singleton: ClassVar[Optional["_TaskLifecycleManager"]] = None

    task_id: str
    function_id: str
    function_def: api_pb2.Function
    checkpoint_id: str | None
    _client: _Client
    _cuda_checkpoint_session: gpu_memory_snapshot.CudaCheckpointSession | None

    def _init(
        self,
        task_id: str,
        function_id: str,
        function_def: api_pb2.Function,
        checkpoint_id: str | None,
        client: _Client,
    ) -> None:
        self.task_id = task_id
        self.function_id = function_id
        self.function_def = function_def
        self.checkpoint_id = checkpoint_id
        self._client = client
        self._cuda_checkpoint_session = None

    def __new__(
        cls,
        task_id: str,
        function_id: str,
        function_def: api_pb2.Function,
        checkpoint_id: str | None,
        client: _Client,
    ) -> "_TaskLifecycleManager":
        cls._singleton = super().__new__(cls)
        cls._singleton._init(task_id, function_id, function_def, checkpoint_id, client)
        return cls._singleton

    @classmethod
    def _reset_singleton(cls) -> None:
        """Only used for tests."""
        cls._singleton = None

    @asynccontextmanager
    async def handle_task_lifecycle_exception(
        self,
    ) -> AsyncGenerator[None, None]:
        """Report lifecycle exceptions as task-level failures and stop the task."""
        try:
            yield
        except KeyboardInterrupt:
            # Send no task result in case we get sigint:ed by the runner
            # The status of the input should have been handled externally already in that case
            raise
        except BaseException as exc:
            if isinstance(exc, ImportError):
                # Catches errors raised by global scope imports
                check_fastapi_pydantic_compatibility(exc)

            # Since this is on a different thread, sys.exc_info() can't find the exception in the stack.
            print_exception(type(exc), exc, exc.__traceback__)

            serialized_tb, tb_line_cache = pickle_traceback(exc, self.task_id)

            data_or_blob = await format_blob_data(pickle_exception(exc), self._client.stub)
            result = api_pb2.GenericResult(
                status=api_pb2.GenericResult.GENERIC_STATUS_FAILURE,
                **data_or_blob,
                # TODO: there is no way to communicate the data format here
                #   since it usually goes on the envelope outside of GenericResult
                exception=repr(exc),
                traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                serialized_tb=serialized_tb or b"",
                tb_line_cache=tb_line_cache or b"",
            )

            req = api_pb2.TaskResultRequest(result=result)
            await self._client.stub.TaskResult(req)

            # Shut down the task gracefully
            raise UserException()

    async def volume_commit(self, volume_ids: list[str]) -> None:
        """
        Perform volume commit for given `volume_ids`.
        Only used on container exit to persist uncommitted changes on behalf of user.
        """
        if not volume_ids:
            return
        await asyncify(os.sync)()
        results = await asyncio.gather(
            *[
                self._client.stub.VolumeCommit(
                    api_pb2.VolumeCommitRequest(volume_id=v_id),
                    retry=Retry(
                        max_retries=9,
                        base_delay=0.25,
                        max_delay=256,
                        delay_factor=2,
                    ),
                )
                for v_id in volume_ids
            ],
            return_exceptions=True,
        )
        for volume_id, res in zip(volume_ids, results):
            if isinstance(res, Exception):
                logger.error(f"modal.Volume background commit failed for {volume_id}. Exception: {res}")
            else:
                logger.debug(f"modal.Volume background commit success for {volume_id}.")

    async def memory_restore(self) -> None:
        # Busy-wait for restore. `/__modal/restore-state.json` is created
        # by the worker process with updates to the container config.
        restored_path = Path(config.get("restore_state_path"))
        logger.debug("Waiting for restore")
        while not restored_path.exists():
            await asyncio.sleep(0.01)
            continue
        logger.debug("Container: restored")

        # Look for state file and create new client with updated credentials.
        # State data is serialized with key-value pairs, example: {"task_id": "tk-000"}
        with restored_path.open("r") as file:
            restored_state = json.load(file)

        # Start a debugger if the worker tells us to
        if int(restored_state.get("snapshot_debug", 0)):
            logger.debug("Entering snapshot debugger")
            breakpoint()  # noqa: T100

        # Local task lifecycle manager state.
        manager_name = type(self).__name__.removeprefix("_")
        for key in ["task_id", "function_id"]:
            if value := restored_state.get(key):
                logger.debug(f"Updating {manager_name}.{key} = {value}")
                setattr(self, key, restored_state[key])

        # Env vars and global state.
        for key, value in restored_state.items():
            # Empty string indicates that value does not need to be updated.
            if value != "":
                config.override_locally(key, value)

        # Restore GPU memory.
        if self.function_def._experimental_enable_gpu_snapshot and self.function_def.resources.gpu_config.gpu_type:
            logger.debug("GPU memory snapshot enabled. Attempting to restore GPU memory.")

            try:
                if self._cuda_checkpoint_session is None:
                    raise gpu_memory_snapshot.CudaCheckpointException("CudaCheckpointSession not found")
                self._cuda_checkpoint_session.restore()
            except gpu_memory_snapshot.CudaCheckpointException as exc:
                logger.warning(f"GPU memory snapshot restore failed; retrying task without snapshot. Error: {exc}")
                sys.stderr.flush()
                # Exit with a sentinel code that the runtime will use to retry the task without a snapshot.
                os._exit(gpu_memory_snapshot.SNAPSHOT_RESTORE_FAILED_EXIT_CODE)

        self._client = await _Client.from_env()

    async def memory_snapshot(self) -> None:
        """Message server indicating that function is ready to be checkpointed."""
        if self.checkpoint_id:
            logger.debug(f"Checkpoint ID: {self.checkpoint_id} (Memory Snapshot ID)")
        else:
            raise ValueError("No checkpoint ID provided for memory snapshot")

        if self.function_def._experimental_enable_gpu_snapshot and self.function_def.resources.gpu_config.gpu_type:
            logger.debug("GPU memory snapshot enabled. Attempting to snapshot GPU memory.")

            self._cuda_checkpoint_session = gpu_memory_snapshot.CudaCheckpointSession()
            self._cuda_checkpoint_session.checkpoint()

        await self._client.stub.ContainerCheckpoint(
            api_pb2.ContainerCheckpointRequest(checkpoint_id=self.checkpoint_id)
        )

        await self._client._close(prep_for_restore=True)

        logger.debug("Memory snapshot request sent. Connection closed.")
        await self.memory_restore()


TaskLifecycleManager = synchronize_api(_TaskLifecycleManager)


def check_fastapi_pydantic_compatibility(exc: ImportError) -> None:
    """Add a helpful note to an exception that is likely caused by a pydantic<>fastapi version incompatibility.

    We need this becasue the legacy set of container requirements (image_builder_version=2023.12) contains a
    version of fastapi that is not forwards-compatible with pydantic 2.0+, and users commonly run into issues
    building an image that specifies a more recent version only for pydantic.
    """
    note = (
        "Please ensure that your Image contains compatible versions of fastapi and pydantic."
        " If using pydantic>=2.0, you must also install fastapi>=0.100."
    )
    name = exc.name or ""
    if name.startswith("pydantic"):
        try:
            fastapi_version = parse_major_minor_version(importlib.metadata.version("fastapi"))
            pydantic_version = parse_major_minor_version(importlib.metadata.version("pydantic"))
            if pydantic_version >= (2, 0) and fastapi_version < (0, 100):
                if sys.version_info < (3, 11):
                    # https://peps.python.org/pep-0678/
                    exc.__notes__ = [note]  # type: ignore
                else:
                    exc.add_note(note)
        except Exception:
            # Since we're just trying to add a helpful message, don't fail here
            pass
