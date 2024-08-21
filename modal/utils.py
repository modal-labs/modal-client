import asyncio
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, AsyncIterator, Callable, List, Optional, Tuple, Union

from click import UsageError
from grpclib import GRPCError, Status

from modal._pty import get_pty_info
from modal._utils.async_utils import TaskContext, synchronizer
from modal._utils.grpc_utils import retry_transient_errors
from modal.config import logger
from modal.container_process import _ContainerProcess
from modal.network_file_system import _NetworkFileSystem
from modal.volume import FileEntry, FileEntryType, _Volume, _VolumeUploadContextManager
from modal_proto import api_pb2

from ._output import OutputManager, get_app_logs_loop
from .client import _Client
from .environments import ensure_env
from .exception import NotFoundError
from .object import _get_environment_name

# helpers


async def _stream_app_or_task_logs(app_id: Optional[str] = None, task_id: Optional[str] = None):
    client = await _Client.from_env()
    output_mgr = OutputManager(status_spinner_text=f"Tailing logs for {app_id}")
    try:
        with output_mgr.show_status_spinner():
            await get_app_logs_loop(client, output_mgr, app_id, task_id)
    except asyncio.CancelledError:
        pass
    except GRPCError as exc:
        if exc.status in (Status.INVALID_ARGUMENT, Status.NOT_FOUND):
            raise UsageError(exc.message)
        else:
            raise
    except KeyboardInterrupt:
        pass


async def _get_app_id_from_name(name: str, env: Optional[str], client: Optional[_Client] = None) -> str:
    if client is None:
        client = await _Client.from_env()
    env_name = ensure_env(env)
    request = api_pb2.AppGetByDeploymentNameRequest(
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE, name=name, environment_name=env_name
    )
    try:
        resp = await client.stub.AppGetByDeploymentName(request)
    except GRPCError as exc:
        if exc.status in (Status.INVALID_ARGUMENT, Status.NOT_FOUND):
            raise UsageError(exc.message or "")
        raise
    if not resp.app_id:
        env_comment = f" in the '{env_name}' environment" if env_name else ""
        raise NotFoundError(f"Could not find a deployed app named '{name}'{env_comment}.")
    return resp.app_id


@synchronizer.create_blocking
async def get_app_id(app_identifier: str, env: Optional[str], client: Optional[_Client] = None) -> str:
    """Resolve an app_identifier that may be a name or an ID into an ID."""
    if re.match(r"^ap-[a-zA-Z0-9]{22}$", app_identifier):
        return app_identifier
    return await _get_app_id_from_name(app_identifier, env, client)


PIPE_PATH = Path("-")


async def _volume_download(
    volume: Union[_NetworkFileSystem, _Volume],
    remote_path: str,
    local_destination: Path,
    overwrite: bool,
    progress_cb: Callable,
):
    is_pipe = local_destination == PIPE_PATH

    q: asyncio.Queue[Tuple[Optional[Path], Optional[FileEntry]]] = asyncio.Queue()
    num_consumers = 1 if is_pipe else 10  # concurrency limit for downloading files

    async def producer():
        iterator: AsyncIterator[FileEntry]
        if isinstance(volume, _Volume):
            iterator = volume.iterdir(remote_path, recursive=True)
        else:
            iterator = volume.iterdir(remote_path)  # NFS still supports "glob" paths

        async for entry in iterator:
            if is_pipe:
                await q.put((None, entry))
            else:
                start_path = Path(remote_path).parent.as_posix().split("*")[0]
                rel_path = PurePosixPath(entry.path).relative_to(start_path.lstrip("/"))
                if local_destination.is_dir():
                    output_path = local_destination / rel_path
                else:
                    output_path = local_destination
                if output_path.exists():
                    if overwrite:
                        if output_path.is_file():
                            os.remove(output_path)
                        else:
                            shutil.rmtree(output_path)
                    else:
                        raise UsageError(
                            f"Output path '{output_path}' already exists. Use --force to overwrite the output directory"
                        )
                await q.put((output_path, entry))
        # No more entries to process; issue one shutdown message for each consumer.
        for _ in range(num_consumers):
            await q.put((None, None))

    async def consumer():
        while True:
            output_path, entry = await q.get()
            if entry is None:
                return
            try:
                if is_pipe:
                    if entry.type == FileEntryType.FILE:
                        progress_task_id = progress_cb(name=entry.path, size=entry.size)
                        async for chunk in volume.read_file(entry.path):
                            sys.stdout.buffer.write(chunk)
                            progress_cb(task_id=progress_task_id, advance=len(chunk))
                        progress_cb(task_id=progress_task_id, complete=True)
                else:
                    if entry.type == FileEntryType.FILE:
                        progress_task_id = progress_cb(name=entry.path, size=entry.size)
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        with output_path.open("wb") as fp:
                            b = 0
                            async for chunk in volume.read_file(entry.path):
                                b += fp.write(chunk)
                                progress_cb(task_id=progress_task_id, advance=len(chunk))
                        logger.debug(f"Wrote {b} bytes to {output_path}")
                        progress_cb(task_id=progress_task_id, complete=True)
                    elif entry.type == FileEntryType.DIRECTORY:
                        output_path.mkdir(parents=True, exist_ok=True)
            finally:
                q.task_done()

    consumers = [consumer() for _ in range(num_consumers)]
    await TaskContext.gather(producer(), *consumers)
    progress_cb(complete=True)
    sys.stdout.flush()


# deploy

# NOTE:
# only has deploy command -- probably not relevant from programmatic use

# launch

# NOTE:
# only has launch vscode/jupyter command -- probably not relevant from programmatic use

# run

# NOTE:
# only has run command -- probably not relevant from programmatic use

# serve

# NOTE:
# only has serve command -- probably not relevant from programmatic use

# shell

# NOTE:
# only has shell command -- probably not relevant from programmatic use

# app


@synchronizer.create_blocking
async def stream_app_logs(app_id: str):
    await _stream_app_or_task_logs(app_id=app_id)


@dataclass
class AppDataClass:
    app_id: str
    description: str
    state: int
    created_at: datetime
    stopped_at: datetime
    n_running_tasks: int
    name: str


def timestamp_to_local(ts: float) -> str:
    if ts > 0:
        locale_tz = datetime.now().astimezone().tzinfo
        return datetime.fromtimestamp(ts, tz=locale_tz)
    else:
        return None


@synchronizer.create_blocking
async def list_apps(env: Optional[str] = None) -> List[AppDataClass]:
    env = ensure_env(env)
    client = await _Client.from_env()

    resp: api_pb2.AppListResponse = await client.stub.AppList(
        api_pb2.AppListRequest(environment_name=_get_environment_name(env))
    )

    return [
        AppDataClass(
            app_id=app.app_id,
            description=app.description,
            state=app.state,
            created_at=timestamp_to_local(app.created_at),
            stopped_at=timestamp_to_local(app.stopped_at),
            n_running_tasks=app.n_running_tasks,
            name=app.name,
        )
        for app in resp.apps
    ]


@synchronizer.create_blocking
async def rollback_app(app_identifier: str, version: str, env: Optional[str] = None):
    env = ensure_env(env)
    client = await _Client.from_env()
    app_id = await get_app_id.aio(app_identifier, env, client)
    if not version:
        version_number = -1
    else:
        if m := re.match(r"v(\d+)", version):
            version_number = int(m.group(1))
        else:
            raise UsageError(f"Invalid version specifer: {version}")
    req = api_pb2.AppRollbackRequest(app_id=app_id, version=version_number)
    await client.stub.AppRollback(req)


@synchronizer.create_blocking
async def stop_app(app_identifier: str, env: Optional[str] = None):
    client = await _Client.from_env()
    app_id = await get_app_id.aio(app_identifier, env)
    req = api_pb2.AppStopRequest(app_id=app_id, source=api_pb2.APP_STOP_SOURCE_CLI)
    await client.stub.AppStop(req)


@dataclass
class AppDeploymentHistoryDataClass:
    app_id: str
    version: int
    client_version: str
    deployed_at: float
    deployed_by: str
    tag: str
    rollback_version: int
    rollback_allowed: bool


@synchronizer.create_blocking
async def get_app_deployment_history(app_identifier: str, env: Optional[str] = None):
    env = ensure_env(env)
    client = await _Client.from_env()
    app_id = await get_app_id.aio(app_identifier, env, client)
    resp = await client.stub.AppDeploymentHistory(api_pb2.AppDeploymentHistoryRequest(app_id=app_id))

    return [
        AppDeploymentHistoryDataClass(
            app_id=app_deployment_history.app_id,
            version=app_deployment_history.version,
            client_version=app_deployment_history.client_version,
            deployed_at=app_deployment_history.deployed_at,
            deployed_by=app_deployment_history.deployed_by,
            tag=app_deployment_history.tag,
            rollback_version=app_deployment_history.rollback_version,
            rollback_allowed=app_deployment_history.rollback_allowed,
        )
        for app_deployment_history in resp.app_deployment_histories
    ]


# container


@synchronizer.create_blocking
async def stream_container_logs(task_id: str):
    await _stream_app_or_task_logs(task_id=task_id)


@dataclass
class ContainerDataClass:
    task_id: str
    app_id: str
    app_description: str
    started_at: datetime


@synchronizer.create_blocking
async def list_containers():
    client = await _Client.from_env()
    res = await client.stub.TaskList(api_pb2.TaskListRequest())
    res.tasks.sort(key=lambda task: task.started_at, reverse=True)

    return [
        ContainerDataClass(
            task_id=task.task_id,
            app_id=task.app_id,
            app_description=task.app_description,
            started_at=timestamp_to_local(task.started_at),
        )
        for task in res.tasks
    ]


@synchronizer.create_blocking
async def container_exec(container_id: str, commands: List[str], pty: Optional[bool] = True):
    client = await _Client.from_env()

    req = api_pb2.ContainerExecRequest(
        task_id=container_id, command=commands, pty_info=get_pty_info(shell=True) if pty else None
    )
    res = await client.stub.ContainerExec(req)
    await _ContainerProcess(res.exec_id, client).attach(pty=pty)


@synchronizer.create_blocking
async def stop_container(container_id: str):
    client = await _Client.from_env()
    request = api_pb2.ContainerStopRequest(task_id=container_id)
    await retry_transient_errors(client.stub.ContainerStop, request)


# config
# environment
# profile
# token
# dict
# nfs
# queue
# secret
# volume


def create_volume(
    name: str,
    env: Optional[str] = None,
    version: Optional[int] = None,
):
    ensure_env(env)
    _Volume.create_deployed(name, environment_name=env, version=version)


@synchronizer.create_blocking
async def get_volume(
    volume_name: str,
    remote_path: str,
    local_destination: Optional[str] = ".",
    force: bool = False,
    env: Optional[str] = None,
    progress_cb: Optional[Callable[[Any], None]] = None,
):
    ensure_env(env)
    destination = Path(local_destination)
    volume = await _Volume.lookup(volume_name, environment_name=env)
    progress_cb = progress_cb or (lambda *_, **__: None)
    await _volume_download(volume, remote_path, destination, force, progress_cb=progress_cb)


@dataclass
class VolumeDataClass:
    label: str
    volume_id: str
    created_at: datetime
    environment_name: str


@synchronizer.create_blocking
async def list_volumes(env: Optional[str] = None):
    client = await _Client.from_env()
    res = await retry_transient_errors(client.stub.VolumeList, api_pb2.VolumeListRequest(environment_name=env))

    return [
        VolumeDataClass(
            label=volume.label,
            volume_id=volume.volume_id,
            created_at=timestamp_to_local(volume.created_at),
            environment_name=res.environment_name,
        )
        for volume in res.items
    ]


@synchronizer.create_blocking
async def volume_ls(
    volume_name: str,
    path: Optional[str] = "/",
    env: Optional[str] = None,
):
    ensure_env(env)
    vol = await _Volume.lookup(volume_name, environment_name=env)
    if not isinstance(vol, _Volume):
        raise UsageError("The specified app entity is not a modal.Volume")

    try:
        return await vol.listdir(path)
    except GRPCError as exc:
        if exc.status in (Status.INVALID_ARGUMENT, Status.NOT_FOUND):
            raise UsageError(exc.message)
        raise


@synchronizer.create_blocking
async def upload_to_volume(
    volume_name: str,
    local_path: str,
    remote_path: Optional[str] = "/",
    force: Optional[bool] = False,
    env: Optional[str] = None,
):
    ensure_env(env)
    vol = _Volume.lookup(volume_name, environment_name=env)
    if not isinstance(vol, Volume):
        raise UsageError("The specified app entity is not a modal.Volume")

    if remote_path.endswith("/"):
        remote_path = remote_path + os.path.basename(local_path)

    if Path(local_path).is_dir():
        try:
            async with _VolumeUploadContextManager(
                vol.object_id, vol._client, progress_cb=progress_handler.progress, force=force
            ) as batch:
                batch.put_directory(local_path, remote_path)
        except FileExistsError as exc:
            raise UsageError(str(exc))
    elif "*" in local_path:
        raise UsageError("Glob uploads are currently not supported")
    else:
        try:
            async with _VolumeUploadContextManager(
                vol.object_id, vol._client, progress_cb=progress_handler.progress, force=force
            ) as batch:
                batch.put_file(local_path, remote_path)

        except FileExistsError as exc:
            raise UsageError(str(exc))


@synchronizer.create_blocking
async def volume_rm(
    volume_name: str,
    remote_path: str,
    recursive: Optional[bool] = False,
    env: Optional[str] = None,
):
    pass


@synchronizer.create_blocking
async def volume_cp(
    volume_name: str,
    paths: List[str],  # accepts multiple paths, last path is treated as destination path
    env: Optional[str] = None,
):
    pass


@synchronizer.create_blocking
async def delete_volume(
    volume_name: str,
    yes: Optional[bool] = False,
    confirm: Optional[bool] = False,
    env: Optional[str] = None,
):
    pass


# setup
