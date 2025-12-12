import os
from pathlib import Path
from typing import Optional

from modal._utils.async_utils import synchronize_api
from modal.exception import InvalidError
from modal.image import _Image
from modal.sandbox import _Sandbox
from modal_proto import task_command_router_pb2 as sr_pb2


async def _mount_directory(sandbox: _Sandbox, path: Path, image: Optional[_Image]):
    """Mount an image at a path in the sandbox filesystem."""
    image_id = None

    if image:
        if not image._object_id:
            # FIXME
            raise InvalidError("currently only images created with from_id are supported")
        image_id = image._object_id

    task_id = await sandbox._get_task_id()
    command_router_client = await sandbox._get_command_router_client(task_id)
    req = sr_pb2.TaskMountImageRequest(task_id=task_id, path=os.fsencode(path), image_id=image_id)
    await command_router_client.mount_image(req)


mount_directory = synchronize_api(_mount_directory)


async def _snapshot_directory(sandbox: _Sandbox, path: Path) -> _Image:
    """Snapshot a directory to a new image."""

    task_id = await sandbox._get_task_id()
    command_router_client = await sandbox._get_command_router_client(task_id)
    req = sr_pb2.TaskSnapshotImageMountRequest(task_id=task_id, path=os.fsencode(path))
    res = await command_router_client.snapshot_image_mount(req)
    return _Image._new_hydrated(res.image_id, sandbox._client, None)


snapshot_directory = synchronize_api(_snapshot_directory)
