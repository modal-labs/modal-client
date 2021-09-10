import asyncio
import concurrent.futures
import hashlib
import os
import sys
import time

from typing import Dict

from .async_utils import retry
from .config import logger
from .function import decorate_function
from .grpc_utils import GRPC_REQUEST_TIMEOUT, BLOCKING_REQUEST_TIMEOUT
from .object import Object
from .proto import api_pb2


def get_sha256_hex_from_content(content):
    m = hashlib.sha256()
    m.update(content)
    return m.hexdigest()


def get_sha256_hex_from_filename(filename, rel_filename):
    # Somewhat CPU intensive, so we run it in a thread/process
    content = open(filename, 'rb').read()
    return filename, rel_filename, get_sha256_hex_from_content(content)


async def get_files(local_dir, condition):
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor() as exe:
        futs = []
        for root, dirs, files in os.walk(local_dir):
            for name in files:
                filename = os.path.join(root, name)
                rel_filename = os.path.relpath(filename, local_dir)
                if condition(filename):
                    futs.append(loop.run_in_executor(exe, get_sha256_hex_from_filename, filename, rel_filename))
        logger.debug(f'Computing checksums for {len(futs)} files using {exe._max_workers} workers')
        for fut in asyncio.as_completed(futs):
            filename, rel_filename, sha256_hex = await fut
            yield filename, rel_filename, sha256_hex


class Mount(Object):
    def __init__(self, local_dir, remote_dir, condition):
        super().__init__(
            args=dict(
                local_dir=local_dir,
                remote_dir=remote_dir,
                condition=condition,
            )
        )

    async def _register_file_requests(self, mount_id, hashes):
        async for filename, rel_filename, sha256_hex in get_files(self.args.local_dir, self.args.condition):
            remote_filename = os.path.join(self.args.remote_dir, rel_filename)  # won't work on windows
            self._remote_to_local_filename[remote_filename] = filename
            request = api_pb2.MountRegisterFileRequest(
                filename=remote_filename, sha256_hex=sha256_hex, mount_id=mount_id
            )
            hashes[filename] = sha256_hex
            yield request

    async def _upload_file_requests(self, client, mount_id, hashes, remote_to_local_filename):
        t0 = time.time()
        n_files, n_missing_files, total_bytes = 0, 0, 0
        async for response in client.stub.MountRegisterFile(self._register_file_requests(mount_id, hashes)):
            n_files += 1
            if not response.exists:
                filename = remote_to_local_filename[response.filename]
                data = open(filename, 'rb').read()
                n_missing_files += 1
                total_bytes += len(data)
                logger.debug(f'Uploading file {filename} to {response.filename} ({len(data)} bytes)')
                request = api_pb2.MountUploadFileRequest(data=data, sha256_hex=hashes[filename], size=len(data), mount_id=mount_id)
                yield request

        logger.info(f'Uploaded {n_missing_files}/{n_files} files and {total_bytes} bytes in {time.time() - t0}s')

    async def join(self, client):
        # TODO: I think in theory we could split the get_files iterator and launch multiple concurrent
        # calls to MountRegisterFileRequest and MountUploadFileRequest. This would speed up a lot of the
        # serial operations on the server side (like hitting Redis for every file serially).
        # Another option is to parallelize more on the server side.

        hashes = {}  # TODO: get rid of
        remote_to_local_filename = {}  # TODO: get rid of

        if not self.object_id:
            req = api_pb2.MountCreateRequest(session_id=client.session_id)
            resp = await client.stub.MountCreate(req)
            mount_id = resp.mount_id

            logger.debug(f'Uploading mount {mount_id}')
            await client.stub.MountUploadFile(self._upload_file_requests(client, mount_id))

            req = api_pb2.MountDoneRequest(mount_id=mount_id)
            await client.stub.MountDone(req)

            self.object_id = object_id

        logger.debug('Waiting for mount %s' % self.mount_id)
        while True:
            request = api_pb2.MountJoinRequest(mount_id=self.object_id, timeout=BLOCKING_REQUEST_TIMEOUT)
            response = await retry(client.stub.MountJoin)(request, timeout=GRPC_REQUEST_TIMEOUT)
            if not response.result.status:
                continue
            elif response.result.status == api_pb2.GenericResult.Status.FAILURE:
                raise Exception(response.result.exception)
            elif response.result.status == api_pb2.GenericResult.Status.SUCCESS:
                break
            else:
                raise Exception('Unknown status %s!' % response.result.status)

        return self.mount_id


# The default mount will upload all Python files in the current workdir into /root
mount_py_in_workdir_into_root = Mount(
    '.',
    '/root',
    lambda filename: os.path.splitext(filename)[-1] == '.py'
)


def _make_bytes(s):
    assert type(s) in (str, bytes)
    return s.encode("ascii") if type(s) is str else s


class Layer(Object):
    def __init__(
        self,
        tag=None,
        base_layers={},
        dockerfile_commands=[],
        context_files={},
        must_create=False,
    ):
        dockerfile_commands = [_make_bytes(s) for s in dockerfile_commands]

        # Construct the local id
        local_id_args = []
        for docker_tag, layer in base_layers.items():
            local_id_args.append("b:%s:(%s)" % (docker_tag, layer.args.local_id))
        local_id_args.append("c:%s" % get_sha256_hex_from_content(b"\n".join(dockerfile_commands)))
        for filename, content in context_files.items():
            local_id_args.append('f:%s:%s' % (filename, get_sha256_hex_from_content(content)))

        super().__init__(args=dict(
            local_id=','.join(local_id_args),
            tag=tag,
            base_layers=base_layers,
            dockerfile_commands=dockerfile_commands,
            context_files=context_files,
            must_create=must_create,
        ))

    async def join(self, client):
        # TODO: there's some risk of a race condition here
        if self.object_id is None:
            if self.args.tag:
                req = api_pb2.LayerGetByTagRequest(tag=self.args.tag)
                resp = await client.stub.LayerGetByTag(req)
                self.object_id = resp.layer_id

            else:
                # Recursively build base layers
                base_layer_ids = await asyncio.gather(
                    *(layer.join(client) for layer in self.args.base_layers.values())
                )
                base_layers = [
                    api_pb2.BaseLayer(
                        docker_tag=docker_tag,
                        layer_id=layer_id
                    )
                    for docker_tag, layer_id
                    in zip(self.args.base_layers.keys(), base_layer_ids)
                ]

                context_files = [
                    api_pb2.LayerContextFile(filename=filename, data=data)
                    for filename, data in self.args.context_files.items()
                ]

                layer_definition = api_pb2.Layer(
                    base_layers=base_layers,
                    dockerfile_commands=self.args.dockerfile_commands,
                    context_files=context_files,
                )

                req = api_pb2.LayerGetOrCreateRequest(
                    session_id=client.session_id,
                    layer=layer_definition,
                    must_create=self.args.must_create,
                )
                resp = await client.stub.LayerGetOrCreate(req)
                self.object_id = resp.layer_id

        logger.debug('Waiting for layer %s' % self.object_id)
        while True:
            request = api_pb2.LayerJoinRequest(
                layer_id=self.object_id,
                timeout=BLOCKING_REQUEST_TIMEOUT,
                session_id=client.session_id,
            )
            response = await retry(client.stub.LayerJoin)(request, timeout=GRPC_REQUEST_TIMEOUT)
            if not response.result.status:
                continue
            elif response.result.status == api_pb2.GenericResult.Status.FAILURE:
                raise Exception(response.result.exception)
            elif response.result.status == api_pb2.GenericResult.Status.SUCCESS:
                break
            else:
                raise Exception('Unknown status %s!' % response.result.status)

        return self.object_id

    async def set_tag(self, tag, client):
        assert self.object_id
        req = api_pb2.LayerSetTagRequest(layer_id=self.object_id, tag=tag)
        await client.stub.LayerSetTag(req)


class EnvDict(Object):
    def __init__(self, env_dict):
        super().__init__(
            args=dict(
                env_dict=env_dict,
            )
        )

    async def join(self, client):
        if not self.object_id:
            req = api_pb2.EnvDictCreateRequest(session_id=client.session_id, env_dict=self.args.env_dict)
            resp = await client.stub.EnvDictCreate(req)
            self.object_id = resp.env_dict_id

        return self.object_id


class Image(Object):
    def __init__(self, layer, mounts=[], env_dict=None, **kwargs):
        local_id = "i:(%s)" % layer.args.local_id  # TODO: include the mounts in the local id too!!!
        super().__init__(args=dict(
            layer=layer,
            mounts=mounts,
            env_dict=env_dict,
            local_id=local_id,
            **kwargs
        ))

    async def join(self, client):
        # TODO: there's some risk of a race condition here
        if self.object_id is None:
            coros = [self.layer.join(client)]
            if self.args.env_dict:
                coros.append(self.args.env_dict.join(client))
            for mount in self.args.mounts:
                coros.append(mount.join(client))

            results = await asyncio.gather(*coros)

            # mutating results for readability
            layer_id, results = results[0], results[1:]
            if self.args.env_dict:
                env_dict_id, results = results[0], results[1:]
            else:
                env_dict_id = None
            mount_ids = results[:]

            image = api_pb2.Image(
                layer_id=self.args.layer.layer_id,
                mount_ids=mount_ids,
                local_id=self.args.local_id,
                env_dict_id=env_dict_id,
            )

            request = api_pb2.ImageCreateRequest(
                session_id=client.session_id,
                image=image
            )
            response = await client.stub.ImageCreate(request)
            self.object_id = response.image_id

        return self.object_id

    def set_env_vars(self, env_vars: Dict[str, str]):
        return Image(self.args.layer, self.args.mounts, EnvDict(env_vars))

    def function(self, raw_f):
        """Primarily to be used as a decorator."""
        return decorate_function(raw_f, self)

    def is_inside(self):
        # This is used from inside of containers to know whether this container is active or not
        return os.getenv("POLYESTER_IMAGE_LOCAL_ID") == self.args.local_id


class DebianSlim(Image):
    def __init__(self, layer=None, python_version=None):
        if python_version is None:
            python_version = "%d.%d.%d" % sys.version_info[:3]
        if layer is None:
            layer = Layer(tag="python-%s-slim-buster-base" % python_version)
        super().__init__(
            layer=layer,
            mounts=[mount_py_in_workdir_into_root],
            python_version=python_version,
        )

    def add_python_packages(self, python_packages):
        layer = Layer(
            base_layers={
                "base": self.args.layer,
                "builder": Layer(tag="python-%s-slim-buster-builder" % self.args.python_version),
            },
            dockerfile_commands=[
                'FROM builder as builder-vehicle',
                'RUN pip wheel %s -w /tmp/wheels' % ' '.join(python_packages),
                'FROM base',
                'COPY --from=builder-vehicle /tmp/wheels /tmp/wheels',
                'RUN pip install /tmp/wheels/*',
                'RUN rm -rf /tmp/wheels',
            ]
        )
        return DebianSlim(layer=layer)

    def run_commands(self, commands):
        layer = Layer(
            base_layers={"base": self.args.layer},
            dockerfile_commands=["FROM base"] + ["RUN " + command for command in commands],
        )
        return DebianSlim(layer=layer)


debian_slim = DebianSlim()
base_image = debian_slim
