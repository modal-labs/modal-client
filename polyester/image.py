import asyncio
import os
import sys
from typing import Dict

from .async_utils import retry
from .config import config, logger
from .function import decorate_function
from .grpc_utils import BLOCKING_REQUEST_TIMEOUT, GRPC_REQUEST_TIMEOUT
from .mount import get_sha256_hex_from_content  # TODO: maybe not
from .object import Object, requires_create
from .proto import api_pb2


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
            local_id_args.append("f:%s:%s" % (filename, get_sha256_hex_from_content(content)))

        super().__init__(
            args=dict(
                local_id=",".join(local_id_args),
                tag=tag,
                base_layers=base_layers,
                dockerfile_commands=dockerfile_commands,
                context_files=context_files,
                must_create=must_create,
            )
        )

    async def _create_or_get(self):
        if self.args.tag:
            req = api_pb2.LayerGetByTagRequest(tag=self.args.tag)
            resp = await self.client.stub.LayerGetByTag(req)
            layer_id = resp.layer_id

        else:
            # Recursively build base layers
            base_layer_objs = await asyncio.gather(
                *(self.session.create_or_get(layer) for layer in self.args.base_layers.values())
            )
            base_layers_pb2s = [
                api_pb2.BaseLayer(docker_tag=docker_tag, layer_id=layer.object_id)
                for docker_tag, layer in zip(self.args.base_layers.keys(), base_layer_objs)
            ]

            context_file_pb2s = [
                api_pb2.LayerContextFile(filename=filename, data=data)
                for filename, data in self.args.context_files.items()
            ]

            layer_definition = api_pb2.Layer(
                base_layers=base_layers_pb2s,
                dockerfile_commands=self.args.dockerfile_commands,
                context_files=context_file_pb2s,
            )

            req = api_pb2.LayerGetOrCreateRequest(
                session_id=self.session.session_id,
                layer=layer_definition,
                must_create=self.args.must_create,
            )
            resp = await self.client.stub.LayerGetOrCreate(req)
            layer_id = resp.layer_id

        logger.debug("Waiting for layer %s" % layer_id)
        while True:
            request = api_pb2.LayerJoinRequest(
                layer_id=layer_id,
                timeout=BLOCKING_REQUEST_TIMEOUT,
                session_id=self.session.session_id,
            )
            response = await retry(self.client.stub.LayerJoin)(request, timeout=GRPC_REQUEST_TIMEOUT)
            if not response.result.status:
                continue
            elif response.result.status == api_pb2.GenericResult.Status.FAILURE:
                raise Exception(response.result.exception)
            elif response.result.status == api_pb2.GenericResult.Status.SUCCESS:
                break
            else:
                raise Exception("Unknown status %s!" % response.result.status)

        return layer_id

    @requires_create
    async def set_tag(self, tag):
        req = api_pb2.LayerSetTagRequest(layer_id=self.object_id, tag=tag)
        await self.client.stub.LayerSetTag(req)


class EnvDict(Object):
    def __init__(self, env_dict):
        super().__init__(
            args=dict(
                env_dict=env_dict,
            )
        )

    async def _create_or_get(self):
        req = api_pb2.EnvDictCreateRequest(session_id=self.session.session_id, env_dict=self.args.env_dict)
        resp = await self.client.stub.EnvDictCreate(req)
        return resp.env_dict_id


class Image(Object):
    def __init__(self, layer=None, env_dict=None, local=False, **kwargs):
        if local:
            local_id = "local_image"
        else:
            local_id = "i:(%s)" % layer.args.local_id
        super().__init__(args=dict(layer=layer, env_dict=env_dict, local_id=local_id, local=local, **kwargs))

    async def _create_or_get(self):
        if self.args.env_dict:
            env_dict_id = await self.session.create_or_get(args.env_dict)
        else:
            env_dict_id = None

        if self.args.layer:
            layer = await self.session.create_or_get(self.args.layer)
            layer_id = layer.object_id
        else:
            layer_id = None

        image = api_pb2.Image(
            layer_id=layer_id,
            local_id=self.args.local_id,
            env_dict_id=env_dict_id,
            local=self.args.local,
        )

        request = api_pb2.ImageCreateRequest(session_id=self.session.session_id, image=image)
        response = await self.client.stub.ImageCreate(request)
        return response.image_id

    def set_env_vars(self, env_vars: Dict[str, str]):
        return Image(self.args.layer, EnvDict(env_vars))

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
            python_version=python_version,
        )

    def add_python_packages(self, python_packages):
        layer = Layer(
            base_layers={
                "base": self.args.layer,
                "builder": Layer(tag="python-%s-slim-buster-builder" % self.args.python_version),
            },
            dockerfile_commands=[
                "FROM builder as builder-vehicle",
                "RUN pip wheel %s -w /tmp/wheels" % " ".join(python_packages),
                "FROM base",
                "COPY --from=builder-vehicle /tmp/wheels /tmp/wheels",
                "RUN pip install /tmp/wheels/*",
                "RUN rm -rf /tmp/wheels",
            ],
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
