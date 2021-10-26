import asyncio
import os
import sys
from typing import Dict

from .async_utils import retry
from .config import config, logger
from .exception import RemoteException
from .grpc_utils import BLOCKING_REQUEST_TIMEOUT, GRPC_REQUEST_TIMEOUT
from .mount import get_sha256_hex_from_content  # TODO: maybe not
from .object import Object, requires_create
from .proto import api_pb2


def _make_bytes(s):
    assert type(s) in (str, bytes)
    return s.encode("ascii") if type(s) is str else s


class Image(Object):
    def __init__(
        self,
        tag=None,
        base_images={},
        dockerfile_commands=[],
        context_files={},
        must_create=False,
        local_id=None,
        local_image_python_executable=None,
    ):
        """
        An image can be created either as a reference to an existing image, or as a new image
        The `tag` parameter refers to an existing image (built elsewhere)
        The `local_id` is a property used to identify images within sessions (but across processes)

        This is a bit confusing right now, but I hope to address it as a part of refactoring persistence.
        """

        if local_id is None and local_image_python_executable is False:
            raise Exception("Every image needs a local_id")

        dockerfile_commands = [_make_bytes(s) for s in dockerfile_commands]

        super().__init__(
            args=dict(
                local_id=local_id,
                tag=tag,
                base_images=base_images,
                dockerfile_commands=dockerfile_commands,
                context_files=context_files,
                must_create=must_create,
                local_image_python_executable=local_image_python_executable,
            )
        )

    async def create_or_get(self):
        if self.args.tag:
            # Just fetch the image id from some existing image
            req = api_pb2.ImageGetByTagRequest(tag=self.args.tag)
            resp = await self.client.stub.ImageGetByTag(req)
            image_id = resp.image_id

        else:
            # Recursively build base images
            base_image_objs = await asyncio.gather(
                *(self.session.create_or_get_object(image) for image in self.args.base_images.values())
            )
            base_images_pb2s = [
                api_pb2.BaseImage(docker_tag=docker_tag, image_id=image.object_id)
                for docker_tag, image in zip(self.args.base_images.keys(), base_image_objs)
            ]

            context_file_pb2s = [
                api_pb2.ImageContextFile(filename=filename, data=data)
                for filename, data in self.args.context_files.items()
            ]

            image_definition = api_pb2.Image(
                base_images=base_images_pb2s,
                dockerfile_commands=self.args.dockerfile_commands,
                context_files=context_file_pb2s,
                local_id=self.args.local_id,
                local_image_python_executable=self.args.local_image_python_executable,
            )

            req = api_pb2.ImageGetOrCreateRequest(
                session_id=self.session.session_id,
                image=image_definition,
                must_create=self.args.must_create,
            )
            resp = await self.client.stub.ImageGetOrCreate(req)
            image_id = resp.image_id

        logger.debug("Waiting for image %s" % image_id)
        while True:
            request = api_pb2.ImageJoinRequest(
                image_id=image_id,
                timeout=BLOCKING_REQUEST_TIMEOUT,
                session_id=self.session.session_id,
            )
            response = await retry(self.client.stub.ImageJoin)(request, timeout=GRPC_REQUEST_TIMEOUT)
            if not response.result.status:
                continue
            elif response.result.status == api_pb2.GenericResult.Status.FAILURE:
                raise RemoteException(response.result.exception)
            elif response.result.status == api_pb2.GenericResult.Status.SUCCESS:
                break
            else:
                raise RemoteException("Unknown status %s!" % response.result.status)

        return image_id

    @requires_create
    async def set_tag(self, tag):
        req = api_pb2.ImageSetTagRequest(image_id=self.object_id, tag=tag)
        await self.client.stub.ImageSetTag(req)

    def is_inside(self):
        # This is used from inside of containers to know whether this container is active or not
        return os.getenv("POLYESTER_IMAGE_LOCAL_ID") == self.args.local_id


class EnvDict(Object):
    def __init__(self, env_dict):
        super().__init__(
            args=dict(
                env_dict=env_dict,
            )
        )

    async def create_or_get(self):
        req = api_pb2.EnvDictCreateRequest(session_id=self.session.session_id, env_dict=self.args.env_dict)
        resp = await self.client.stub.EnvDictCreate(req)
        return resp.env_dict_id


def get_python_version():
    return config["image_python_version"] or "%d.%d.%d" % sys.version_info[:3]


class DebianSlim(Image):
    def __init__(self, python_version=None):
        if python_version is None:
            python_version = get_python_version()
        tag = "python-%s-slim-buster-base" % python_version
        self.python_version = python_version
        super().__init__(tag=tag, local_id=tag)

    def add_python_packages(self, python_packages, find_links=None):
        find_links_arg = f"-f {find_links}" if find_links else ""
        h = get_sha256_hex_from_content(b",".join(p.encode("ascii") for p in python_packages))
        new_local_id = self.args.local_id + "/" + h
        builder_tagged = Image(
            local_id="python-%s-slim-buster-builder" % self.python_version,
            tag="python-%s-slim-buster-builder" % self.python_version
        )
        image = Image(
            local_id=new_local_id,
            base_images={
                "base": self,
                "builder": builder_tagged,
            },
            dockerfile_commands=[
                "FROM builder as builder-vehicle",
                f"RUN pip wheel {' '.join(python_packages)} -w /tmp/wheels {find_links_arg}",
                "FROM base",
                "COPY --from=builder-vehicle /tmp/wheels /tmp/wheels",
                "RUN pip install /tmp/wheels/*",
                "RUN rm -rf /tmp/wheels",
            ],
        )
        return image

    def run_commands(self, commands):
        h = get_sha256_hex_from_content(b",".join(c.encode("ascii") for c in commands))
        new_local_id = self.args.local_id + "/" + h
        image = Image(
            local_id=new_local_id,
            base_images={"base": self},
            dockerfile_commands=["FROM base"] + ["RUN " + command for command in commands],
        )
        return image


debian_slim = DebianSlim()
base_image = debian_slim
