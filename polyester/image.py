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


class EnvDict(Object):
    def __init__(self, env_dict):
        super().__init__(tag=None)
        self.env_dict = env_dict

    async def create_or_get(self, session):
        req = api_pb2.EnvDictCreateRequest(session_id=session.session_id, env_dict=self.env_dict)
        resp = await session.client.stub.EnvDictCreate(req)
        return resp.env_dict_id


def get_python_version():
    return config["image_python_version"] or "%d.%d.%d" % sys.version_info[:3]


async def _build_custom_image(
    client, session, local_id, base_images={}, context_files={}, dockerfile_commands=[], must_create=False
):
    # Recursively build base images
    base_image_ids = await asyncio.gather(*(session.create_or_get_object(image) for image in base_images.values()))
    base_images_pb2s = [
        api_pb2.BaseImage(docker_tag=docker_tag, image_id=image_id)
        for docker_tag, image_id in zip(base_images.keys(), base_image_ids)
    ]

    context_file_pb2s = [
        api_pb2.ImageContextFile(filename=filename, data=data) for filename, data in context_files.items()
    ]

    dockerfile_commands = [_make_bytes(s) for s in dockerfile_commands]
    image_definition = api_pb2.Image(
        base_images=base_images_pb2s,
        dockerfile_commands=dockerfile_commands,
        context_files=context_file_pb2s,
        local_id=local_id,
    )

    req = api_pb2.ImageGetOrCreateRequest(
        session_id=session.session_id,
        image=image_definition,
        must_create=must_create,
    )
    resp = await client.stub.ImageGetOrCreate(req)
    image_id = resp.image_id

    logger.debug("Waiting for image %s" % image_id)
    while True:
        request = api_pb2.ImageJoinRequest(
            image_id=image_id,
            timeout=BLOCKING_REQUEST_TIMEOUT,
            session_id=session.session_id,
        )
        response = await retry(client.stub.ImageJoin)(request, timeout=GRPC_REQUEST_TIMEOUT)
        if not response.result.status:
            continue
        elif response.result.status == api_pb2.GenericResult.Status.FAILURE:
            raise RemoteException(response.result.exception)
        elif response.result.status == api_pb2.GenericResult.Status.SUCCESS:
            break
        else:
            raise RemoteException("Unknown status %s!" % response.result.status)

    return image_id


class Image(Object):
    def __init__(self, tag, session=None):
        if tag is None:
            raise Exception("Every image needs a local_id")
        super().__init__(tag=tag, session=session)

    @requires_create
    async def set_image_tag(self, image_tag):
        req = api_pb2.ImageSetTagRequest(image_id=self.object_id, tag=image_tag)
        await self.session.client.stub.ImageSetTag(req)

    def is_inside(self):
        # This is used from inside of containers to know whether this container is active or not
        env_local_id = os.getenv("POLYESTER_IMAGE_LOCAL_ID")
        logger.info(f"Is image inside? env {env_local_id} image {self.tag}")
        return env_local_id == self.tag


class TaggedImage(Image):
    def __init__(self, existing_image_tag, session=None):
        super().__init__(tag=existing_image_tag, session=session)
        self.existing_image_tag = existing_image_tag

    async def create_or_get(self, session):
        req = api_pb2.ImageGetByTagRequest(tag=self.existing_image_tag)
        resp = await session.client.stub.ImageGetByTag(req)
        image_id = resp.image_id
        return image_id


class LocalImage(Image):
    def __init__(self, python_executable, session=None):
        super().__init__(tag="local", session=session)
        self.python_executable = python_executable

    async def create_or_get(self, session):
        image_definition = api_pb2.Image(
            local_id=self.tag,  # rename local_id
            local_image_python_executable=self.python_executable,
        )
        req = api_pb2.ImageGetOrCreateRequest(
            session_id=session.session_id,
            image=image_definition,
        )
        resp = await session.client.stub.ImageGetOrCreate(req)
        return resp.image_id


class DebianSlim(Image):
    def __init__(self, python_version=None, build_instructions=[], session=None):
        if python_version is None:
            python_version = get_python_version()
        else:
            # We need to make sure that the version *inside* the image matches the version *outside*
            # This is important or else image.is_inside() won't work
            numbers = [int(z) for z in python_version.split(".")]
            assert len(numbers) == 3

        self.python_version = python_version
        self.build_instructions = build_instructions
        h = get_sha256_hex_from_content(repr(build_instructions).encode("ascii"))
        tag = f"debian-slim-{python_version}-{h}"
        super().__init__(tag=tag, session=session)

    def add_python_packages(self, python_packages):
        return DebianSlim(self.python_version, self.build_instructions + [("py", python_packages)], session=self.session)

    def run_commands(self, commands):
        return DebianSlim(self.python_version, self.build_instructions + [("cmd", commands)], session=self.session)

    def copy_from_image(self, image, src, dest):
        return DebianSlim(self.python_version, self.build_instructions + [("cp", (image, src, dest))], session=self.session)

    async def create_or_get(self, session):
        base_images = {
            "builder": TaggedImage(f"python-{self.python_version}-slim-buster-builder"),
            "base": TaggedImage(f"python-{self.python_version}-slim-buster-base"),
        }
        if not self.build_instructions:
            return await session.create_or_get_object(base_images["base"])

        dockerfile_commands = ["FROM base as target"]
        for t, data in self.build_instructions:
            if t == "py":
                dockerfile_commands += [
                    "FROM builder as builder-vehicle",
                    f"RUN pip wheel {' '.join(data)} -w /tmp/wheels",  #  {find_links_arg}
                    "FROM target",
                    "COPY --from=builder-vehicle /tmp/wheels /tmp/wheels",
                    "RUN pip install /tmp/wheels/*",
                    "RUN rm -rf /tmp/wheels",
                ]
            elif t == "cmd":
                dockerfile_commands += [f"RUN {cmd}" for cmd in data]
            elif t == "cp":
                image, src, dest = data
                dockerfile_commands += [f"COPY --from={image} {src} {dest}"]

        return await _build_custom_image(
            session.client,
            session,
            self.tag,
            dockerfile_commands=dockerfile_commands,
            base_images=base_images,
        )


debian_slim = DebianSlim()
base_image = debian_slim
