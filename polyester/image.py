import asyncio
import os
import sys
from typing import Dict

from .async_utils import retry
from .config import config, logger
from .exception import RemoteError
from .grpc_utils import BLOCKING_REQUEST_TIMEOUT, GRPC_REQUEST_TIMEOUT
from .mount import get_sha256_hex_from_content  # TODO: maybe not
from .object import Object, requires_create
from .proto import api_pb2


def _make_bytes(s):
    assert type(s) in (str, bytes)
    return s.encode("ascii") if type(s) is str else s


def get_python_version():
    return config["image_python_version"] or "%d.%d.%d" % sys.version_info[:3]


async def _build_custom_image(
    session, local_id, base_images={}, context_files={}, dockerfile_commands=[], must_create=False
):
    # Recursively build base images
    base_image_ids = await asyncio.gather(*(session.create_object(image) for image in base_images.values()))
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
    resp = await session.client.stub.ImageGetOrCreate(req)
    image_id = resp.image_id

    logger.debug("Waiting for image %s" % image_id)
    while True:
        request = api_pb2.ImageJoinRequest(
            image_id=image_id,
            timeout=BLOCKING_REQUEST_TIMEOUT,
            session_id=session.session_id,
        )
        response = await retry(session.client.stub.ImageJoin)(request, timeout=GRPC_REQUEST_TIMEOUT)
        if not response.result.status:
            continue
        elif response.result.status == api_pb2.GenericResult.Status.FAILURE:
            raise RemoteError(response.result.exception)
        elif response.result.status == api_pb2.GenericResult.Status.SUCCESS:
            break
        else:
            raise RemoteError("Unknown status %s!" % response.result.status)

    return image_id


class Image(Object):
    def __init__(self, session, tag):
        if tag is None:
            raise Exception("Every image needs a local_id")
        super().__init__(tag=tag, session=session)

    def extend(self, arg):
        return ExtendedImage(self._session, self, arg)

    def is_inside(self):
        # This is used from inside of containers to know whether this container is active or not
        env_local_id = os.getenv("POLYESTER_IMAGE_LOCAL_ID")
        logger.info(f"Is image inside? env {env_local_id} image {self.tag}")
        return env_local_id == self.tag


class LocalImage(Image):
    def __init__(self, session, python_executable):
        super().__init__(tag="local", session=session)
        self.python_executable = python_executable

    async def _create_impl(self, session):
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
    # TODO: every time you extend this image, it registers a new image to be created
    # Eg if you extend it n time then you will create n images that will be built individually
    # The solution is either to
    # (a) chain all images so they are based on the previous
    # (b) have images without sessions that aren't built
    def __init__(self, session=None, python_version=None, build_instructions=[]):
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

    def add_python_packages(self, python_packages, find_links=None):
        return DebianSlim(
            self._session, self.python_version, self.build_instructions + [("py", (python_packages, find_links))]
        )

    def run_commands(self, commands):
        return DebianSlim(self._session, self.python_version, self.build_instructions + [("cmd", commands)])

    def copy_from_image(self, image, src, dest):
        return DebianSlim(self._session, self.python_version, self.build_instructions + [("cp", (image, src, dest))])

    async def _create_impl(self, session):
        base_images = {
            "builder": Image.use(session, f"python-{self.python_version}-slim-buster-builder"),
            "base": Image.use(session, f"python-{self.python_version}-slim-buster-base"),
        }
        if not self.build_instructions:
            return await session.create_object(base_images["base"])

        dockerfile_commands = ["FROM base as target"]
        for t, data in self.build_instructions:
            if t == "py":
                (packages, find_links) = data

                find_links_arg = f"-f {find_links}" if find_links else ""

                dockerfile_commands += [
                    "FROM builder as builder-vehicle",
                    f"RUN pip wheel {' '.join(packages)} -w /tmp/wheels {find_links_arg}",
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
            session,
            self.tag,
            dockerfile_commands=dockerfile_commands,
            base_images=base_images,
        )


class ExtendedImage(Image):
    def __init__(self, session, base, arg):
        if callable(arg):
            tag = arg.__name__
        else:
            tag = get_sha256_hex_from_content(base.tag.encode("ascii") + b"/" + repr(arg).encode("ascii"))
        self.base = base
        self.arg = arg
        super().__init__(session=session, tag=tag)

    async def _create_impl(self, session):
        build_instructions = ["FROM base"]
        if callable(self.arg):
            build_instructions += self.arg()
        else:
            build_instructions += self.arg
        return await _build_custom_image(
            session,
            self.tag,
            dockerfile_commands=build_instructions,
            base_images={"base": self.base},
        )
