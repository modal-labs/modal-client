import asyncio
import functools
import os
import sys
from typing import Dict

from ._async_utils import retry
from ._grpc_utils import BLOCKING_REQUEST_TIMEOUT, GRPC_REQUEST_TIMEOUT
from .config import config, logger
from .exception import RemoteError
from .object import Object, requires_create
from .proto import api_pb2


def _make_bytes(s):
    assert type(s) in (str, bytes)
    return s.encode("ascii") if type(s) is str else s


class Image(Object):
    """Base class for container images to run functions in.

    Do not construct this class directly; instead use
    :py:func:`modal.image.debian_slim` or
    :py:func:`modal.image.extend_image`.
    """

    def __init__(self, session):
        super().__init__(session=session)

    def is_inside(self):
        """Returns whether this container is active or not.

        Useful for conditionally importing libraries when inside images.
        """
        # This is used from inside of containers to know whether this container is active or not
        env_image_id = config.get("image_id")
        image_id = self.object_id
        logger.debug(f"Is image inside? env {env_image_id} image {image_id}")
        return image_id is not None and env_image_id == image_id

    def _should_default_register(self):
        return False


class CustomImage(Image):
    """A custom image built using docker commands.

    Generally, you should instead use :py:func:`modal.image.extend_image`
    """

    """This might be a temporary thing until we can simplify other code.

    Needed to rewrite all the other subclasses to use composition instead of inheritance."""

    def __init__(
        self,
        base_images={},
        context_files={},
        dockerfile_commands=[],
        local_image_python_executable=None,
        version=None,
    ):
        self._base_images = base_images
        self._context_files = context_files
        self._dockerfile_commands = dockerfile_commands
        self._local_image_python_executable = local_image_python_executable
        self._version = version
        # Note that these objects have neither sessions nor tags
        # They rely on the factories for this
        super().__init__(session=None)

    async def _create_impl(self, session):
        # Recursively build base images
        base_image_ids = await asyncio.gather(*(session.create_object(image) for image in self._base_images.values()))
        base_images_pb2s = [
            api_pb2.BaseImage(docker_tag=docker_tag, image_id=image_id)
            for docker_tag, image_id in zip(self._base_images.keys(), base_image_ids)
        ]

        context_file_pb2s = [
            api_pb2.ImageContextFile(filename=filename, data=data) for filename, data in self._context_files.items()
        ]

        dockerfile_commands = [_make_bytes(s) for s in self._dockerfile_commands]
        image_definition = api_pb2.Image(
            base_images=base_images_pb2s,
            dockerfile_commands=dockerfile_commands,
            context_files=context_file_pb2s,
            local_image_python_executable=self._local_image_python_executable,
            version=self._version,
        )

        req = api_pb2.ImageGetOrCreateRequest(
            session_id=session.session_id,
            image=image_definition,
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


@Image.factory
def local_image(python_executable):
    """Only used for various integration tests."""
    return CustomImage(local_image_python_executable=python_executable)


def _dockerhub_python_version(python_version=None):
    if python_version is None:
        python_version = config["image_python_version"]
    if python_version is None:
        python_version = "%d.%d" % sys.version_info[:2]

    # We use the same major/minor version, but the highest micro version
    # See https://hub.docker.com/_/python
    latest_micro_version = {
        "3.10": "1",
        "3.9": "9",
        "3.8": "12",
        "3.7": "12",
        "3.6": "15",
    }
    major_minor_version = ".".join(python_version.split(".")[:2])
    python_version = major_minor_version + "." + latest_micro_version[major_minor_version]
    return python_version


@Image.factory
def debian_slim(extra_commands=None, python_packages=None, python_version=None):
    """A default base image, built on the official python:<version>-slim-buster Docker hub images

    Can also be called as a function to build a new image with additional bash
    commands or python packages.
    """
    python_version = _dockerhub_python_version(python_version)
    base_image = Image.use(None, f"python-{python_version}-slim-buster-base", api_pb2.ShareNamespace.GLOBAL)
    builder_image = Image.use(None, f"python-{python_version}-slim-buster-builder", api_pb2.ShareNamespace.GLOBAL)

    if extra_commands is None and python_packages is None:
        return base_image

    dockerfile_commands = ["FROM base as target"]
    base_images = {"base": base_image}
    if extra_commands is not None:
        dockerfile_commands += [f"RUN {cmd}" for cmd in extra_commands]

    if python_packages is not None:
        base_images["builder"] = builder_image
        dockerfile_commands += [
            "FROM builder as builder-vehicle",
            f"RUN pip wheel {' '.join(python_packages)} -w /tmp/wheels",
            "FROM target",
            "COPY --from=builder-vehicle /tmp/wheels /tmp/wheels",
            "RUN pip install /tmp/wheels/*",
            "RUN rm -rf /tmp/wheels",
        ]

    return CustomImage(
        dockerfile_commands=dockerfile_commands,
        base_images=base_images,
    )


def extend_image(base_image, extra_dockerfile_commands):
    """Extend an image with arbitrary dockerfile commands"""
    return CustomImage(base_images={"base": base_image}, dockerfile_commands=["FROM base"] + extra_dockerfile_commands)
