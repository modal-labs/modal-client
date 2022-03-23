import asyncio
import os
import sys

from modal_proto import api_pb2
from modal_utils.async_utils import retry, synchronize_apis
from modal_utils.grpc_utils import BLOCKING_REQUEST_TIMEOUT, GRPC_REQUEST_TIMEOUT

from ._factory import _factory
from .config import config, logger
from .exception import RemoteError
from .object import Object


def _make_bytes(s):
    assert type(s) in (str, bytes)
    return s.encode("ascii") if type(s) is str else s


class _Image(Object, type_prefix="im"):
    """Base class for container images to run functions in.

    Do not construct this class directly; instead use
    :py:func:`modal.image.debian_slim` or
    :py:func:`modal.image.extend_image`.
    """

    def _init(self):
        pass

    def is_inside(self):
        """Returns whether this container is active or not.

        Useful for conditionally importing libraries when inside images.
        """
        # This is used from inside of containers to know whether this container is active or not
        env_image_id = config.get("image_id")
        image_id = self.object_id
        logger.debug(f"Is image inside? env {env_image_id} image {image_id}")
        return image_id is not None and env_image_id == image_id


class _CustomImage(_Image):
    """A custom image built using docker commands.

    Generally, you should instead use :py:func:`modal.image.extend_image`
    """

    """This might be a temporary thing until we can simplify other code.

    Needed to rewrite all the other subclasses to use composition instead of inheritance."""

    @classmethod
    async def create(
        cls,
        base_images={},
        context_files={},
        dockerfile_commands=[],
        local_image_python_executable=None,
        version=None,
        app=None,
    ):
        app = cls._get_app(app)

        # Recursively build base images
        base_image_ids = await asyncio.gather(*(app.create_object(image) for image in base_images.values()))
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
            local_image_python_executable=local_image_python_executable,
            version=version,
        )

        req = api_pb2.ImageGetOrCreateRequest(
            app_id=app.app_id,
            image=image_definition,
        )
        resp = await app.client.stub.ImageGetOrCreate(req)
        image_id = resp.image_id

        logger.debug("Waiting for image %s" % image_id)
        while True:
            request = api_pb2.ImageJoinRequest(
                image_id=image_id,
                timeout=BLOCKING_REQUEST_TIMEOUT,
                app_id=app.app_id,
            )
            response = await retry(app.client.stub.ImageJoin)(request, timeout=GRPC_REQUEST_TIMEOUT)
            if not response.result.status:
                continue
            elif response.result.status == api_pb2.GenericResult.GENERIC_STATUS_FAILURE:
                raise RemoteError(response.result.exception)
            elif response.result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS:
                break
            else:
                raise RemoteError("Unknown status %s!" % response.result.status)

        return cls._create_object_instance(image_id, app)


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


@_factory(_Image)
async def _DebianSlim(
    extra_commands=None,
    python_packages=None,
    python_version=None,
    pip_find_links=None,
    version=None,
):
    """A default base image, built on the official python:<version>-slim-buster Docker hub images

    Can also be called as a function to build a new image with additional bash
    commands or python packages.
    """
    python_version = _dockerhub_python_version(python_version)
    base_image = _Image.include(f"debian-slim-{python_version}", namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL)

    if extra_commands is None and python_packages is None:
        return base_image

    dockerfile_commands = ["FROM base as target"]
    base_images = {"base": base_image}
    if extra_commands is not None:
        dockerfile_commands += [f"RUN {cmd}" for cmd in extra_commands]

    if python_packages is not None:
        find_links_arg = f"-f {pip_find_links}" if pip_find_links else ""

        dockerfile_commands += [
            f"RUN pip install {' '.join(python_packages)} {find_links_arg}",
        ]

    return await _CustomImage.create(
        dockerfile_commands=dockerfile_commands,
        base_images=base_images,
        version=version,
    )


async def _extend_image(base_image, extra_dockerfile_commands):
    """Extend an image with arbitrary dockerfile commands"""
    return await _CustomImage.create(
        base_images={"base": base_image}, dockerfile_commands=["FROM base"] + extra_dockerfile_commands
    )


def get_client_requirements():
    # Locate Modal client requirements.txt
    import modal

    modal_path = modal.__path__[0]
    requirements_fn = os.path.join(modal_path, "requirements.txt")

    requirements_data = open(requirements_fn, "rb").read()

    return requirements_fn, requirements_data


@_factory(_Image)
async def _DockerhubImage(tag):
    """
    Build a modal image from a pre-existing image on DockerHub.

    This assumes the following about the image:
    - Python 3.6 or above is present, and is available as `python`
    - `pip` is installed correctly
    - The image is built for the `linux/amd64` platform
    """

    requirements_fn, requirements_data = get_client_requirements()

    dockerfile_commands = [
        f"FROM {tag}",
        f"COPY {requirements_fn} {requirements_fn}",
        "RUN pip install --upgrade pip",
        f"RUN pip install -r {requirements_fn}",
    ]

    return await _CustomImage.create(
        dockerfile_commands=dockerfile_commands,
        context_files={requirements_fn: requirements_data},
    )


Image, AioImage = synchronize_apis(_Image)
CustomImage, AioCustomImage = synchronize_apis(_CustomImage)
DebianSlim, AioDebianSlim = synchronize_apis(_DebianSlim)
DockerhubImage, AioDockerhubImage = synchronize_apis(_DockerhubImage)
extend_image, aio_extend_image = synchronize_apis(_extend_image)
