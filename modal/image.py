import asyncio
import os
import shlex
import sys
import warnings
from typing import Dict, List, Optional

from modal_proto import api_pb2
from modal_utils.async_utils import retry, synchronize_apis

from ._app_singleton import get_container_app
from .config import config, logger
from .exception import NotFoundError, RemoteError
from .object import Object, ref


def _make_bytes(s):
    assert type(s) in (str, bytes)
    return s.encode("utf-8") if type(s) is str else s


class _Image(Object, type_prefix="im"):
    """Base class for container images to run functions in.

    Do not construct this class directly; instead use
    `modal.image.DebianSlim` or
    `modal.image.extend_image`
    """

    def __init__(
        self,
        base_images={},
        context_files={},
        dockerfile_commands=[],
        secrets=[],
        version=None,
    ):
        self._base_images = base_images
        self._context_files = context_files
        self._dockerfile_commands = dockerfile_commands
        self._version = version
        self._secrets = secrets
        super().__init__()

    async def load(self, app, existing_image_id):
        # Recursively build base images
        base_image_ids = await asyncio.gather(*(app.create_object(image) for image in self._base_images.values()))
        base_images_pb2s = [
            api_pb2.BaseImage(docker_tag=docker_tag, image_id=image_id)
            for docker_tag, image_id in zip(self._base_images.keys(), base_image_ids)
        ]

        secret_ids = []
        for secret in self._secrets:
            try:
                secret_id = await app.create_object(secret)
            except NotFoundError as ex:
                raise NotFoundError(
                    f"Could not find secret {ex.obj_repr}\n"
                    + "You can add secrets to your account at https://modal.com/secrets",
                    ex.obj_repr,
                )
            secret_ids.append(secret_id)

        context_file_pb2s = [
            api_pb2.ImageContextFile(filename=filename, data=open(path, "rb").read())
            for filename, path in self._context_files.items()
        ]

        dockerfile_commands = [_make_bytes(s) for s in self._dockerfile_commands]
        image_definition = api_pb2.Image(
            base_images=base_images_pb2s,
            dockerfile_commands=dockerfile_commands,
            context_files=context_file_pb2s,
            version=self._version,
            secret_ids=secret_ids,
        )

        req = api_pb2.ImageGetOrCreateRequest(
            app_id=app.app_id,
            image=image_definition,
            existing_image_id=existing_image_id,  # TODO: ignored
        )
        resp = await app.client.stub.ImageGetOrCreate(req)
        image_id = resp.image_id

        logger.debug("Waiting for image %s" % image_id)
        while True:
            request = api_pb2.ImageJoinRequest(
                image_id=image_id,
                timeout=60,
                app_id=app.app_id,
            )
            response = await retry(app.client.stub.ImageJoin)(request)
            if not response.result.status:
                continue
            elif response.result.status == api_pb2.GenericResult.GENERIC_STATUS_FAILURE:
                raise RemoteError(response.result.exception)
            elif response.result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS:
                break
            else:
                raise RemoteError("Unknown status %s!" % response.result.status)

        return image_id

    def is_inside(self):
        """Returns whether this container is active or not.

        Useful for conditionally importing libraries when inside images.
        """
        # TODO(erikbern): This method only works if an image is assigned to an app
        # This is pretty confusing so let's figure out a way to clean it up.
        #
        # image = DebianSlim()
        # if image.is_inside():  # This WILL NOT work
        #
        # app["image"] = DebianSlim()
        # if app["image"].is_inside()  # This WILL work

        if get_container_app() is None:
            return False

        env_image_id = config.get("image_id")
        logger.debug(f"Is image inside? env {env_image_id} image {self.object_id}")
        return self.object_id == env_image_id


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
    }
    major_minor_version = ".".join(python_version.split(".")[:2])
    python_version = major_minor_version + "." + latest_micro_version[major_minor_version]
    return python_version


def _DebianSlim(
    app=None,
    extra_commands: List[str] = [],  # A list of shell commands executed while building the image
    python_packages: List[str] = [],  # A list of Python packages, eg. ["numpy", "matplotlib>=3.5.0"]
    python_version: Optional[str] = None,  # Set a specific Python version
    pip_find_links: Optional[str] = None,
    requirements_txt: Optional[str] = None,  # File contents of a requirements.txt
    context_files: Dict[
        str, bytes
    ] = {},  # A dict containing any files that will be present during the build to use with COPY
    secrets: List[
        Object
    ] = [],  # List of Modal secrets that will be available as environment variables during the build process
    version: Optional[str] = None,  # Custom string to break the image hashing and force the image to be rebuilt
):
    """A default base image, built on the official python:<version>-slim-bullseye Docker hub images

    Can also be called as a function to build a new image with additional bash
    commands or python packages.
    """
    if app is not None:
        warnings.warn("Passing `app` to the image constructor is deprecated", DeprecationWarning)

    python_version = _dockerhub_python_version(python_version)
    base_image = ref(f"debian-slim-{python_version}", namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL)

    dockerfile_commands = ["FROM base as target"]
    base_images = {"base": base_image}
    dockerfile_commands += [f"RUN {cmd}" for cmd in extra_commands]

    if requirements_txt is not None:
        context_files = context_files.copy()
        context_files["/.requirements.txt"] = requirements_txt

        dockerfile_commands += [
            "COPY /.requirements.txt /.requirements.txt",
            "RUN pip install -r /.requirements.txt",
        ]

    if python_packages:
        find_links_arg = f"-f {pip_find_links}" if pip_find_links else ""
        package_args = " ".join(shlex.quote(pkg) for pkg in python_packages)

        dockerfile_commands += [
            f"RUN pip install {package_args} {find_links_arg}",
        ]

    if len(dockerfile_commands) == 1:
        return base_image

    return _Image(
        dockerfile_commands=dockerfile_commands,
        context_files=context_files,
        base_images=base_images,
        version=version,
        secrets=secrets,
    )


def _extend_image(base_image, extra_dockerfile_commands, context_files={}, secrets=[]):
    """Extend an image with arbitrary dockerfile commands"""
    return _Image(
        base_images={"base": base_image},
        dockerfile_commands=["FROM base"] + extra_dockerfile_commands,
        context_files=context_files,
        secrets=secrets,
    )


def get_client_requirements_path():
    # Locate Modal client requirements.txt
    import modal

    modal_path = modal.__path__[0]
    return os.path.join(modal_path, "requirements.txt")


def _DockerhubImage(app=None, tag=None):
    """
    Build a modal image from a pre-existing image on DockerHub.

    This assumes the following about the image:
    - Python 3.7 or above is present, and is available as `python`
    - `pip` is installed correctly
    - The image is built for the `linux/amd64` platform
    """
    if app is not None:
        warnings.warn("Passing `app` to the image constructor is deprecated", DeprecationWarning)

    requirements_path = get_client_requirements_path()

    dockerfile_commands = [
        f"FROM {tag}",
        "COPY /modal_requirements.txt /modal_requirements.txt",
        "RUN pip install --upgrade pip",
        "RUN pip install -r /modal_requirements.txt",
    ]

    return _Image(
        dockerfile_commands=dockerfile_commands,
        context_files={"/modal_requirements.txt": requirements_path},
    )


Image, AioImage = synchronize_apis(_Image)
DebianSlim, AioDebianSlim = synchronize_apis(_DebianSlim)
DockerhubImage, AioDockerhubImage = synchronize_apis(_DockerhubImage)
extend_image, aio_extend_image = synchronize_apis(_extend_image)
