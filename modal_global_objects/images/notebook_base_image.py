# Copyright Modal Labs 2025
import shlex
from typing import Optional

from modal import Image
from modal.image import DockerfileSpec, ImageBuilderVersion
from modal_proto import api_pb2


def notebook_base_image(*, python_version: Optional[str] = None, force_build: bool = False) -> Image:
    """Private API. Default image used for Modal notebook kernels."""
    # Include several common packages, as well as kernelshim dependencies (except 'modal').
    # These packages aren't pinned right now. Notebooks are not yet stable.
    base_image = Image.debian_slim(python_version=python_version)

    # TODO: Compile a better list, this is just a quick MVP.
    # https://pypistats.org/top
    # https://github.com/vinta/awesome-python
    # https://github.com/Paperspace/jupyter-docker-stacks/blob/master/scipy-notebook/Dockerfile
    # https://github.com/modal-labs/modal-examples
    environment_packages = [
        "accelerate",
        "aiohttp",
        "altair",
        "anthropic",
        "asyncpg",
        "beautifulsoup4",
        "bokeh",
        "boto3[crt]",
        "click",
        "diffusers[torch,flax]",
        "dm-sonnet",
        "flax",
        "ftfy",
        "h5py",
        "urllib3",
        "httpx",
        "huggingface-hub",
        "ipywidgets",
        "jax[cuda12]",
        "keras",
        "matplotlib",
        "numba",
        "numpy",
        "openai",
        "optax",
        "pandas",
        "plotly[express]",
        "polars",
        "psycopg2",
        "requests",
        "safetensors",
        "scikit-image",
        "scikit-learn",
        "scipy",
        "seaborn",
        "sentencepiece",
        "sqlalchemy",
        "statsmodels",
        "sympy",
        "tabulate",
        "tensorboard",
        "toml",
        "transformers",
        "triton",
        "typer",
        "vega-datasets",
        "watchfiles",
        "websockets",
    ]

    # Kernelshim dependencies.
    kernelshim_packages = [
        "basedpyright>=1.28",
        "fastapi>=0.100",
        "ipykernel>=6",
        "pydantic>=2",
        "pyzmq>=26",
        "ruff>=0.11",
        "uvicorn>=0.32",
    ]

    commands: list[str] = [
        "apt-get update",
        "apt-get install -y libpq-dev pkg-config cmake",
        # Install uv since it's faster than pip for installing packages.
        "pip install uv",
        # https://github.com/astral-sh/uv/issues/11480
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126",
        f"uv pip install --system {shlex.join(sorted(environment_packages))}",
        f"uv pip install --system {shlex.join(sorted(kernelshim_packages))}",
    ]

    def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
        return DockerfileSpec(commands=["FROM base", *(f"RUN {cmd}" for cmd in commands)], context_files={})

    return Image._from_args(
        base_images={"base": base_image},
        dockerfile_function=build_dockerfile,
        force_build=force_build,
        _namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL,
    )
