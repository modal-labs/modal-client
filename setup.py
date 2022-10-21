# Copyright Modal Labs 2022
import os
import sys

# Add current directory to the path
sys.path.append(os.getcwd())

# PEP 396 prescribes setting a __version__ on the package,
# but there's no obvious way for setup.py to access it.
# A workaround is to put it in a separate package with no dependencies on modal.
# TODO: might want to look at pbr:
# https://docs.openstack.org/pbr/latest/user/index.html

from setuptools import setup  # noqa

from modal_version import __version__  # noqa

setup(version=__version__)
