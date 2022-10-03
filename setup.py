from modal_version import __version__
from setuptools import setup

# PEP 396 prescribes setting a __version__ on the package,
# but there's no obvious way for setup.py to access it.
# A workaround is to put it in a separate package with no dependencies on modal.
# TODO: might want to look at pbr:
# https://docs.openstack.org/pbr/latest/user/index.html

setup(version=__version__)
