from setuptools import setup

from modal._version import __version__

# This needs to be set from setup.py, rather than setup.cfg, because the `attr:`
# syntax for accessing attributes in setup.cfg does not work.

setup(version=__version__)
