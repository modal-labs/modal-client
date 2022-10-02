# This file is executed by setup.py
#
# This is used by:
# * setup.py: defines a package version (will later affect PyPI)
# * client connection: it sends the version to the server
# * web: set the URL for the package
# * base image builder: use this as the path for package

from ._build_number import build_number  # Written by Github

# As long as we're on 0.*, all versions are published automatically
major_number = 0

# Bump this manually on any major changes
minor_number = 35

# Right now, set the patch number (the 3rd field) to the job run number in Github
__version__ = f"{major_number}.{minor_number}.{build_number}"
