from setuptools import setup

# PEP 396 prescribes setting a __version__ on the package,
# but there's no obvious way for setup.py to access it.
# Since __init__.py imports various third-party packages,
# we can't access __version__ at this point.
# Instead, we're using the using the approach suggested in
# https://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package
# and set the version info by executing a Python module.
# TODO: might want to look at pbr:
# https://docs.openstack.org/pbr/latest/user/index.html

about = {}
with open("modal/version.py") as f:
    exec(f.read(), about)

setup(name="modal", version=about["__version__"])
