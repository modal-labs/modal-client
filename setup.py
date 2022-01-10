from setuptools import setup

# Parse version data
about = {}
with open("modal/version.py") as f:
    exec(f.read(), about)

setup(name="modal", version=about["__version__"])
