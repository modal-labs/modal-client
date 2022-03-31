import sys

from modal import App, DockerhubImage
from modal.image import _dockerhub_python_version


def test_python_version():
    assert _dockerhub_python_version("3.9.1") == "3.9.9"
    assert _dockerhub_python_version("3.9") == "3.9.9"
    v = _dockerhub_python_version().split(".")
    assert len(v) == 3
    assert (int(v[0]), int(v[1])) == sys.version_info[:2]


def test_image_tag():
    app = App()
    image = DockerhubImage(app, tag="foo")
    assert image.tag == 'modal.image._DockerhubImage("foo")'
