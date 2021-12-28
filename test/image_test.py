import sys

from modal.image import CustomImage, Image, debian_slim, dockerhub_python_version


def test_python_version():
    assert dockerhub_python_version("3.9.1") == "3.9.9"
    assert dockerhub_python_version("3.9") == "3.9.9"
    v = dockerhub_python_version().split(".")
    assert len(v) == 3
    assert (int(v[0]), int(v[1])) == sys.version_info[:2]


@Image.factory
def my_image(pkg="python-numpy"):
    return CustomImage(dockerfile_commands=[f"apt-get install {pkg}"])


def test_image_factory():
    assert isinstance(my_image, Image)
    assert isinstance(my_image.tag, str)
    assert my_image.tag == "test.image_test.my_image"

    my_image_2 = my_image(pkg="python-scipy")
    assert isinstance(my_image_2, Image)
    assert isinstance(my_image_2.tag, str)
    assert my_image_2.tag == 'test.image_test.my_image("python-scipy")'


def test_debian_slim():
    assert isinstance(debian_slim, Image)
    assert isinstance(debian_slim(["numpy"]), Image)
