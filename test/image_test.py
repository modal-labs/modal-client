from polyester.image import CustomImage, Image, debian_slim, image_factory


@image_factory
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
