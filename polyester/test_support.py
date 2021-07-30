from . import base_image

@base_image.function
def square(x):
    return x*x
