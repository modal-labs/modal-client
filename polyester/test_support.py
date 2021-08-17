from . import base_image

@base_image.function
def square(x):
    return x*x


if __name__ == '__main__':
    raise Exception('This line is not supposed to be reachable')
