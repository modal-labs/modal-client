# Copyright Modal Labs 2024
from unittest import mock

import modal
import modal.experimental
from modal import App

app = App()


def double_it(x: int):
    return x * 2


@app.function()
@modal.experimental.grouped(size=2)
def f1(x: int):
    return double_it(x)


@app.function()
def f2():
    pass


def test_experimental_group(servicer, client):
    with app.run(client=client):
        assert len(servicer.app_functions) == 2

        fn1 = servicer.app_functions["fu-1"]  # f1
        assert fn1._experimental_group_size == 2

        fn2 = servicer.app_functions["fu-2"]  # f2
        assert not fn2._experimental_group_size


def test_experimental_group_call(servicer, client):
    custom_function = modal.experimental._networked(double_it)
    servicer.function_body(custom_function)

    with mock.patch("socket.getaddrinfo", return_value=[(0, 1, None, None, ("fdaa:bbcc:ddee:0:0:0:0:1", None))]):
        with app.run(client=client):
            f1.client = client
            output = f1.remote(x=10)
            assert output == 1
