# Copyright Modal Labs 2024
import modal
import modal.experimental
from modal import App

app = App()


@app.function()
@modal.experimental.grouped(size=2)
def f1():
    pass


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
