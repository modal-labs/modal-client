# Copyright Modal Labs 2022
import modal

app = modal.App()


@app.function()
def square(x):
    return x**2


# This should fail in a container
with app.run():
    print(square.remote(42))
