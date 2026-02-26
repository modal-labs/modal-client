# Copyright Modal Labs 2023
import modal

app = modal.App()


@app.function()
def foo():
    pass


@app.local_entrypoint()
def main():
    with app.run():  # should error here
        print("unreachable")
        foo.remote()  # should not get here
