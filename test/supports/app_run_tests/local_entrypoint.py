# Copyright Modal Labs 2022

import modal

app = modal.App()


@app.function()
def foo():
    pass


@app.local_entrypoint()
def main():
    print("called locally")
    foo.remote()
    foo.remote()
