# Copyright Modal Labs 2022
import modal

app = modal.App("my-app")

nfs = modal.NetworkFileSystem.lookup("volume_app")


@app.function()
def foo():
    print("foo")
