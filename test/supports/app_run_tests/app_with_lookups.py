# Copyright Modal Labs 2022
import modal

app = modal.App("my-app")

nfs = modal.Volume.lookup("volume_app", nfs=True)


@app.function()
def foo():
    print("foo")
