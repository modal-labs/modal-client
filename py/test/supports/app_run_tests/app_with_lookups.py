# Copyright Modal Labs 2022
import modal

app = modal.App("my-app")

volume = modal.Volume.from_name("volume_app").hydrate()


@app.function()
def foo():
    print("foo")
