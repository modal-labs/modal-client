# Copyright Modal Labs 2022
import modal

app = modal.App("my-app", include_source=True)  # TODO: remove include_source=True)

nfs = modal.NetworkFileSystem.from_name("volume_app").hydrate()


@app.function()
def foo():
    print("foo")
