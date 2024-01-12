# Copyright Modal Labs 2022
import modal

stub = modal.Stub("my-app")

nfs = modal.NetworkFileSystem.lookup("volume_app")


@stub.function()
def foo():
    print("foo")
