# Copyright Modal Labs 2022
import modal

stub = modal.Stub("my-app")


@stub.function(shared_volumes={"/vol": modal.SharedVolume.from_name("volume_app")})
def foo():
    print("foo")
