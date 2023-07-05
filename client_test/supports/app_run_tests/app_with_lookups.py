# Copyright Modal Labs 2022
import modal

stub = modal.Stub("my-app")


@stub.function(network_file_systems={"/vol": modal.NetworkFileSystem.from_name("volume_app")})
def foo():
    print("foo")
