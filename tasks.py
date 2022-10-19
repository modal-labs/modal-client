import sys
from invoke import task


@task
def protoc(ctx):
    py_protoc = (
        f"{sys.executable} -m grpc_tools.protoc"
        + " --python_out=. --grpclib_python_out=. --grpc_python_out=. --mypy_out=. --mypy_grpc_out=."
    )
    print(py_protoc)
    ctx.run(f"{py_protoc} -I . modal_proto/api.proto")
