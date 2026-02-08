#!/bin/bash
set -euo pipefail

# Generate protos for client's container entrypoint tests
WORKING_DIRECTORY=$1
PYTHON_VERSION=$2

cd "$WORKING_DIRECTORY"

python -m venv venv

# shellcheck source=/dev/null
source venv/bin/activate

# Pin setuptools<82: v82.0.0 removed pkg_resources, which is needed at runtime
# by grpcio-tools (grpc_tools/protoc.py).
pip install "setuptools<82"

if [ "$PYTHON_VERSION" == "3.10" ]; then
pip install grpcio-tools==1.48.2 grpclib==0.4.7;
elif [ "$PYTHON_VERSION" == "3.12" ]; then
pip install grpcio-tools==1.59.2 grpclib==0.4.7;
elif [ "$PYTHON_VERSION" == "3.13" ]; then
pip install grpcio-tools==1.66.2 grpclib==0.4.7;
elif [ "$PYTHON_VERSION" == "3.14" ]; then
pip install grpcio-tools==1.76.0 grpclib==0.4.9;
fi
python -m grpc_tools.protoc --python_out=. --grpclib_python_out=. --grpc_python_out=. -I . modal_proto/api.proto modal_proto/task_command_router.proto
python -m grpc_tools.protoc --plugin=protoc-gen-modal-grpclib-python=protoc_plugin/plugin.py --modal-grpclib-python_out=. -I . modal_proto/api.proto

deactivate
