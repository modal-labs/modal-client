#!/bin/bash
# 
# Build Go proto package.
set -o errexit
set -o nounset
set -o pipefail

# Set the specific git commit SHA
PROTO_PATH="pkg/modal"

# Download protos
mkdir -p $PROTO_PATH

# Build proto
echo "buidling Go gRPC client"
protoc --go_out=pkg --go_opt=paths=source_relative \
    --go-grpc_out=pkg --go-grpc_opt=paths=source_relative \
    --go_opt=M$PROTO_PATH/api.proto="github.com/modal-labs/modal-client" \
    --go_opt=M$PROTO_PATH/options.proto="github.com/modal-labs/modal-client" \
    --go-grpc_opt=M$PROTO_PATH/api.proto="github.com/modal-labs/modal-client" \
    --go-grpc_opt=M$PROTO_PATH/options.proto="github.com/modal-labs/modal-client" \
    $PROTO_PATH/api.proto $PROTO_PATH/options.proto
