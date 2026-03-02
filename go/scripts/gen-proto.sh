#!/bin/bash

rm -rf proto && mkdir -p proto

protoc \
  --go_out=paths=source_relative:proto \
  --go_opt=default_api_level=API_OPAQUE \
  --go-grpc_out=paths=source_relative:proto \
  --proto_path=../ \
  ../modal_proto/*.proto

# Find all 'package proto' declarations and replace with 'package pb'
find . -type f -name '*.go' -exec sed -i 's/^package proto$/package pb/' {} +
