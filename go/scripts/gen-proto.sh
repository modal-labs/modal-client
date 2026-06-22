#!/bin/bash
set -o errexit

mkdir -p proto && find proto -type f -name '*.go' -delete

protoc \
  --go_out=paths=source_relative:proto \
  --go_opt=default_api_level=API_OPAQUE \
  --go-grpc_out=paths=source_relative:proto \
  --proto_path=../ \
  ../modal_proto/*.proto

# Find all 'package proto' declarations and replace with 'package pb'.
# Use a backup suffix so the -i flag is portable across GNU and BSD (macOS) sed,
# then remove the backups.
find . -type f -name '*.go' -exec sed -i.bak 's/^package proto$/package pb/' {} +
find . -type f -name '*.go.bak' -delete
