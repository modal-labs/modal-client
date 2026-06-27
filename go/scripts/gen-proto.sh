#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset

# Download the pinned protoc + plugins (sets PROTOC and TOOLS_BIN). The shared
# toolchain lives alongside this script so it is mirrored to the public
# modal-client repo and gen-proto.sh stays self-contained there.
# shellcheck source=client/go/scripts/proto_toolchain.sh
source "$(dirname "$0")/proto_toolchain.sh"

# Run from client/go (the parent of this script's directory) so the relative
# paths below work regardless of the caller's working directory.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p proto && find proto -type f -name '*.go' -delete

"$PROTOC" \
  --plugin=protoc-gen-go="$TOOLS_BIN/protoc-gen-go" \
  --plugin=protoc-gen-go-grpc="$TOOLS_BIN/protoc-gen-go-grpc" \
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
