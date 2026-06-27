#!/bin/bash
# Shared Go proto toolchain. Downloads a pinned protoc and installs the pinned
# protoc-gen-go / protoc-gen-go-grpc plugins into a local cache, then exposes
# their paths so the pinned versions and checksums live in exactly one place.
#
# Sourced by client/go/scripts/gen-proto.sh and go/proto/protoc.sh. It lives
# under client/go/scripts/ because client/go/ is mirrored to the public
# modal-client repo, where gen-proto.sh must run self-contained; the
# monorepo-only go/proto/protoc.sh sources it via a relative path.
#
# After sourcing, the caller can use:
#   PROTOC      path to the pinned protoc binary
#   TOOLS_BIN   directory containing protoc-gen-go and protoc-gen-go-grpc
# The caller is responsible for `set -o errexit`/`pipefail`; this assumes both.

# Pinned tool versions. These MUST match the versions baked into the checked-in
# client/go *.pb.go headers and the hermetic Bazel generation
# (//client/go/proto/modal_proto:proto_generated); otherwise regeneration
# produces spurious version-header and codegen churn. Whatever protoc and
# plugins happen to be on PATH are deliberately ignored, so the output is
# reproducible regardless of the developer's local toolchain.
PROTOC_VERSION=21.6
PROTOC_GEN_GO_VERSION=v1.36.11
PROTOC_GEN_GO_GRPC_VERSION=v1.6.2

# protoc release archive name + checksum for this platform. The checksums are
# sha256 in base64 ("sha256-..." form), identical to the `integrity` values in
# the repo's MODULE.bazel so the two generation paths can be cross-checked.
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)  protoc_plat=linux-x86_64;   protoc_sha256="sha256-ap/DY2Oi0F1z/DY6Rs1X2EkGjTMwXbOfd9qsi6Bz6Bg=" ;;
  Linux-aarch64) protoc_plat=linux-aarch_64; protoc_sha256="sha256-T6l5fr85Fofjk5flgivdAluRlwXtR/h04lG6dJyXv0k=" ;;
  Darwin-x86_64) protoc_plat=osx-x86_64;     protoc_sha256="sha256-RayeZprwcIpHICHv4YdTx5V8Lz01kQAgL4DTCcHDlwg=" ;;
  Darwin-arm64)  protoc_plat=osx-aarch_64;   protoc_sha256="sha256-D/GVg2LbcWuTE1zNsQhL3MsOtCAoTc6CA4NchXLKvgs=" ;;
  *) echo "proto_toolchain.sh: unsupported platform $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac

cache_dir="${XDG_CACHE_HOME:-$HOME/.cache}/modal-go-proto"
TOOLS_BIN="$cache_dir/bin"
mkdir -p "$TOOLS_BIN"

# protoc: download and unpack the pinned release once, verifying the checksum.
# The archive bundles the well-known-type protos under include/, which protoc
# resolves automatically relative to its own binary.
PROTOC="$cache_dir/protoc-$PROTOC_VERSION/bin/protoc"
if [ ! -x "$PROTOC" ]; then
  echo "Downloading protoc $PROTOC_VERSION ($protoc_plat)..."
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  url="https://github.com/protocolbuffers/protobuf/releases/download/v$PROTOC_VERSION/protoc-$PROTOC_VERSION-$protoc_plat.zip"
  curl -fsSL -o "$tmp/protoc.zip" "$url"
  got="sha256-$(openssl dgst -sha256 -binary "$tmp/protoc.zip" | base64)"
  if [ "$got" != "$protoc_sha256" ]; then
    echo "proto_toolchain.sh: protoc checksum mismatch: got $got, want $protoc_sha256" >&2
    exit 1
  fi
  rm -rf "$cache_dir/protoc-$PROTOC_VERSION"
  unzip -oq "$tmp/protoc.zip" -d "$cache_dir/protoc-$PROTOC_VERSION"
  rm -rf "$tmp"
  trap - EXIT
fi

# Plugins: install the pinned versions into the cache via `go install`. Reuse an
# already-installed binary only when it reports the expected version.
install_plugin() {
  bin="$1" pkg="$2" version="$3" want="$4"
  have=""
  if [ -x "$TOOLS_BIN/$bin" ]; then
    have="$("$TOOLS_BIN/$bin" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
  fi
  if [ "$have" != "$want" ]; then
    echo "Installing $bin $version..."
    GOBIN="$TOOLS_BIN" go install "$pkg@$version"
  fi
}
install_plugin protoc-gen-go google.golang.org/protobuf/cmd/protoc-gen-go "$PROTOC_GEN_GO_VERSION" "${PROTOC_GEN_GO_VERSION#v}"
install_plugin protoc-gen-go-grpc google.golang.org/grpc/cmd/protoc-gen-go-grpc "$PROTOC_GEN_GO_GRPC_VERSION" "${PROTOC_GEN_GO_GRPC_VERSION#v}"
