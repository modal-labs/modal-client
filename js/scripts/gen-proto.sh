#!/bin/bash
# Called from package.json scripts.

mkdir -p proto

./node_modules/.bin/grpc_tools_node_protoc \
  --plugin=protoc-gen-ts_proto=./node_modules/.bin/protoc-gen-ts_proto \
  --ts_proto_out=./proto \
  --ts_proto_opt=outputServices=nice-grpc,outputServices=generic-definitions,useExactTypes=false \
  --proto_path=../ \
  ../modal_proto/*.proto

# Add @ts-nocheck to all generated files.
find proto -name '*.ts' | while read -r file; do
  if ! grep -q '@ts-nocheck' "$file"; then
    (echo '// @ts-nocheck'; cat "$file") > "$file.tmp" && mv "$file.tmp" "$file"
  fi
done

# HACK: Patch for bad Protobuf codegen: fix the "Object" type conflicting with
# builtin `Object` API in JavaScript and breaking Protobuf import.
perl -pi -e 's/Object\.entries/PLACEHOLDER_OBJECT_ENTRIES/g' proto/modal_proto/api.ts
perl -pi -e 's/\bObject\b/Object_/g' proto/modal_proto/api.ts
perl -pi -e 's/PLACEHOLDER_OBJECT_ENTRIES/Object.entries/g' proto/modal_proto/api.ts
