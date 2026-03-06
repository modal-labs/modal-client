#!/usr/bin/env python3
# Copyright Modal Labs 2026
"""Mock modal-sandbox-fs-tools binary for unit tests.

Emulates the Rust binary's behavior for ReadFile, including structured
JSON error payloads on stderr. Accepts a single JSON argument.
"""

import json
import os
import sys


def _error_payload(error_kind, message, detail=None):
    payload = {"version": 1, "error_kind": error_kind, "message": message}
    if detail is not None:
        payload["detail"] = detail
    sys.stderr.write(json.dumps(payload))
    raise SystemExit(1)


if len(sys.argv) != 2:
    raise SystemExit("usage: modal-sandbox-fs-tools <command-json>")

command = json.loads(sys.argv[1])

if "ReadFile" in command:
    source = command["ReadFile"]["path"]
    if not os.path.exists(source):
        _error_payload("NotFound", "path does not exist")
    if os.path.isdir(source):
        _error_payload("IsDirectory", "expected a file path")
    try:
        with open(source, "rb") as src:
            sys.stdout.buffer.write(src.read())
        sys.stdout.flush()
    except PermissionError:
        _error_payload("PermissionDenied", "permission denied")
    raise SystemExit(0)

raise SystemExit(f"unknown command: {command}")
