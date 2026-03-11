#!/usr/bin/env python3
# Copyright Modal Labs 2026
"""Mock modal-sandbox-fs-tools binary for unit tests.

Emulates the Rust binary's behavior for ReadFile and WriteFile, including
structured JSON error payloads on stderr. Accepts a single JSON argument.
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

# Allow override via env var for testing with small files.
_MAX_READ_FILE_SIZE = int(os.environ.get("_MODAL_TEST_MAX_READ_FILE_SIZE", 5 * 1024 * 1024 * 1024))

if "ReadFile" in command:
    source = command["ReadFile"]["path"]
    if not os.path.exists(source):
        _error_payload("NotFound", "path does not exist")
    if os.path.isdir(source):
        _error_payload("IsDirectory", "expected a file path")
    try:
        size = os.path.getsize(source)
        if size > _MAX_READ_FILE_SIZE:
            _error_payload(
                "FileTooLarge",
                f"file is {size} bytes, which exceeds the {_MAX_READ_FILE_SIZE} byte limit",
            )
        with open(source, "rb") as src:
            sys.stdout.buffer.write(src.read())
        sys.stdout.flush()
    except PermissionError:
        _error_payload("PermissionDenied", "permission denied")
    raise SystemExit(0)

if "WriteFile" in command:
    target = command["WriteFile"]["path"]
    if os.path.isdir(target):
        _error_payload("IsDirectory", "expected a file path")
    parent = os.path.dirname(target)
    if parent:
        if os.path.exists(parent) and not os.path.isdir(parent):
            _error_payload("AlreadyExists", "a component of the path is not a directory")
        try:
            os.makedirs(parent, exist_ok=True)
        except NotADirectoryError:
            _error_payload("NotDirectory", "a component of the path is not a directory")
        except PermissionError:
            _error_payload("PermissionDenied", "permission denied")
    try:
        with open(target, "wb") as dst:
            while True:
                chunk = sys.stdin.buffer.read(65536)
                if not chunk:
                    break
                dst.write(chunk)
    except PermissionError:
        _error_payload("PermissionDenied", "permission denied")
    raise SystemExit(0)

raise SystemExit(f"unknown command: {command}")
