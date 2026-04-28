#!/usr/bin/env python3
# Copyright Modal Labs 2026
"""Mock modal-sandbox-fs-tools binary for unit tests.

Emulates the Rust binary's behavior for ReadFile and WriteFile, including
structured JSON error payloads on stderr. Accepts a single JSON argument.
"""

import grp
import json
import os
import pwd
import stat
import sys


def _error_payload(error_kind, message, detail=None):
    payload = {"version": 1, "error_kind": error_kind, "message": message}
    if detail is not None:
        payload["detail"] = detail
    sys.stderr.write(json.dumps(payload))
    raise SystemExit(1)


def _build_file_entry(name, path, st):
    """Build a FileEntry dict matching the Rust crate's `build_file_entry`."""
    if stat.S_ISLNK(st.st_mode):
        file_type = "symlink"
    elif stat.S_ISDIR(st.st_mode):
        file_type = "directory"
    else:
        file_type = "file"
    try:
        owner = pwd.getpwuid(st.st_uid).pw_name
    except KeyError:
        owner = str(st.st_uid)
    try:
        group = grp.getgrgid(st.st_gid).gr_name
    except KeyError:
        group = str(st.st_gid)
    entry = {
        "name": name,
        "path": path,
        "type": file_type,
        "size": st.st_size,
        "mode": st.st_mode,
        "permissions": f"{stat.S_IMODE(st.st_mode):04o}",
        "owner": owner,
        "group": group,
        "modified_time": st.st_mtime,
    }
    if file_type == "symlink":
        entry["symlink_target"] = os.readlink(path)
    return entry


if len(sys.argv) != 2:
    raise SystemExit("usage: modal-sandbox-fs-tools <command-json>")

command = json.loads(sys.argv[1])

# Allow override via env var for testing with small files.
_MAX_READ_FILE_SIZE = int(os.environ.get("_MODAL_TEST_MAX_READ_FILE_SIZE", 5 * 1024 * 1024 * 1024))

if "ListFiles" in command:
    target = command["ListFiles"]["path"]
    try:
        os.stat(target)
    except NotADirectoryError:
        # A path component is a file (OS-level ENOTDIR).
        _error_payload("NotDirectory", "a component of the path is not a directory")
    except PermissionError:
        _error_payload("PermissionDenied", "permission denied")
    except OSError:
        _error_payload("NotFound", "path does not exist")
    if not os.path.isdir(target):
        _error_payload("IsFile", "expected a directory path")
    try:
        entries = []
        for name in sorted(os.listdir(target)):
            full_path = os.path.join(target, name)
            try:
                st = os.lstat(full_path)
            except OSError:
                continue
            entries.append(_build_file_entry(name, os.path.abspath(full_path), st))
        sys.stdout.write(json.dumps(entries))
        sys.stdout.flush()
    except PermissionError:
        _error_payload("PermissionDenied", "permission denied")
    raise SystemExit(0)

if "MakeDirectory" in command:
    target = command["MakeDirectory"]["path"]
    create_parents = command["MakeDirectory"].get("parents", True)
    try:
        if create_parents:
            os.makedirs(target, exist_ok=True)
        else:
            os.mkdir(target)
    except FileExistsError:
        _error_payload("PathAlreadyExists", "path already exists")
    except FileNotFoundError:
        _error_payload("NotFound", "parent directory does not exist")
    except NotADirectoryError:
        _error_payload("NotDirectory", "a component of the path is not a directory")
    except PermissionError:
        _error_payload("PermissionDenied", "permission denied")
    raise SystemExit(0)

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

if "Remove" in command:
    target = command["Remove"]["path"]
    recursive = command["Remove"].get("recursive", False)
    if not os.path.lexists(target):
        _error_payload("NotFound", "path does not exist")
    try:
        if os.path.islink(target) or not os.path.isdir(target):
            os.remove(target)
        elif recursive:
            import shutil

            shutil.rmtree(target)
        else:
            os.rmdir(target)
    except PermissionError:
        _error_payload("PermissionDenied", "permission denied")
    except OSError as e:
        import errno

        if e.errno == errno.ENOTEMPTY:
            _error_payload("DirectoryNotEmpty", "directory is not empty")
        else:
            _error_payload("Io", str(e))
    raise SystemExit(0)

if "Stat" in command:
    target = command["Stat"]["path"]
    try:
        st = os.lstat(target)
    except FileNotFoundError:
        _error_payload("NotFound", "path does not exist")
    except NotADirectoryError:
        _error_payload("NotDirectory", "a component of the path is not a directory")
    except PermissionError:
        _error_payload("PermissionDenied", "permission denied")
    except OSError as e:
        _error_payload("Io", "I/O error", detail=str(e))
    name = os.path.basename(target.rstrip("/"))
    sys.stdout.write(json.dumps(_build_file_entry(name, target, st)))
    sys.stdout.flush()
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
