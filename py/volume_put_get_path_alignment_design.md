# Design Doc: Align `modal volume put/get` Path Semantics with `aws s3 cp`

**Status:** Draft  
**Author:** Cloud agent (for #fs follow-up)  
**Date:** 2026-02-28  
**Scope:** Modal Python CLI (`modal volume put`, `modal volume get`)

## 1) Context and Motivation

Multiple user reports described `modal volume put` as succeeding but files seeming to "not appear", especially for v2 volumes and large checkpoints. Investigation in Slack indicates this is primarily a path semantics mismatch rather than delayed persistence:

- `put` currently can create unexpected extra nesting (`dst/src_basename/...`) depending on trailing slash handling.
- `get` currently always preserves the remote source prefix as an extra local top-level directory, even when users expect contents-only copy.
- The resulting behavior is easy to interpret as lost files, because users inspect a different path than intended.

The same user intent in `aws s3 cp --recursive` generally maps to "copy directory contents into destination prefix", without the unexpected extra nesting observed with current `modal volume` behavior.

## 2) Problem Statement

`modal volume put/get` currently uses path rules that are:

1. Inconsistent between `put` and `get`.
2. Sensitive to destination trailing slash for `put`, but largely insensitive for `get`.
3. Different from the mental model many users bring from `aws s3 cp`.

This causes user confusion, failed checkpoint workflows, and support load.

## 3) Goals

1. Make `modal volume put/get` directory behavior match `aws s3 cp` user expectations.
2. Ensure `put` and `get` are symmetric for directory workflows.
3. Preserve current single-file behavior where possible.
4. Provide clear migration guidance for any behavior change.

## 4) Non-goals

1. No protocol/backend storage changes.
2. No changes to Python SDK volume APIs (`Volume.batch_upload`, etc.) in this effort.
3. No glob semantics changes (`volume get` globs are already deprecated).
4. `modal nfs` alignment is considered separately (shared code impact is discussed below).

## 5) Current Behavior Analysis (Code-Level)

### 5.1 `modal volume put`

Current logic in `py/modal/cli/volume.py`:

- If `remote_path.endswith("/")`, CLI rewrites destination to:
  - `remote_path + basename(local_path)`
- This rewrite happens before directory/file dispatch.
- For directory uploads, `batch.put_directory(local_path, remote_path)` then places files relative to that rewritten path.

Effect:

- `modal volume put vol /local/client ./client-test/`
  - current result: `./client-test/client/...` (extra nesting)
- `modal volume put vol /local/client ./client-test`
  - current result: `./client-test/...` (often what users wanted)

So current behavior depends on destination slash in a way users do not anticipate.

### 5.2 `modal volume get`

Current logic in `py/modal/cli/_download.py`:

- Producer computes:
  - `start_path = Path(remote_path).parent.as_posix().split("*")[0]`
  - `rel_path = PurePosixPath(entry.path).relative_to(start_path.lstrip("/"))`
- Destination mapping preserves `rel_path`.

For source `client-test3` and `client-test3/`, `Path(...).parent` is effectively the same in practice, so both resolve to the same base and preserve `client-test3/...` in output.

Effect:

- `modal volume get vol client-test3 localdir`
  - current result: `localdir/client-test3/...`
- `modal volume get vol client-test3/ localdir`
  - current result: same (`localdir/client-test3/...`)

This means users cannot request "contents-only" download via slash conventions.

### 5.3 Why users interpret this as data loss

- Users typically inspect `dst/...` and do not expect an extra top-level segment.
- With large trees/checkpoints, missing expected filenames in immediate listing looks like upload/download inconsistency.
- This is especially confusing when `put` and `get` treat slash-related intent differently.

## 6) Target Semantics (AWS-Aligned)

### 6.1 Normative rules

#### Rule A: `put` file source

- If destination ends with `/`, treat destination as directory prefix and append file basename.
- Otherwise treat destination as exact remote file path.

#### Rule B: `put` directory source

- Treat destination as the target prefix for **directory contents**.
- Do **not** auto-append source directory basename.
- Trailing slash on destination should not introduce additional nesting.

#### Rule C: `get` remote file source

- If local destination is existing directory: write `<dest>/<basename(remote_file)>`.
- Otherwise write exactly to local destination path (current behavior).
- `-` (stdout) continues to work for file downloads.

#### Rule D: `get` remote directory/prefix source

- Download **contents** of source prefix into local destination directory.
- Do not add an extra `<remote_prefix_basename>` directory automatically.
- If local destination does not exist, create it as a directory.

#### Rule E: Source trailing slash for `get`

- `remote_path` with and without trailing slash should behave equivalently for directories/prefixes.
- Neither variant should force extra nesting.

#### Rule F: Root behavior

- `modal volume get <vol> / <dest>` preserves full tree under `<dest>`.
- `modal volume put <vol> <local_dir> /` copies local directory contents to volume root.

### 6.2 Example matrix

| Command | Current behavior | Target behavior |
|---|---|---|
| `put vol ./client ./dst/` | `dst/client/...` | `dst/...` |
| `put vol ./client ./dst` | `dst/...` | `dst/...` |
| `get vol dst local` | `local/dst/...` | `local/...` |
| `get vol dst/ local` | `local/dst/...` | `local/...` |

## 7) Proposed Technical Plan

### 7.1 Introduce explicit path-mapping semantics for CLI

Add internal helpers (module-local or `cli` utility) that map source/destination intent to canonical paths:

- `resolve_put_destination(local_path, remote_path, is_dir_source)`
- `resolve_get_destination(remote_path, local_destination, source_kind)`

Key point: avoid implicit behavior based on `Path(...).parent` and filesystem state alone.

### 7.2 `volume put` changes

In `py/modal/cli/volume.py`:

1. Determine source kind first (`Path(local_path).is_dir()`).
2. Only apply "append basename when destination ends with `/`" for file source.
3. For directory source, pass canonical destination prefix directly (no basename append).
4. Update success message text to reflect effective destination prefix.

### 7.3 `volume get` changes

In `py/modal/cli/_download.py`:

1. Replace `start_path = Path(remote_path).parent...` with source-aware relative-root logic:
   - file source root: parent of file
   - directory/prefix source root: source prefix itself
2. For directory/prefix downloads, always treat local destination as directory target (create if missing).
3. Keep existing single-file-to-explicit-file behavior.
4. Preserve `-` stdout behavior for file downloads only.

### 7.4 Shared-code blast radius (`nfs get`)

`_volume_download` is shared by `volume get` and `nfs get`.

Plan:

- Add a small strategy switch or call-site parameter so behavior can be changed for `volume get` first.
- Defer `nfs get` semantics decision unless explicitly included in scope.

This avoids accidental behavior changes for `nfs` workflows.

### 7.5 CLI help and docs text

Update `modal volume put/get` help text in `py/modal/cli/volume.py` to clearly describe:

- file vs directory semantics
- destination slash handling
- examples for "contents copy" behavior

## 8) Compatibility / Migration

This is behavior-changing for users who rely on current implicit nesting.

### 8.1 Potential breakages

1. Scripts expecting `put dir dst/` to create `dst/<basename(dir)>/...`
2. Scripts expecting `get prefix dest` to create `dest/<prefix>/...`

### 8.2 Migration guidance

- To preserve old `put` result, explicitly include basename in destination:
  - old implicit: `put vol ./client dst/`
  - new explicit: `put vol ./client dst/client`
- To preserve old `get` result, include prefix in local destination:
  - old implicit: `get vol dst local`
  - new explicit equivalent: `get vol dst local/dst`

### 8.3 Communication

- Add release note in changelog calling out path semantics alignment.
- Include before/after examples in release notes.

## 9) Testing Plan

Primary tests: `py/test/cli_test.py` (high-signal CLI behavior coverage).

### 9.1 New/updated test cases

1. `volume put` directory + trailing-slash destination:
   - assert no auto basename nesting.
2. `volume put` directory + non-trailing destination:
   - assert same result as trailing variant (contents under destination prefix).
3. `volume get` directory/prefix to existing directory:
   - assert contents copied directly into destination (no extra prefix directory).
4. `volume get` directory/prefix to non-existing local destination:
   - destination created as directory; contents copied directly.
5. `volume get` file to explicit file path:
   - unchanged behavior.
6. `volume get` file to existing directory:
   - unchanged behavior (`basename` output).
7. `volume get /` full-tree behavior:
   - unchanged.

### 9.2 Version coverage

Run the same behavioral cases for v1 and v2 where feasible, with special confidence focus on v2 (source of user reports).

### 9.3 Manual verification

Run quick CLI round-trip scenario mirroring Slack repro:

1. `put` local dir to `dst/`
2. `ls` destination
3. `get` destination to local dir
4. local tree diff check against original (contents-level expectation)

## 10) Rollout Plan

1. Land implementation + tests in one PR.
2. Add changelog entry and migration note.
3. Monitor support/Slack for reduced "missing file after put/get" reports.
4. Optional follow-up: align `modal nfs put/get` semantics intentionally in a separate change.

## 11) Open Questions

1. Should `modal nfs get` adopt the same semantics in the same release or a follow-up?
2. Do we want an explicit opt-out/legacy flag for one release, or ship direct behavior correction?
3. How should ambiguous remote paths be handled if both `foo` file and `foo/...` prefix exist (prefer file unless slash suffix?)?

## 12) Acceptance Criteria

1. The Slack repro cases no longer produce unexpected extra nesting.
2. `put` and `get` directory workflows are symmetric and intuitive.
3. New CLI tests lock in the target behavior and prevent regressions.
4. Changelog clearly documents migration implications.

