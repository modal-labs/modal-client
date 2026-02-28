# Design Doc: Clobber Post-Put Visibility Instrumentation for VolumeFS v2

**Status:** Draft  
**Author:** Cloud agent (for #fs follow-up)  
**Date:** 2026-02-28  
**Scope:** Clobber-side instrumentation only (no behavior change to client-facing APIs)

## 1) Background

We have multiple independent customer/internal reports where:

- `modal volume put` returns success quickly.
- Follow-up `modal volume ls` / `modal volume get` cannot find one or more newly uploaded files.
- In at least some cases, retrying `put` later eventually makes files visible.

Recent discussion indicates path confusion explains some cases, but does **not** explain all reports. In particular:

- There are reports of single-file puts apparently succeeding while the exact destination path remains missing.
- One report suggests eventual appearance after repeated retries over time.
- There are no corresponding reports of files disappearing from mounted volumes, which shifts suspicion toward control plane / indexing / request path issues rather than durable storage loss.

The proposed near-term action is to instrument Clobber so we can observe whether a successful put is visible in a fresh index shortly after acceptance.

## 2) Problem Statement

Today, we do not have direct, structured telemetry that answers:

1. For a successful `VolumePutFiles2` operation, did the uploaded path(s) become discoverable in the index immediately?
2. If not, is this transient (eventual consistency), persistent, or verification failure?
3. Are non-visible outcomes correlated with specific volume IDs, environments, client versions, file sizes, request shapes, or backend shards?

Without this signal, we cannot efficiently separate:

- true backend visibility/indexing issues,
- client pathing misunderstandings,
- race conditions, or
- partial-upload edge cases.

## 3) Goals

1. Add low-risk instrumentation to Clobber that verifies visibility of newly put files in a **fresh** index shortly after put success.
2. Emit structured logs + metrics that allow correlation by volume/request/client dimensions.
3. Keep user-facing behavior unchanged (no additional failures returned to clients).
4. Keep request latency impact negligible by doing checks asynchronously/off critical path.

## 4) Non-goals

1. No immediate semantic changes to `modal volume put/get`.
2. No automatic retries or server-side repair in this phase.
3. No user-facing error changes.
4. No hard guarantees against transient false negatives in this first instrumentation pass.

## 5) Hypotheses to Validate

H1. Most successful puts are visible in the fresh index on first check.

H2. A small subset may be temporarily non-visible and become visible within short bounded retries.

H3. Persistent non-visibility after retries is rare and likely clustered by specific dimensions (volume, shard, request type, size profile, or code path).

H4. If all verification checks pass while users still report missing files, root cause likely shifts toward client pathing/usage and not indexing.

## 6) Proposed Instrumentation Design

### 6.1 Hook point

Instrument on successful completion of Clobber put handlers for volume puts (focus on v2 path first):

- Primary: `VolumePutFiles2` success path.
- Optional parity: `VolumePutFiles` (v1), gated separately.

Trigger verification only after the put request has been accepted and committed according to current semantics.

### 6.2 Verification mode

Perform asynchronous verification in a background worker/task queue:

1. Build a verification payload from the successful put request:
   - `volume_id`
   - uploaded file paths (or sampled subset for very large batches)
   - request context (`workspace_id`, `environment`, auth principal, client metadata if present)
   - request/trace ID
2. Query index using a **fresh read path** (bypass local caches where possible).
3. Determine visibility result per sampled path:
   - present
   - absent
   - verification_error
4. Retry absent paths with bounded backoff to separate transient delay from persistent issues.

### 6.3 Suggested retry policy (initial)

- Attempt 1: immediately after request success.
- Attempt 2: +2s.
- Attempt 3: +10s.
- Attempt 4: +30s.

Classify final state after max attempts:

- `visible_first_try`
- `visible_after_retry`
- `not_visible_after_retries`
- `verification_failed` (errors checking visibility)

Rationale: this keeps false positives from normal propagation races low while still surfacing meaningful anomalies quickly.

### 6.4 Sampling policy

Start with conservative but informative coverage:

- 100% of v2 puts with <= 64 files.
- For larger batches, sample:
  - first file,
  - last file,
  - random N files (e.g. 8) weighted toward largest files.

This limits overhead while preserving detection for both small and very large upload patterns.

### 6.5 Structured logging

Emit a single summary event per verified put plus optional per-path debug events.

Suggested fields:

- identifiers: `request_id`, `trace_id`, `volume_id`, `workspace_id`, `environment`
- request shape: `file_count`, `sampled_file_count`, `total_bytes`, `is_v2`, `force_overwrite`
- client dims: `client_version`, `cli_vs_sdk` (if available), `platform`
- verification dims: `attempt_count`, `latency_ms_to_first_visible`, `final_state`
- counts: `visible_count`, `absent_count`, `verification_error_count`
- sampled path hashes (not raw paths by default; raw in debug mode only)

For `not_visible_after_retries`, log at warning/error level with enough context for triage.

### 6.6 Metrics

Add metrics suitable for dashboards and alerts:

Counters:

- `clobber.volume_put_verify.total`
- `clobber.volume_put_verify.visible_first_try`
- `clobber.volume_put_verify.visible_after_retry`
- `clobber.volume_put_verify.not_visible_after_retries`
- `clobber.volume_put_verify.error`

Histograms:

- `clobber.volume_put_verify.time_to_visible_ms`
- `clobber.volume_put_verify.sampled_paths_per_request`

Tags:

- `volume_fs_version`, `environment`, `workspace_tier`, `client_version_bucket`, `request_size_bucket`

### 6.7 Correlation with read failures

Add companion instrumentation to `VolumeGetFile2`/list not-found paths (if feasible in same effort):

- log not-found with request/trace and volume dims.
- this allows correlating "put verified absent" vs "get not found" incidents over time.

This is optional but high-value for narrowing root cause.

## 7) Performance and Safety Considerations

1. **Off critical path:** verification must never block the put response.
2. **Bounded resource use:** queue depth limits, per-volume rate limits, and global concurrency caps.
3. **Sampling controls:** adjustable at runtime (feature flags) to reduce overhead during incidents.
4. **Privacy:** avoid logging raw file paths by default; hash paths or log only prefixes.
5. **No user impact:** verification failures should not affect put success response.

## 8) Rollout Plan

### Phase 0: Dark launch scaffolding

- Add feature flags and no-op plumbing.
- Validate payload shape and overhead in staging.

### Phase 1: Logs-only in staging, 100% v2

- Enable summary logs + basic counters.
- Validate signal quality and false-positive rate.

### Phase 2: Prod canary

- Enable for a small percentage of v2 puts (e.g. 5%).
- Monitor overhead, queue behavior, and anomaly rate.

### Phase 3: Ramp

- Increase to 25% then 100% v2 once stable.
- Optional: extend to v1 if useful.

### Phase 4: Alerting + runbooks

- Add alerts on elevated `not_visible_after_retries` ratio.
- Add triage runbook linked to dashboards and representative log queries.

## 9) Triage Playbook (for oncall / #fs)

When `not_visible_after_retries` spikes:

1. Slice by environment/workspace to detect blast radius.
2. Slice by client version and request size profile.
3. Compare with get/list not-found rates.
4. Inspect shard/index backend health and queue lag.
5. Pull representative traces by `request_id` and compare write vs index-read paths.

## 10) Validation Plan

### 10.1 Staging validation scenarios

1. Single-file put, immediate verify success.
2. Multi-file put, sampled verification.
3. Simulated delayed index propagation (if test hooks exist), verify transitions to `visible_after_retry`.
4. Simulated index failure to verify `verification_failed` classification.

### 10.2 Success criteria

1. Instrumentation adds no measurable user-facing latency to put requests.
2. Verification coverage and classification metrics are stable and interpretable.
3. We can answer, from telemetry alone, whether successful puts are visible shortly after success.

## 11) Open Questions

1. What is the canonical "fresh index" read path in Clobber (and can we guarantee cache bypass)?
2. Should we verify all files for small requests and only sample for large ones, or always sample?
3. Should path hashes use salted hashing per workspace to improve privacy?
4. Do we want a one-off trace-level debug mode that logs raw paths for selected workspace IDs?
5. Should this instrumentation also cover NFS/shared volume put flows?

## 12) Future Extensions (Out of Scope for Initial Plan)

1. Automatic repair/requeue when visibility check fails.
2. User-facing warnings for likely delayed visibility.
3. End-to-end synthetic probes that continuously validate put->ls/get consistency.
4. Cross-system correlation with object storage ingestion and index refresh pipelines.

