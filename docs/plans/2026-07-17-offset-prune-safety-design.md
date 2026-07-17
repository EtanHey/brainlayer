# Offset Prune Safety Review Design

## Goal

Prune durable watcher offsets only when the watcher has positive evidence that the file's containing subtree is available, while keeping concurrent registry writes atomic, bounded, and crash-durable.

## Safety invariant

A missing tracked file is eligible for pruning only when all of these conditions hold:

1. Its most-specific configured watch root is available.
2. The tracked path and its ancestors do not cross an unavailable symlink.
3. The tracked file's immediate parent is an accessible directory with at least one live transcript discovered beneath it in the same poll.
4. The final file check confirms absence without a transient `OSError`.

If any check lacks positive evidence, the offset is preserved and the prune pass remains incomplete so a later poll retries it. This is deliberately conservative: a truly deleted single-file directory can retain a stale offset, but an emptied or missing nested mount cannot cause a live transcript to restart from byte zero.

## Alternatives considered

- Persist and compare OS mount-table state. This could prune more aggressively, but is platform-specific and cannot reliably identify every bind, network, or automounted subtree.
- Delay pruning for a grace period. This reduces short-outage risk but still loses offsets when an outage exceeds the grace period.
- Use parent-subtree evidence. This is portable, fail-closed, and uses evidence already produced by the watcher discovery pass. It is the selected approach.

## Registry integrity

- Sanitize persisted tombstones at both load sites, accepting only string paths with finite numeric timestamps.
- Merge only locally dirty paths under the existing exclusive registry lock.
- Preserve tombstones long enough to reject dirty writes from stale registry holders, then compact them after the bounded retention window.
- Flush and `fsync` the temporary JSON file, atomically replace the registry, and `fsync` its parent directory before clearing dirty state.

## Verification

Focused tests will prove:

- An empty nested directory with a live transcript elsewhere in the root cannot authorize pruning.
- A broken symlink used as the tracked JSONL path is preserved.
- A transient candidate stat failure leaves the pass retryable.
- Malformed tombstone JSON cannot break a flush.
- A stale registry holder with the deleted path marked dirty cannot resurrect it.
- The registry file and directory are synchronized before success is reported.

The complete watcher suite, Ruff, formatting, diff checks, and the repository pre-push gate remain required before the next exact-head push.
