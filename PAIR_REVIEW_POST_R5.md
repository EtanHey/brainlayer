# D1 Post-Acceptance R5 Re-review

**Verdict: `ACCEPT`**

The post-acceptance delta closes R5 at the right layer. It does **not** clamp the confirmed offset against the
file size (which would be racy); it binds every durable watermark to the *provenance* of the bytes that
produced it — the source inode **and** the rewind generation in effect when the line was read. Old-inode and
pre-rewind watermarks are dropped; current-generation watermarks advance unchanged.

I reproduced the R5 hazard independently at the pre-delta wiring (registry offset **2050** against a **111**-byte
replacement), confirmed the delta reduces it to exactly **111**, and confirmed the same gate also covers the
same-inode rewind case the brief's test does not exercise.

---

## What was reviewed

- Worktree: `/Users/etanheyman/Gits/worktrees/bl-d1-oversized-ingestion`, branch `fix/d1-oversized-ingestion`
- Baseline: `cc1d21f66534747ee4a50367c290858972629a70`
- Still **uncommitted**, so citations carry blob hashes rather than a commit SHA. **The blobs moved again**
  since `PAIR_REVIEW_FINAL.md`:
  - `src/brainlayer/watcher.py` → `git hash-object` = `dede996f006814873f816b40e5d31d3c8fddcc31`
    (was `16802db5` at acceptance)
  - `tests/test_jsonl_watcher.py` → `git hash-object` = `84d47856bc2b5976aa7fa24638d9e53d371ad8a2`
    (was `b32f3051`)
  - `git diff cc1d21f6 --stat` → `+816 / -179` across 2 files (was `+724 / -176`)
- `16802db5` / `b32f3051` are **not in the object store** (`git cat-file -t` → `could not get object info`), so
  I could not diff acceptance→delta directly. I reviewed the current code against baseline for every named
  symbol, and reconstructed the pre-delta behaviour by reverting the one wiring line in a scratch copy (below).
- Module resolution verified at runtime: `brainlayer.watcher` →
  `…/bl-d1-oversized-ingestion/src/brainlayer/watcher.py`.
- Line numbers below are `watcher.py:N @ dede996f` and `test_jsonl_watcher.py:N @ 84d47856`.

---

## Commands run and exact counts

| command | result |
|---|---|
| the 6 targets in the brief §"Run at minimum" | **47 passed** |
| `python3 -m pytest tests/test_jsonl_watcher.py -q` | **107 passed** |
| the 14-file watcher / health / bridge group | **338 passed** |
| `ruff check src/brainlayer/watcher.py tests/test_jsonl_watcher.py` | **All checks passed!** |
| `ruff format --check` on both files | **2 files already formatted** |
| pre-delta wiring, `tests/test_jsonl_watcher.py` | **1 failed, 106 passed** |

14-file group = `test_jsonl_watcher test_alarm test_watch_backfill_cli test_watcher_bridge
test_watcher_provenance_ingest test_ingest_denylist test_ingest_guard test_throughput_watchdog
test_cli_index_watchdog test_drain_health test_doctor test_status_truthfulness test_fts5_health
test_stability_health_check`.

**The implementer's claims check out exactly.** `107` on `test_jsonl_watcher.py` and `338` on the 14-file group
are both reproduced verbatim, and `338` is `337 + 1` against `PAIR_REVIEW_FINAL.md` — the one new test, with no
other test count moving.

### Red-before-green — verified, not assumed

The delta is not isolatable by git, so I copied the tree to a scratch dir and reverted **only** the wiring at
`watcher.py:1012`, back to the pre-delta form:

```python
-            on_confirm_batch=self._advance_confirmed_batch,
+            on_confirm_offsets=self._advance_confirmed_offsets,
```

That makes the provenance gate inert while leaving the stamping and the accessor in place. Result:

```
1 failed, 106 passed

>       assert watcher.registry.get(str(rollout)) == (replacement_size, replacement_inode)
E       assert (2050, 202124689) == (111, 202124689)
E         At index 0 diff: 2050 != 111
```

Two things this establishes:

1. **The hazard is real and the test catches it.** `2050` is the old inode's watermark; the replacement file is
   `111` bytes. The registry recorded an offset **18.5× the file size** — exactly the poisoned watermark R5
   described, reproduced by me rather than taken from the report.
2. **The delta is surgical.** `106` other tests are indifferent to the wiring — the gate changes nothing about
   healthy confirmation. Only the new test moves.

---

## The delta, line by line

### `OffsetRegistry.generation` — `watcher.py:364-366`

```python
def generation(self, filepath: str) -> int:
    return self._entry_generation(self._data.get(filepath))
```

A read-only accessor over the **already-existing** `_entry_generation` validator (`watcher.py:336-344`, present
byte-identical at `cc1d21f6:312-318`). I diffed the whole `OffsetRegistry` region against baseline: the only
added lines in the class are these three. **No serialization change, no `offsets.json` schema change, no
migration risk** — `set()` (`watcher.py:368-384`), `flush()`, `remove()`, and `mark_rewind()` are untouched.

The generation source of truth is `mark_rewind` (`watcher.py:517-532`), which sets
`max(current+1, tombstone+1, time.time_ns())` and resets the offset to 0. It is called from all three
replacement/rewind detectors: `_ensure_tailer` on an existing tailer (`watcher.py:1471-1476`), `_ensure_tailer`
on a cold tailer whose stored inode disagrees (`watcher.py:1484-1487`), and `_handle_rewind`
(`watcher.py:1501-1502`). So both failure shapes — inode replacement and same-inode truncate — bump it.

### `BatchIndexer.on_confirm_batch` — `watcher.py:832, 838, 884-887, 948-951`

Both flush paths dispatch identically:

```python
# normal flush, watcher.py:884-887
if watermarks and self.on_confirm_batch:
    self.on_confirm_batch(watermarks, batch)
elif watermarks and self.on_confirm_offsets:
    self.on_confirm_offsets(watermarks)
```

```python
# isolated per-item flush, watcher.py:948-951
if confirmed and self.on_confirm_batch:
    self.on_confirm_batch(confirmed, confirmed_items)
elif confirmed and self.on_confirm_offsets:
    self.on_confirm_offsets(confirmed)
```

The isolated path passes `confirmed_items` (`watcher.py:945`) — only the items that actually flushed, not the
whole batch — so the gate can never confirm bytes for a record that was retained. Correct.

**`on_confirm_offsets` compatibility is preserved, not just claimed.** The parameter and attribute still exist
(`watcher.py:831, 837`); the `elif` keeps the legacy single-argument contract for any caller that supplies only
it. `tests/test_jsonl_watcher.py:732` constructs a `BatchIndexer` with `on_confirm_offsets=confirmed.append`
directly and passes. `JSONLWatcher` is now the only in-tree caller that uses the batch form (`watcher.py:1012`);
grep across the repo returns no other `on_confirm_*` consumer.

**`FlushWatermarks` compatibility.** `watcher_bridge.py:74` defines it as `dict[str, int]` subclass;
`_confirmed_watermarks` (`watcher.py:901-904`) gates on `isinstance(result, dict)`, so it still passes through
untouched, and `getattr(result, "inserted", …)` still reads the subclass attribute at `watcher.py:891`.

### `_source_inode` / `_source_generation` stamping — `watcher.py:1621-1627`

```python
normalized_lines = self._normalize_lines(filepath, new_lines) if new_lines else []
if normalized_lines:
    source_generation = self.registry.generation(filepath)
    for line in normalized_lines:
        line["_source_inode"] = tailer.observed_inode
        line["_source_generation"] = source_generation
    self.indexer.add(normalized_lines)
```

Stamped **after** rewind handling (`watcher.py:1611-1619`) and **after** `_ensure_tailer`
(`watcher.py:1602`), so the values are the post-detection generation and the post-replacement tailer.
`observed_inode` is fixed at tailer construction (`watcher.py:657`), and a replacement always constructs a
**new** tailer (`watcher.py:1475`, `1487`), so the two never drift within one tailer's life.

The `drain_buffer` path (`watcher.py:1597-1600`) skips `_ensure_tailer`, but it is only reached when the
explicit inode check at `watcher.py:1582-1586` says the inode is unchanged, so the stamp is still correct there.

**Blast radius of the two new keys:** they ride on the entry dict into `flush_to_db`. The bridge reads entries
by explicit key only (`watcher_bridge.py:162, 298, 339, 442, 457, 582`) and nothing in the ingest path
serializes the whole entry, so the keys cannot leak into stored chunk content. Confirmed by grep: no
`json.dumps(entry|item|line)` anywhere under `src/brainlayer/` on the ingest path. The one place a full entry
*is* serialized is `_quarantine_entries` (`watcher.py:920`), where the extra provenance is a diagnostic
improvement, and both values are plain ints so nothing can raise on serialization.

### `JSONLWatcher._advance_confirmed_batch` — `watcher.py:1047-1067`

```python
current_inode = tailer.observed_inode
current_generation = self.registry.generation(filepath)
eligible_offsets = [
    item["_line_end_offset"]
    for item in batch
    if item.get("_source_file") == filepath
    and item.get("_source_inode") == current_inode
    and item.get("_source_generation") == current_generation
    and isinstance(item.get("_line_end_offset"), int)
    and item["_line_end_offset"] <= reported_offset
]
if eligible_offsets:
    current_watermarks[filepath] = max(eligible_offsets)
self._advance_confirmed_offsets(current_watermarks)
```

Five conjuncts, each doing work:

- `_source_file` — a batch mixes files; without it one file's offset could be written under another's key.
- `_source_inode` — kills the replaced-file case.
- `_source_generation` — kills the same-inode rewind case, which the inode check alone cannot see.
- `isinstance(..., int)` — every dict line gets `_line_end_offset` at `watcher.py:774-778` and it survives
  normalization at `watcher.py:1148-1149`, so this is belt-and-braces, not a silent skip.
- `<= reported_offset` — **the sink still has veto power.** The gate can only ever *narrow* what the sink
  confirmed, never widen it. This is the property that makes the change safe: `max(eligible) <= reported`
  always, so no path can now confirm bytes the sink did not.

`_advance_confirmed_offsets` (`watcher.py:1038-1045`) is unchanged and still monotone (`if offset >=
current_offset`), so the gate composes with it rather than replacing its invariant.

### The RED test — `tests/test_jsonl_watcher.py:1459-1508`

It builds the full shape rather than a mock: 20 records retained by a failing sink
(`test_jsonl_watcher.py:1467-1469`), `os.replace` with a 3-record file (`:1494`), a poll to let the watcher
observe the replacement (`:1497`), then the sink recovers and the retained batch — now 23 items with two
different provenances — flushes in one call (`:1499-1500`). The assertion is on the *tuple*
(`offset, inode`), so it catches a wrong inode as well as a wrong offset. The final append + assert
(`:1504-1508`) is the part I value most: it proves the gate is a filter, not a freeze — the replacement keeps
advancing normally afterwards.

---

## Independent probes

All against the real `JSONLWatcher` / real `poll_once`, temp registries, temp dirs, no production DB, no
production `offsets.json` touched.

**A — same-inode rewind (the case the test does *not* cover).** 20 records retained by a failing sink, then
the same file truncated and rewritten smaller with the inode preserved:

```
inode unchanged: True   retained_stale_items=20
generation before=0  after=1785537351974196000   bumped=True
stale watermark offered=2050   new file size=111
registry offset=111   <= size: True
```

The inode check is inert here (`inode unchanged: True`); it is the **generation** conjunct alone that stops the
2050 watermark. Both halves of the gate are load-bearing.

**B — current-generation watermarks still advance normally.** 30 records, batch size 7, healthy sink:

```
registry offset=1130 / size=1130   fully confirmed: True   gen=0
after append: offset=1166 / size=1166   advanced: True
```

Note `gen=0`: on the healthy path the generation never moves, so the equality check is trivially satisfied and
costs nothing. No stall, no lag introduced.

**C — no data loss, only watermark loss.** The replacement case, inspecting what actually reached the sink:

```
delivered entries: 23
old- records present in delivered payload: 20
new- records present in delivered payload: 3
distinct _source_inode values:      [202132471, 202132472]
distinct _source_generation values: [0, 1785537365240743000]
registry: (111, 202132472)   new size: 111   new inode: 202132472
```

This is the point worth stating plainly: **the retained old-inode records are still indexed.** The delta
discards the stale *watermark*, not the stale *data*. Nothing that was read is dropped.

**D — no throughput regression.** Identical 20,000-record / 24.8 MB drain through `poll_once`, pre-delta wiring
vs. delta, same harness, same machine:

```
PRE-DELTA : parsed=20000 polls=379 cpu=0.132s offset=24828890/24828890
POST-DELTA: parsed=20000 polls=379 cpu=0.124s offset=24828890/24828890
```

Identical poll count, identical final offset, CPU within noise. The gate is O(batch × files-in-watermarks) with
a default batch of 10, and adds two ints per entry.

**E — no lock, offset-format, health, or DB change.**

- **Locks:** `on_confirm_batch` is invoked from `_do_flush` with `BatchIndexer._lock` held — the same position
  `on_confirm_offsets` occupied before, so the lock discipline is unchanged. I traced the new callee for
  re-entry: `_advance_confirmed_batch` → `_advance_confirmed_offsets` → `registry.set` +
  `_advance_quarantined_offsets` (`watcher.py:1069-1088`). None of them touch `self.indexer`, so the
  non-reentrant `threading.Lock` cannot deadlock. (`has_buffered_source` at `watcher.py:871-874` *does* take
  the lock, but it is only called from `_checkpoint_discarded_progress` on the poll thread, outside `_do_flush`.)
- **Offsets:** registry gains a read accessor only; `offsets.json` bytes are unchanged in shape.
- **Health:** no health field, counter, or threshold is touched by the delta; the 14-file group covering
  `test_drain_health`, `test_status_truthfulness`, `test_throughput_watchdog`, `test_stability_health_check` is
  green at 338.
- **DB:** no new DB access. Production `brainlayer.db` and the live `offsets.json` were not opened by any probe.

---

## Residual risk

| # | Area | Risk | Evidence |
|---|---|---|---|
| P1 | Offsets | `tailer is None` → the watermark is **dropped entirely**, not applied via the registry inode as the pre-delta path did (`watcher.py:1051-1053` vs `watcher.py:1041`). Only reachable for paths popped from `_tailers`, which happens only in the two denylist loops (`watcher.py:1559-1571`) where `registry.remove()` runs anyway. Consequence if ever reached otherwise: stalled offset → re-read → duplicates, never loss. | `watcher.py:1051-1053` |
| P2 | Offsets | `set()` bumps the generation when a tombstone exists and the inode is valid (`watcher.py:373-374`). For a path that was removed and then reappears, the first confirm can therefore change the generation *underneath* a batch already stamped with the older value, dropping that one watermark. Self-heals on the next poll (later batches carry the new generation and a higher offset). Cost is a transient lag / duplicate re-read, not loss. | `watcher.py:370-384` |
| P3 | Silent loss | The gate is provenance-based, so it deliberately does **not** clamp against file size. An in-place rewrite that is *larger* than the current offset still trips neither `check_rewind` (`watcher.py:659-680`, byte-identical to baseline) nor the generation bump. This is silent class 3 from `PAIR_REVIEW_FINAL.md` — pre-existing, unchanged, and still deserving its own tracked defect. It is **not** reopened or widened by this delta. | `watcher.py:659-680` |
| P4 | Throughput | `has_buffered_source` (`watcher.py:871-874`) matches on `_source_file` only, with no generation awareness. A permanently-retained stale batch for a path therefore keeps blocking `_checkpoint_discarded_progress` (`watcher.py:1167-1168`) for the *new* generation of that same path. Conservative and pre-existing, but it is now the one place in the offset path that has not learned about generations. Cheap follow-up: reuse the same conjunct there. | `watcher.py:871-874`, `1167-1168` |
| P5 | Offsets | If a sink reports a watermark that is not equal to any single item's `_line_end_offset`, the registry now advances only to the highest eligible item — an under-advance relative to the sink's intent. Not reachable in production: `confirm_entry` (`watcher_bridge.py:297-300`) only ever reports an actual entry's `_line_end_offset`. Affects synthetic sinks only, and errs toward duplicates. | `watcher_bridge.py:297-300, 633` |
| P6 | Offsets | Narrow TOCTOU: the gate reads `tailer.observed_inode` while `_advance_confirmed_offsets` writes `tailer.get_inode()` (a fresh stat, `watcher.py:1041`). A replacement landing between them writes the new inode with a current-generation offset. Self-healing: `_ensure_tailer`'s second clause (`watcher.py:1469`) compares against `tailer.observed_inode`, not the registry, so the next poll still detects it and resets to 0. Pre-existing shape; the delta does not worsen it. | `watcher.py:1041, 1469` |
| — | R1/R3/R4/R7/R8/R9/R10 from `PAIR_REVIEW_FINAL.md` | Untouched by this delta; all still stand as written there. | — |

**R5 itself is closed.** The residual it logged — "a retained flush-failure batch from an old inode … can write
a watermark beyond the new file's size" — is no longer constructible: I reproduced it at the pre-delta wiring
(`2050` vs `111`) and it does not occur post-delta (`111` vs `111`), for both the inode-replacement and the
same-inode-rewind shapes.

---

## What is right, and worth saying plainly

- **The fix is at the correct layer.** A file-size clamp on `_advance_confirmed_offsets` would have been the
  obvious move and would have been racy — a growing file can outrun the stat. Binding the watermark to the
  provenance of the bytes that produced it is exact, and it needs no I/O at confirm time.
- **The gate can only narrow, never widen.** `item["_line_end_offset"] <= reported_offset` means the sink keeps
  its veto. No new way to confirm bytes the sink did not confirm was introduced.
- **It filters watermarks, not data.** Probe C: all 23 records — including the 20 from the dead inode — still
  reach the sink. The delta throws away a stale claim about durability, not durable work.
- **Both conjuncts earn their place.** Probe A is the inode-unchanged case where only generation saves it; the
  brief's test is the generation-adjacent case where the inode is the clearer signal. Neither alone is enough.
- **It reuses existing machinery.** `generation` was already validated, persisted, and bumped by `mark_rewind`
  at `cc1d21f6`; the delta adds a three-line accessor rather than a new concept, which is why the registry file
  format is untouched and 106 of 107 tests are indifferent to it.
- **Zero cost on the healthy path.** Probe B shows the generation stays `0` on an append-only file, so the
  equality is trivially true; probe D shows identical poll counts and CPU within noise.

**Recommended follow-ups, none blocking:** teach `has_buffered_source` the generation conjunct (P4); everything
else carried forward unchanged from `PAIR_REVIEW_FINAL.md`.

---

## Note for the record

While setting up the pre-delta scratch copy, an `rm -rf "$SP/predelta"` was refused by a local safety guard
("rm target too broad"). No file was removed; I used `mktemp -d` instead and did not retry the command. Nothing
in the worktree, the DB, the offsets registry, or git history was modified by this review — the only file
written is this one.

DONE_D1_POST_R5_REVIEW
