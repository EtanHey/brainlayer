# Spotlight Exclusion Migration Runbook

Status: **READY — execute only in an approved writer-stop window.** Never overlap this operation
with a backup, release, or another write-heavy maintenance job. Exact workstation scheduling and
operator coordination belong in the restricted maintenance record, not this repository.

## Why `.metadata_never_index`

BrainLayer uses a marker in each high-churn root. On macOS 26.5.1 (25F80), a disposable recursive
`mdimport -i` probe gave a file below `excluded.noindex` normal `kMDItemFSName` and
`kMDItemContentType` attributes. The equivalent file below a directory containing
`.metadata_never_index` retained null attributes. `mdimport -t` is useful importer evidence but not
an exclusion verdict: its manual says test imports return attributes without writing the Spotlight
index.

[Apple's documented Search Privacy UI](https://support.apple.com/guide/mac-help/mchl1bb43b84/mac)
remains the fallback if a post-migration marker probe fails. It is not the primary mechanism
because setup must work unattended.

## Complete high-churn inventory

| Excluded root | Covered paths |
| --- | --- |
| `$HOME/.local/share/brainlayer` | `brainlayer.db`, `brainlayer.db-wal`, `brainlayer.db-shm`; sqlite-vec data and future vector sidecars; `chromadb`, `chromadb.backup`, `style`, `storage`, `experiments`; `enrichment_checkpoints.db`, `reembed_bgem3_checkpoint.json`, `enrichment-scratch`; `prompts`; `offsets.json`, watcher/drain/T3/health state, pause sentinel, pending-store files; `backups`, `jsonl-backups`; `logs` |
| active `BRAINLAYER_DB` parent, when outside the canonical root | overridden DB, WAL, SHM, and adjacent vector sidecars; setup marks both this parent and the canonical root |
| `$HOME/.brainlayer` | `queue`; `quarantine` including stale-queue quarantine; `logs` including drain logs; runtime repository state |
| `$HOME/Library/Logs/brainlayer` | launchd stdout/stderr and BrainBar runtime logs |
| `$HOME/.brainlayer-p0-counter` | longitudinal counter logs |

`/tmp` locks, sockets, and pidfiles are already outside Spotlight's normal user-content scope.
`~/.config/brainlayer` and `~/.brainlayer-brain` are deliberately not excluded: they are low-churn
configuration and user-facing exports.

## Preconditions

1. Confirm the merged release containing this runbook is installed.
2. Confirm the current time is inside the approved window and no backup or release job overlaps it.
3. Confirm no staging path below exists. If one exists, stop and inspect it; never overwrite it.
4. Record `brainlayer status --json`, `brainlayer doctor --json`, DB/WAL/SHM sizes, queue count, and
   loaded `com.brainlayer.*` labels. Resolve and record the active DB parent from `BRAINLAYER_DB`;
   when it differs from `$HOME/.local/share/brainlayer`, add it as a fifth migration root.
5. Keep the terminal open until restart and the real store/search probe both pass.

Staging paths:

```text
$HOME/.local/share/brainlayer.spotlight-migration-staging
$HOME/.brainlayer.spotlight-migration-staging
$HOME/Library/Logs/brainlayer.spotlight-migration-staging
$HOME/.brainlayer-p0-counter.spotlight-migration-staging
<active-BRAINLAYER_DB-parent>.spotlight-migration-staging  # only when outside the canonical root
```

## Execute in one stop window

### 1. Reduce the live WAL before the stop

This first checkpoint is deliberately non-destructive and may report busy. Do not proceed to moves
from this result alone.

```bash
brainlayer wal-checkpoint --mode PASSIVE --json
```

### 2. Stop every writer, scheduler, and self-healer

Disable/kick out the installed `com.brainlayer.*` labels as a group, including BrainBar daemon,
watch, drain, hotlane, enrichment, index, T3 ingest, maintenance, repair, decay, WAL checkpoint,
backup jobs, watchdogs, and the P0 counter. Record the exact loaded-label list before stopping it so
only that list is restored. Verify with both `launchctl print` and `pgrep -fal` that no BrainLayer
Python, BrainBar daemon, watcher, drain, hotlane, or enrichment process remains.

Do not let a watchdog restart a writer between this step and the final checkpoint.

### 3. Checkpoint after writers are stopped

```bash
brainlayer wal-checkpoint --mode TRUNCATE --retry-busy --json
```

Require `busy: false`, a zero-byte or absent `brainlayer.db-wal`, and no process holding
`brainlayer.db`, `brainlayer.db-wal`, or `brainlayer.db-shm` (`lsof` must return no writer). If any
condition fails, keep services stopped and investigate; do not move the tree.

### 4. Stage, mark, and restore each canonical tree

For each of the four canonical roots, plus the active `BRAINLAYER_DB` parent when it is distinct,
perform a same-filesystem rename to its exact staging path, create
`.metadata_never_index` inside the staged tree, then atomically rename the same tree back to its
canonical path. This preserves the root's mode, ownership, ACLs, extended attributes, and children;
the marker is already present before the canonical path becomes visible again. Never copy the tree.

Example for the canonical data root; repeat with the exact runtime/log/counter pairs listed above
and the recorded override parent/staging pair when applicable:

```bash
set -euo pipefail
brainlayer_data_root="${HOME:?}/.local/share/brainlayer"
brainlayer_stage_root="${HOME:?}/.local/share/brainlayer.spotlight-migration-staging"
test -d "$brainlayer_data_root"
test ! -e "$brainlayer_stage_root"
mv "$brainlayer_data_root" "$brainlayer_stage_root"
touch "$brainlayer_stage_root/.metadata_never_index"
test -f "$brainlayer_stage_root/.metadata_never_index"
mv "$brainlayer_stage_root" "$brainlayer_data_root"
```

Before each `mv`, resolve and print both paths, assert the source is the expected canonical root,
assert the staging target does not exist, and confirm both parents are on the same filesystem.
Apply this recovery state machine before continuing any root:

- canonical exists, staging absent: expected; continue;
- canonical missing, staging exists: restore with an atomic staging-to-canonical rename, verify the
  tree, and restart this root's preflight from the beginning;
- both exist: duplicate/conflict; stop without moving or merging either tree;
- both absent: for the canonical/override data root, stop because data is missing. For an optional
  runtime/log/counter root, first prove from the preflight label and path inventory that it was never
  created and never held data; record that evidence, create the exact canonical directory with mode
  `0700`, create and verify `.metadata_never_index`, and do not perform a staging rename for it.

This same state machine is the recovery procedure if the shell or machine stops between either
rename. Never infer completion from the presence of only one path without inspecting it.

### 5. Re-run setup and verify paths before restart

```bash
brainlayer setup --no-launchd
brainlayer doctor --json
```

Doctor may still report daemon liveness while services are intentionally stopped, but it must not
report `spotlight_indexing_enabled`. Confirm all four canonical marker files and the conditional
override marker exist, and that the DB, WAL, SHM, queue, prompt, scratch, vector, and log paths still
resolve to their pre-stop locations. No symlink or environment rewrite is expected for the marker
design; if an override is present, verify it points into a marked ancestor before restart.

### 6. Restart exactly the labels recorded in preflight

Bootstrap the recorded launchd plist set, then verify each expected label is loaded and each daemon
has a live PID. Do not restore a label that was deliberately disabled before the window.

Confirm none of the four canonical staging paths, or the conditional override staging path, remains
after its atomic rename back. If one remains, stop and investigate before restarting writers.

Run a real request through a fresh installed MCP client session (not an in-process Python test):

```text
brain_store(content="spotlight migration live probe", project="brainlayer-maintenance")
brain_search(query="spotlight migration live probe", project="brainlayer-maintenance")
```

Then run the CLI status gates:

```bash
brainlayer status --json
brainlayer doctor --json
```

### 7. Spotlight evidence

First establish that Spotlight is enabled for the containing volume:

```bash
mdutil -s /
```

Then create a unique, non-clobbering plain-text probe beneath the marked data root. Record these
outputs and retain the two printed variable values with the evidence:

```bash
spotlight_probe_path="$(mktemp "${HOME:?}/.local/share/brainlayer/spotlight-exclusion-live-probe.txt.XXXXXX")"
spotlight_probe_name="$(basename "$spotlight_probe_path")"
printf 'BrainLayer Spotlight exclusion live probe %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$spotlight_probe_path"
printf 'probe_path=%s\nprobe_name=%s\n' "$spotlight_probe_path" "$spotlight_probe_name"
mdimport -t -d1 "$spotlight_probe_path"
mdimport -i "$spotlight_probe_path"
mdls -name kMDItemFSName -name kMDItemContentType \
  "$spotlight_probe_path"
mdfind -onlyin "${HOME:?}/.local/share/brainlayer" \
  "kMDItemFSName == '$spotlight_probe_name'cd"
rm -- "$spotlight_probe_path"
```

Pass criteria: `mdutil` shows the containing volume is indexed; `mdimport -t` identifies the normal
importer; the real `mdimport -i` leaves both `mdls` attributes null; `mdfind` returns no probe path.
Repeat the real import/`mdls`/`mdfind` check once under each of the other three canonical marked
roots and under the active override parent when distinct. Remove only the exact probe files after
saving evidence.

If a marker-backed root receives metadata, keep BrainLayer stopped and add that exact root through
System Settings → Spotlight → Search Privacy, then repeat the probe. Do not disable Spotlight for
the whole volume.

## Rollback

Rollback is allowed only before writers restart. Keep all processes stopped. For each canonical
root and the conditional override root, first
apply the same four-state recovery table above: restore staging to canonical when only staging
exists, and stop on both-exist or both-absent states. Once canonical exists and staging is absent,
atomically rename canonical to staging, remove only `.metadata_never_index`, then atomically rename
the same tree back to canonical. This preserves the original root metadata. Verify DB/WAL/SHM and
queue counts before bootstrapping only the preflight label set. Never overwrite or merge paths.

## Completion record

Append the exact installed version/head, pre/post DB and WAL sizes, restored labels, real
store/search output, doctor output, all four canonical Spotlight probe results, and the conditional
override probe result to the maintenance source task. Only the live-window operator adds the
separate live-migration completion marker.
