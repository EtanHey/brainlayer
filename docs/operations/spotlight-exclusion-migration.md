# Spotlight Exclusion Migration Runbook

Status: **READY — execute only in the lead-scheduled writer-stop window.** This runbook was written
for the canonical Etan workstation paths. It must not run during the 03:17 backup window. Prefer the
Tuesday 20:00 combined stop window so this migration and the other scheduled maintenance operation
share one writer stop.

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
| `/Users/etanheyman/.local/share/brainlayer` | `brainlayer.db`, `brainlayer.db-wal`, `brainlayer.db-shm`; sqlite-vec data and future vector sidecars; `chromadb`, `chromadb.backup`, `style`, `storage`, `experiments`; `enrichment_checkpoints.db`, `reembed_bgem3_checkpoint.json`, `enrichment-scratch`; `prompts`; `offsets.json`, watcher/drain/T3/health state, pause sentinel, pending-store files; `backups`, `jsonl-backups`; `logs` |
| `/Users/etanheyman/.brainlayer` | `queue`; `quarantine` including stale-queue quarantine; `logs` including drain logs; runtime repository state |
| `/Users/etanheyman/Library/Logs/brainlayer` | launchd stdout/stderr and BrainBar runtime logs |
| `/Users/etanheyman/.brainlayer-p0-counter` | longitudinal counter logs |

`/tmp` locks, sockets, and pidfiles are already outside Spotlight's normal user-content scope.
`~/.config/brainlayer` and `~/.brainlayer-brain` are deliberately not excluded: they are low-churn
configuration and user-facing exports.

## Preconditions

1. Confirm the merged release containing this runbook is installed.
2. Confirm the current time is inside the approved window and not near 03:17.
3. Confirm no staging path below exists. If one exists, stop and inspect it; never overwrite it.
4. Record `brainlayer status --json`, `brainlayer doctor --json`, DB/WAL/SHM sizes, queue count, and
   loaded `com.brainlayer.*` labels.
5. Keep the terminal open until restart and the real store/search probe both pass.

Staging paths:

```text
/Users/etanheyman/.local/share/brainlayer.spotlight-migration-staging
/Users/etanheyman/.brainlayer.spotlight-migration-staging
/Users/etanheyman/Library/Logs/brainlayer.spotlight-migration-staging
/Users/etanheyman/.brainlayer-p0-counter.spotlight-migration-staging
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

For each of the four roots, perform a same-filesystem rename to its exact staging path, recreate the
canonical root, create `.metadata_never_index`, then move every child (including dotfiles) back.
Never copy across filesystems and never create the marker after children return.

Example for the data root; repeat with the exact runtime/log/counter pairs listed above:

```bash
mv /Users/etanheyman/.local/share/brainlayer \
  /Users/etanheyman/.local/share/brainlayer.spotlight-migration-staging
mkdir /Users/etanheyman/.local/share/brainlayer
touch /Users/etanheyman/.local/share/brainlayer/.metadata_never_index
find /Users/etanheyman/.local/share/brainlayer.spotlight-migration-staging \
  -mindepth 1 -maxdepth 1 -exec mv {} /Users/etanheyman/.local/share/brainlayer/ \;
```

Before each `mv`, resolve and print both paths, assert the source is the expected canonical root,
assert the staging target does not exist, and confirm both parents are on the same filesystem.

### 5. Re-run setup and verify paths before restart

```bash
brainlayer setup --no-launchd
brainlayer doctor --json
```

Doctor may still report daemon liveness while services are intentionally stopped, but it must not
report `spotlight_indexing_enabled`. Confirm all four marker files exist and the canonical DB, WAL,
SHM, queue, prompt, scratch, vector, and log paths still resolve to their pre-stop locations. No
symlink or environment rewrite is expected for the marker design; if an override is present, verify
it points into a marked ancestor before restart.

### 6. Restart exactly the labels recorded in preflight

Bootstrap the recorded launchd plist set, then verify each expected label is loaded and each daemon
has a live PID. Do not restore a label that was deliberately disabled before the window.

Keep the now-empty staging directories until every verification below passes. Afterward, verify
each is empty and remove only those four exact directories.

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

Then create a unique plain-text probe beneath the marked data root. Record these outputs:

```bash
mdimport -t -d1 /Users/etanheyman/.local/share/brainlayer/spotlight-exclusion-live-probe.txt
mdimport -i /Users/etanheyman/.local/share/brainlayer/spotlight-exclusion-live-probe.txt
mdls -name kMDItemFSName -name kMDItemContentType \
  /Users/etanheyman/.local/share/brainlayer/spotlight-exclusion-live-probe.txt
mdfind -onlyin /Users/etanheyman/.local/share/brainlayer \
  'kMDItemFSName == "spotlight-exclusion-live-probe.txt"cd'
```

Pass criteria: `mdutil` shows the containing volume is indexed; `mdimport -t` identifies the normal
importer; the real `mdimport -i` leaves both `mdls` attributes null; `mdfind` returns no probe path.
Repeat the real import/`mdls`/`mdfind` check once under each of the other three marked roots. Remove
only the four exact probe files after saving evidence.

If a marker-backed root receives metadata, keep BrainLayer stopped and add that exact root through
System Settings → Spotlight → Search Privacy, then repeat the probe. Do not disable Spotlight for
the whole volume.

## Rollback

Rollback is allowed only before writers restart. Keep all processes stopped. For each root, move
every restored child except `.metadata_never_index` back into its still-present staging directory,
remove only the exact marker, remove the now-empty canonical directory, and rename staging back to
the canonical name. Verify DB/WAL/SHM and queue counts before bootstrapping only the preflight label
set. If either tree contains an unexpected duplicate, stop; never overwrite or merge it.

## Completion record

Append the exact installed version/head, pre/post DB and WAL sizes, restored labels, real
store/search output, doctor output, and all four Spotlight probe results to the maintenance source
task. Only the live-window operator adds the separate live-migration completion marker.
