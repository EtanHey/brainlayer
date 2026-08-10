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
   when it differs from `$HOME/.local/share/brainlayer`, add it as a fifth migration root only
   after validating that it is a dedicated BrainLayer-owned directory. Its basename must contain
   `brainlayer`, and it must not be `/`, `$HOME`, a mount root, an ancestor of the canonical data
   root, overlap another migration root, or contain unrelated data. If it fails any check, stop:
   never mark or rename that parent. Relocate the override DB and configuration into a dedicated
   directory in a separate approved migration before using this runbook.
5. For every migration root, provisionally record whether the root and `.metadata_never_index`
   already exist. The authoritative rollback baseline is refreshed after all writers are stopped.
6. Keep the terminal open until restart and the real store/search probe both pass.

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

After confirming all processes are stopped, re-record whether every root and its marker exist.
This writer-stop snapshot is the authoritative rollback baseline. If either state changed from the
provisional preflight inventory, explain and record the change before proceeding; never infer that
the migration created a marker observed before the first rename.

### 3. Checkpoint after writers are stopped

```bash
brainlayer wal-checkpoint --mode TRUNCATE --retry-busy --json
```

Require `busy: false`. Set `brainlayer_active_db` to the absolute active DB path recorded during
preflight—not a hardcoded canonical path—and require its WAL to be zero-byte or absent. Check the
active DB plus its actual `-wal` and `-shm` sidecars; every existing file must have no `lsof`
holder:

```bash
brainlayer_active_db="<preflight-resolved-active-db>"
test "${brainlayer_active_db#/}" != "$brainlayer_active_db"
test ! -s "${brainlayer_active_db}-wal"
for brainlayer_db_file in \
  "$brainlayer_active_db" "${brainlayer_active_db}-wal" "${brainlayer_active_db}-shm"; do
  test ! -e "$brainlayer_db_file" || ! lsof -- "$brainlayer_db_file"
done
```

If any condition fails, keep services stopped and investigate; do not move any tree.

### 4. Stage, mark, and restore each canonical tree

For each of the four canonical roots, plus the active `BRAINLAYER_DB` parent when it is distinct
and passed the dedicated-directory validation above, perform a same-filesystem rename to its exact
staging path, create
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
test ! -L "$brainlayer_data_root"
brainlayer_expected_root="$(cd "${HOME:?}/.local/share" && pwd -P)/brainlayer"
brainlayer_expected_stage="$(cd "${HOME:?}/.local/share" && pwd -P)/brainlayer.spotlight-migration-staging"
test "$(realpath "$brainlayer_data_root")" = "$brainlayer_expected_root"
test ! -e "$brainlayer_stage_root" && test ! -L "$brainlayer_stage_root"
test "$(stat -f %d "$(dirname "$brainlayer_data_root")")" = \
  "$(stat -f %d "$(dirname "$brainlayer_stage_root")")"
mv "$brainlayer_data_root" "$brainlayer_stage_root"
test -d "$brainlayer_stage_root" && test ! -L "$brainlayer_stage_root"
test "$(realpath "$brainlayer_stage_root")" = "$brainlayer_expected_stage"
touch "$brainlayer_stage_root/.metadata_never_index"
test -f "$brainlayer_stage_root/.metadata_never_index"
test ! -e "$brainlayer_data_root" && test ! -L "$brainlayer_data_root"
test "$(stat -f %d "$(dirname "$brainlayer_stage_root")")" = \
  "$(stat -f %d "$(dirname "$brainlayer_data_root")")"
mv "$brainlayer_stage_root" "$brainlayer_data_root"
test -d "$brainlayer_data_root" && test ! -L "$brainlayer_data_root"
test "$(realpath "$brainlayer_data_root")" = "$brainlayer_expected_root"
test -f "$brainlayer_data_root/.metadata_never_index"
test ! -e "$brainlayer_stage_root" && test ! -L "$brainlayer_stage_root"
```

Before each `mv`, resolve and print both paths, reject symbolic links (including dangling target
links), and confirm both parents are on the same filesystem. For the first move, assert the source
resolves to the preflight-recorded canonical root and staging is absent. For the restore move,
assert the source resolves to the exact staging root and canonical is absent. `test -d` alone is
not sufficient because it accepts a directory symlink and the subsequent `touch` could write
through that link.
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

Confirm none of the four canonical staging paths, or the conditional override staging path,
remains. Apply the recovery state machine before running setup if any staging path exists; setup
must never create a fresh canonical tree beside staged data.

```bash
brainlayer setup --no-launchd
brainlayer doctor --json
```

Doctor may still report daemon liveness while services are intentionally stopped, but it must not
report `spotlight_indexing_enabled`. Confirm all four canonical marker files and the conditional
override marker exist, and that the DB, WAL, SHM, queue, prompt, scratch, vector, and log paths still
resolve to their pre-stop locations. No symlink or environment rewrite is expected for the marker
design; if an override is present, verify it points into a marked ancestor before restart.

### 6. Spotlight evidence while writers remain stopped

First establish that Spotlight is enabled for the containing volume of every migration root. For
each canonical root and the validated distinct override root, resolve the filesystem mount point,
record the root-to-volume mapping, and query that mount point directly:

```bash
brainlayer_probe_root="${HOME:?}/.local/share/brainlayer"
brainlayer_probe_volume="$(stat -f %m "$brainlayer_probe_root")"
printf 'root=%s\nvolume=%s\n' "$brainlayer_probe_root" "$brainlayer_probe_volume"
mdutil -s "$brainlayer_probe_volume"
```

Repeat this check for every migrated root. Do not infer an override root's Spotlight state from
`mdutil -s /`: an override may reside on another volume. Every containing volume must report that
indexing is enabled before the exclusion probe can be interpreted.

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

### 7. Restart exactly the labels recorded in preflight

Only after every root passes Step 6, bootstrap the recorded launchd plist set, then verify each
expected label is loaded and each daemon has a live PID. Do not restore a label that was
deliberately disabled before the window.

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

## Rollback

Rollback is allowed only before writers restart. Keep all processes stopped. For each root that
existed in the writer-stop baseline, restore staging to canonical when only staging exists, stop on
both-exist, and treat both-absent as data loss. Once canonical exists and staging is absent,
atomically rename canonical to staging. Consult the writer-stop marker record and remove
`.metadata_never_index` only when this execution created it; preserve every marker that was already
present. Then atomically rename the same tree back to canonical. This preserves the original root
metadata.

For an optional root recorded as absent at writer-stop, both-absent is the successfully restored
state and must stay absent. If the migration created its canonical tree, require staging to be
absent and prove the tree contains only the marker plus known setup-created empty directories.
Remove only the exact execution-created marker file, then remove those known empty directories and
the root with exact, non-recursive `rmdir` operations; stop on any unexpected or nonempty entry.
Never apply the forward absent/absent rule during rollback.

Verify DB/WAL/SHM and queue counts before bootstrapping only the preflight label set. Never
overwrite or merge paths.

## Completion record

Append the exact installed version/head, pre/post DB and WAL sizes, restored labels, real
store/search output, doctor output, all four canonical Spotlight probe results, and the conditional
override probe result to the maintenance source task. Only the live-window operator adds the
separate live-migration completion marker.
