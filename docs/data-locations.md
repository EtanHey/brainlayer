# BrainLayer Data Locations

> Single source of truth for where all data lives, where it moved from, and the archive strategy.

## Active Data

| What | Path | Size | Notes |
|------|------|------|-------|
| **Main database** | `~/.local/share/zikaron/zikaron.db` | ~3.8 GB | 268K+ chunks, sqlite-vec + FTS5 |
| **knowledge.db** | `~/.local/share/zikaron/knowledge.db` | symlink | Points to zikaron.db |
| **Current sessions** | `~/.claude/projects/{encoded-path}/*.jsonl` | ~805 files | Claude Code session transcripts |
| **Archived sessions** | `~/.claude-archive/{project-id}/archive-{timestamp}/` | 1.2 GB | Moved by session-archiver |

## Path Resolution

BrainLayer resolves the database path in this order (see `src/brainlayer/paths.py`):

1. **`BRAINLAYER_DB` env var** — explicit override
2. **`~/.local/share/zikaron/zikaron.db`** — legacy path (if exists, use it)
3. **`~/.local/share/brainlayer/brainlayer.db`** — canonical path (for fresh installs)

### Why the legacy path?

The project was originally called "Zikaron" and all data lives at the legacy path.
Renaming the 3.8 GB database is risky and unnecessary — the code resolves it automatically.
When users install BrainLayer fresh (no existing data), it uses the canonical path.

## Session Archiver

**Service:** `com.golems.session-archiver` (launchd, runs daily at 4am)
**Script:** `golems/packages/services/src/session-archiver.ts`

### How it works:

1. Scans `~/.claude/projects/` for all session JSONL files
2. Keeps last 7 days of active sessions per project
3. Moves older sessions to `~/.claude-archive/{project-id}/archive-{timestamp}/`
4. Writes `manifest.json` per batch (UUIDs, timestamps, sizes)
5. After BrainLayer indexes the archived sessions, the archiver cleans up verified copies

### Archive structure:

```
~/.claude-archive/
  golems/
    archive-2026-02-09T02-00-05/
      {uuid}.jsonl           # Archived session transcript
      {uuid}/                # Optional: subagent files
      manifest.json          # Batch metadata
    archive-2026-02-10T02-00-05/
      ...
  domica/
    ...
  songscript/
    ...
```

### Manifest format:

```json
{
  "archivedAt": "2026-02-09T02:00:05.123Z",
  "projectId": "golems",
  "originalPath": "/Users/janedev/Gits/golems",
  "sessions": [
    {
      "uuid": "abc123...",
      "originalMtime": "2026-02-07T15:30:00.000Z",
      "size": 524288,
      "hasSubdir": true,
      "firstMessageTimestamp": "2026-02-07T15:28:42.123Z",
      "gitBranch": "feature/some-branch"
    }
  ],
  "metadata": {
    "archiver_version": "1.1.0",
    "sessions_kept": 7,
    "total_archived": 12,
    "total_size_bytes": 6291456
  }
}
```

## iCloud Backups (Manual)

These are one-time manual backups, NOT session archives:

| Directory | What | When |
|-----------|------|------|
| `iCloud/golem-backup-2026-02-01-2245/` | Config files, LaunchAgents, ralph config | Feb 1, 2026 |
| `iCloud/golem-archives/2026-02-02/` | Old zikaron scripts, photorec recovery | Feb 2, 2026 |
| `iCloud/golem-backup-2026-02-16-0223/` | zikaron.db snapshot (pre-Vertex backfill) | Feb 16, 2026 |
| `iCloud/golems-backups/packages-zikaron-backup-20260219/` | packages/zikaron/ code before extraction | Feb 19, 2026 |
| `iCloud/Golems/` | Symlinks to plans | Feb 6, 2026 |

(iCloud base: `~/Library/Mobile Documents/com~apple~CloudDocs/`)

**None of these contain session JSONL files.** They're config, DB, and code backups.

## Historical: Data Migrations

### Repo path change (Jan-Feb 2026)

Repos moved from `~/Desktop/Gits/` to `~/Gits/`. This means:
- Old chunks reference `~/.claude/projects/-Users-janedev-Desktop-Gits-{repo}/`
- New chunks reference `~/.claude/projects/-Users-janedev-Gits-{repo}/`
- The old JSONL session files at the Desktop paths no longer exist

### Session archiver setup (Feb 9, 2026)

Before the archiver was set up, old sessions were manually deleted.
~160K chunks reference sessions that no longer exist anywhere.
These chunks are still searchable — they just don't have `created_at` timestamps.

### BrainLayer extraction (Feb 19, 2026)

Extracted from `golems/packages/zikaron/` to standalone `~/Gits/brainlayer/`.
Code moved, data stayed at `~/.local/share/zikaron/zikaron.db`.
`paths.py` handles the legacy path transparently.

## Vertex AI Batch Enrichment (Feb 17-18, 2026)

- 153,825 chunks submitted to Vertex AI batch prediction
- Results imported Feb 18 at 08:00 (135,865 chunks enriched)
- Job tracking: `scripts/backfill_data/vertex_jobs.json`
- Predictions stored in: `scripts/backfill_data/predictions/`

## Coverage Stats (as of Feb 19, 2026)

| Metric | Count | Percentage |
|--------|-------|------------|
| Total chunks | 268,864 | 100% |
| Have `created_at` | 107,935 | 40.1% |
| Missing `created_at` | 160,929 | 59.9% |
| Enriched | 144,146 | 53.6% |
| Enrichable but not enriched | 22,974 | 8.5% |
| Too small to enrich (<50 chars) | 101,744 | 37.8% |

The 160K chunks without `created_at` are from pre-archiver sessions whose JSONL files
were deleted. The chunks themselves are fully indexed and searchable — date filtering
just won't apply to them (they'll always be included in unfiltered searches).
