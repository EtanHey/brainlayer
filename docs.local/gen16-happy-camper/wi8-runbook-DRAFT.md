# Happy Camper 2nd Account Runbook Draft

Status: DRAFT - depends on PR #487 deployment plus a real-cmux BASIC split re-run.

## Purpose

Set up the Happy Camper macOS account so BrainLayer memory is isolated by the account's own database, then apply within-DB role scoping for day-to-day agents.

Etan's OF-3 ruling: cross-account isolation is physical. The new macOS account gets its own BrainLayer DB and MCP wiring, so it cannot reach the current user's DB unless explicitly pointed at it.

## Hard Gates

1. Do not point the Happy Camper account at `~/.local/share/brainlayer/brainlayer.db` from the main account.
2. Do not share the main account's MCP socket path with the Happy Camper account.
3. Load a BrainLayer build that includes request-scoped `consumer` support before relying on role separation inside one DB.
4. Run the real-cmux BASIC split gate after loading that build:
   - Orchestrator call with `consumer="orchestrator"` sees repo A, repo B, and null-project user-local rows.
   - Worker call with `consumer="worker"` and its project sees only its allowed project set.
5. Only after BASIC passes, run the lead/worktree gate:
   - Lead sees its main repo plus all configured worktrees, plus configured parallel repos.
   - Worker on main sees main only.
   - Worker on a worktree sees that worktree plus main.

## Account Setup

1. Create the second macOS account through System Settings.
2. Log into the new account directly, not through the existing account's shell.
3. Install or link the needed repos inside the new account home.
4. Confirm the new account resolves its BrainLayer DB independently:
   ```bash
   python3 - <<'PY'
   from brainlayer.paths import get_db_path
   print(get_db_path())
   PY
   ```
   Expected shape: `/Users/<happy-camper-user>/.local/share/brainlayer/brainlayer.db`.
5. Initialize/index only the Happy Camper account's intended data.
6. Confirm the main account's DB is not referenced by environment:
   ```bash
   env | grep -E '^BRAINLAYER_DB=|mcplayer|brainlayer'
   ```

## Role Wiring

Every shared-socket BrainLayer search call must carry the role explicitly. Process-level `BRAINLAYER_CONSUMER` is a fallback only; it cannot differentiate multiple agents sharing one MCP server.

Required request roles:

- Orchestrator: pass `consumer="orchestrator"`.
- Lead: pass `consumer="lead"` plus the lead repo project.
- Worker on main: pass `consumer="worker"` plus the main repo project.
- Worker on worktree: pass `consumer="worker"` plus the worktree project.
- Coach/personal lane: pass `consumer="coach"`.

Until an automatic injection layer exists, put this in each agent boot brief and role-specific MCP wrapper:

```text
When calling BrainLayer search/recall, always include consumer="<role>".
For workers and leads, include the current project name.
```

## Worktree Config

Configure worktree relationships in `~/.config/brainlayer/scopes.yaml`:

```yaml
scopes:
  ~/Gits/repo-a: repo-a
  ~/Gits/repo-a.wt/feature-x: repo-a.worktree.feature-x
  ~/Gits/repo-b: repo-b

worktrees:
  repo-a.worktree.feature-x: repo-a

lead_parallel_projects:
  repo-a:
    - repo-b
```

Semantics:

- `worktrees` maps worktree project IDs to their main repo project ID.
- A lead scoped to `repo-a` sees `repo-a` plus all worktrees mapped to `repo-a`.
- If `lead_parallel_projects.repo-a` includes `repo-b`, the lead also sees `repo-b` and worktrees mapped to `repo-b`.
- A worker scoped to `repo-a.worktree.feature-x` sees `repo-a.worktree.feature-x` and `repo-a`.
- A worker scoped to `repo-a` sees `repo-a` only.

## Born-Bitemporal Check

After the new account starts writing memories, sample recent rows and confirm bitemporal fields are present where the schema supports them:

```bash
sqlite3 ~/.local/share/brainlayer/brainlayer.db \
  "SELECT id, project, created_at, valid_at, invalid_at FROM chunks ORDER BY rowid DESC LIMIT 10;"
```

Expected:

- `created_at` is populated.
- `valid_at` is populated for born-bitemporal writes after the bitemporal migration.
- `invalid_at` is null for current facts.

## Dry-Run Commands

Run the deterministic proof harness on the target build:

```bash
PYTHONPATH=src python3 -m brainlayer.isolation_proof \
  --db /tmp/happy-camper-isolation-proof.db \
  --json
```

Expected output:

- `worker-repo-a`: `repo-a-main-proof`
- `orchestrator`: all seeded IDs
- `coach`: `personal-checkpoint-proof`, `null-user-local-proof`
- `lead-repo-a`: `repo-a-main-proof`, `repo-a-worktree-proof`, `repo-b-main-proof`, `repo-b-worktree-proof`
- `worker-repo-a-main`: `repo-a-main-proof`
- `worker-repo-a-worktree`: `repo-a-main-proof`, `repo-a-worktree-proof`

Then run the real-cmux gate with the live MCP server loaded from the same build. Paste the observed `brain_search` outputs into the collab before merging.

## Current Open Items

1. PR #487 must be reviewed and loaded before live real-agent verification can be meaningful.
2. The role injection is currently explicit per request. A future wrapper or launcher layer should auto-fill `consumer` so agents cannot forget it.
3. The Happy Camper account can proceed because account-level isolation is physical, but within-DB role scoping should not be called fully live-green until the post-deploy real-cmux gates pass.
