# Happy Camper 2nd Account Runbook (FINAL)

Status: **FINAL** — 2026-06-15. Backed by PR #487 (merged to `main`, merge commit `d56cc430c`). Per-consumer BrainLayer scoping — basic split + lead tier + worktree-awareness — is on `main`. The effectiveness gate PASSED (per-role usefulness proven with real MCP transport; see §Effectiveness Evidence).

## Purpose

Set up the Happy Camper macOS account so BrainLayer memory is isolated by the account's own database, then apply within-DB role scoping for day-to-day agents.

Etan's OF-3 ruling: cross-account isolation is **physical**. The new macOS account gets its own BrainLayer DB and MCP wiring, so it cannot reach the current user's DB unless explicitly pointed at it. Within-DB role scoping is a **quality win on top**, not the account's isolation guarantee.

## Effectiveness Evidence (what the gate proved, per role)

Gate = real MCP transport (per-request `consumer=`) against a SEPARATE test server with realistic seeded memories, plus the deterministic harness (`failures: []`). Independently re-verified by the lead (not paste-trusted). Result — is each role's view genuinely useful?

| role (`consumer=`) | sees | excludes | useful? |
| --- | --- | --- | --- |
| orchestrator | all repos + user-local | nothing | ✅ full cross-repo + personal context, no over-restriction |
| lead (+ repo) | its repo + all its worktrees + configured parallel repos | other repos, user-local | ✅ right repo picture incl. in-flight worktrees, no other-repo leak |
| worker on worktree (+ worktree project) | that worktree + main (trunk) | other repos, other worktrees, user-local | ✅ own WIP + trunk, no foreign noise |
| worker on main (+ main project) | main only | other repos, un-merged worktrees | ⚠️ useful (no foreign/unmerged noise) **but misses merged-worktree work that landed on main** — see §Known Gap |
| coach | personal / user-local | all repo rows | ✅ personal only |

## Hard Gates

1. Do NOT point the Happy Camper account at `~/.local/share/brainlayer/brainlayer.db` from the main account.
2. Do NOT share the main account's MCP socket path with the Happy Camper account.
3. The build with request-scoped `consumer` support is now on `main` (PR #487). **Deploying it to the LIVE shared server is a deliberate step that must FIRST wire `consumer="orchestrator"` (per-request) or `BRAINLAYER_CONSUMER=orchestrator` for existing orchestrator/lead sessions** — otherwise live sessions fail-closed to `worker` and over-restrict their own memory. Never hot-swap the live server without that wiring. (For a brand-new Happy Camper account with its own server, this is moot — wire roles from the start.)
4. Run the role gates after loading the build (see §Dry-Run): orchestrator sees all; worker sees only its allowed set.
5. Only after BASIC passes, confirm the lead/worktree gate: lead = repo + worktrees (+ parallel); worker-on-worktree = worktree + main; worker-on-main = main.

## Account Setup

1. Create the second macOS account through System Settings (**human gate: admin password / GUI**).
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

Every shared-socket BrainLayer search/recall call must carry the role explicitly. Process-level `BRAINLAYER_CONSUMER` is a fallback only; it cannot differentiate multiple agents sharing one MCP server.

**Interim (manual, available today):** each agent passes `consumer="<role>"` (+ project for worker/lead) on every call. Put this in each agent's boot brief / role-specific MCP wrapper:
```text
When calling BrainLayer search/recall, always include consumer="<role>".
For workers and leads, include the current project name.
```
Required request roles: orchestrator → `consumer="orchestrator"`; lead → `consumer="lead"` + lead repo project; worker on main → `consumer="worker"` + main repo project; worker on worktree → `consumer="worker"` + worktree project; coach → `consumer="coach"`.

**Robust (auto-injection via the spawn layer) — recommended follow-up (designed, not yet built):**
- **Injection point:** cmuxlayer `AgentEngine.spawnAgent`, immediately after `inferAgentRole` and before `buildLaunchCommand` (repo, cli, role, cwd/worktree path, and branch are all known there). Add a repoGolem helper fallback for direct shell launches outside cmux.
- **Role derivation:** explicit `brainlayer_consumer` override wins; else orchestrator launcher → `orchestrator`; non-orchestrator Claude / `role=ic` → `lead`; Codex/Cursor / `role=worker` → `worker`; coach launcher/repo → `coach`. (Keep cmux placement-role separate from the BrainLayer consumer.)
- **Delivery (NOT env-only):** the shared socat socket means an agent's env never reaches the already-running BrainLayer server. Generate a per-agent MCP config that replaces the brainlayer command with a tiny **scoped proxy** which forwards stdio to `/tmp/mcplayer-brainlayer.sock` and injects the missing `consumer`/`project` into `brain_search`/`brain_recall` tool-call args. The per-request `consumer` (already shipped) remains the source of truth.
- **Worker main vs worktree:** compute once at spawn — if cwd resolves to the canonical repo path, `project=<repo>`; if cwd is a prepared worktree / repoGolem `-w` path, `project=<repo>.worktree.<slug(branch||basename(cwd))>` and ensure `scopes.yaml` maps that worktree project back to the root repo.
- **Effort:** ~1–2 days (cmux + direct repoGolem launches + MCP scoped proxy + tests) — bigger than a tiny follow-up but bounded. Recommend the proxy path; prompt/env-only injection is **not** acceptable for the shared-socket setup.

Until this ships, the interim manual rule above is load-bearing — agents that forget `consumer=` fall back to the shared process env.

## Known Gap — worker-on-main + merged worktrees (Etan's refinement, deferred)

Current `main` implements **worker-on-main = main-only**. Etan's refinement: a worker on main should ALSO see memory of PAST worktrees that were **merged into** main ("merged stuff needs to be where it was merged to"), but NOT un-merged / parallel worktrees. Target: **worker-on-main = main + merged-worktree memory**.

Gate evidence: worker-on-main correctly excluded the other repo and the un-merged worktree, **but also missed a seeded merged chunk (`rate-limiter`) that had landed on main** — a real "needs-it / missing-it" gap.

Design options (decide before implementing):
- **Query-time git detection** — `git merge-base --is-ancestor <wt-branch> main` / `git branch --merged main`. Always-correct, but adds a git call per search and needs the worktree→branch mapping.
- **Merge-time re-attribution** — re-attribute worktree chunks to main at merge. Zero query cost, but needs a merge hook + a backfill for already-merged worktrees, and must preserve original worktree provenance.

NOT account-blocking (OF-3: cross-account isolation is physical). Next iteration. Tester probe to add: worker-on-main SEES a merged-worktree chunk, does NOT see an un-merged worktree chunk.

## Worktree Config

Configure worktree relationships in `~/.config/brainlayer/scopes.yaml`:
```yaml
scopes:
  ~/Gits/repo-a: repo-a
  ~/Gits/repo-a.wt/feature-x: repo-a.worktree.feature-x
  ~/Gits/repo-b: repo-b

worktrees:
  repo-a.worktree.feature-x: repo-a

lead_parallel_projects:   # alias: parallel_repos
  repo-a:
    - repo-b
```
Semantics:
- `worktrees` maps worktree project IDs to their main repo project ID.
- A lead scoped to `repo-a` sees `repo-a` plus all worktrees mapped to `repo-a`.
- If `lead_parallel_projects.repo-a` includes `repo-b`, the lead also sees `repo-b` and worktrees mapped to `repo-b`.
- A worker scoped to `repo-a.worktree.feature-x` sees `repo-a.worktree.feature-x` and `repo-a`.
- A worker scoped to `repo-a` sees `repo-a` only.

## Born-Bitemporal (opt-in)

After the new account's DB is initialized, apply the bitemporal migration so writes are born-bitemporal:
```bash
python3 - <<'PY'
import sqlite3
from brainlayer.paths import get_db_path
from brainlayer.bitemporal import apply_bitemporal_migration
conn = sqlite3.connect(str(get_db_path()))
apply_bitemporal_migration(conn)
conn.commit(); conn.close()
print("bitemporal migration applied")
PY
```
Then sample recent rows and confirm bitemporal fields. The migration adds `valid_from`, `invalid_at`, `sys_period_start`, `sys_period_end` (all nullable/defaulted; **note the column is `valid_from`, not `valid_at`**):
```bash
sqlite3 ~/.local/share/brainlayer/brainlayer.db \
  "SELECT id, project, created_at, valid_from, invalid_at FROM chunks ORDER BY rowid DESC LIMIT 10;"
```
Expected: `created_at` populated for all rows; `valid_from` populated for writes made **after** the migration (NULL on rows that predate it — they are not retroactively born-bitemporal); `invalid_at` NULL for current facts. (Verified in dry-run on a copy of the proof DB: migration is idempotent; pre-existing rows show `valid_from`=NULL, `invalid_at`=NULL.)

## Dry-Run Commands

Run the deterministic proof harness on the target build (all roles in one command):
```bash
PYTHONPATH=src python3 -m brainlayer.isolation_proof \
  --db /tmp/happy-camper-isolation-proof.db \
  --json
```
Expected `visible_ids_by_probe` (from the merged all-role harness; `failures: []`):
- `worker` / `worker-repo-a`: `repo-a-main-proof`
- `orchestrator`: all seeded IDs
- `coach`: `personal-checkpoint-proof`, `null-user-local-proof`
- `lead-repo-a`: `repo-a-main-proof`, `repo-a-worktree-proof`, `repo-b-main-proof`, `repo-b-worktree-proof`
- `worker-repo-a-main`: `repo-a-main-proof`
- `worker-repo-a-worktree`: `repo-a-main-proof`, `repo-a-worktree-proof`

Then (when the live server runs this build) run the real-agent role gate and paste observed `brain_search` outputs before relying on within-DB separation.

## Open Items

1. Deploy the merged `main` build to the live shared server — paired with orchestrator/lead consumer wiring first (deliberate; not account-blocking).
2. Auto-injection of `consumer=` via the spawn layer (robust role-wiring) — see §Role Wiring.
3. Merged-worktree refinement for worker-on-main — see §Known Gap.
