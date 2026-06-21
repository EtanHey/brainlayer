# BrainLayer — Robust Brew Layer Conformance

> Self-audit of BrainLayer against the cross-layer **Robust Brew Layer Standard**
> (`orchestrator/standards/robust-brew-layer.md`, owner: cmuxlayerClaude).
> Status as of **2026-06-21**, after the M1-clone (`happycampr`) friction sprint.
> Every ✅ below is verified either on the live M4 dev machine or the M1 clone
> end-to-end install (`brew install --cask brainbar` → working `1.3.0,1`).

## Headline (§0 / §0.1) — two-consumer split

BrainLayer is the standard's reference **two-consumer split**:

- **App / fleet (socket + daemon):** `BrainBarDaemon` serves `brain_search` over the
  **canonical socket `/tmp/brainbar.sock`** for the live BrainBar UI and for
  cross-repo fleet agents. This is the genuinely-needed daemon case (§0).
- **Agents (stdio MCP):** `brainlayer-mcp` (`brainlayer serve`) is a **stdio** server
  launched per-agent that talks to the SQLite store directly — *not* through the
  app's socket. This removes the agent path's dependency on the socket/daemon
  failure classes (7a/7b/7c) entirely.

### Write-path invariant (§0.1 hard requirement)

The store runs in **WAL mode**. The concurrency invariant is preserved as follows:

- **Reads** (`brain_search`, both DB-direct stdio and socket-served): safe under WAL —
  SQLite WAL permits concurrent readers alongside writers. Agent read connections
  open read-only.
- **Writes** (`brain_store`, watcher, enrichment): serialized through the
  **`com.brainlayer.drain` queue → single applying writer** (`docs/configuration.md`:
  "Drain queued writes as the single writer"), with `WAL` + `busy_timeout` retry on
  `SQLITE_BUSY` for any direct writer. The single-writer consistency assumption is
  thus held by the drain, not by the GUI daemon — so the DB-direct stdio agent MCP
  is a **reader**, never a competing writer.

This satisfies §0.1: consumers are documented (socket = app/fleet, stdio = agents),
and the write-path invariant is stated.

## Conformance matrix

| § | Class | Status | Evidence |
|---|---|---|---|
| 0 / 0.1 | stdio-preferred / split + write-path invariant documented | ✅ | This doc; `brainlayer-mcp` stdio + `BrainBarDaemon` socket; drain = single writer |
| 1 | account-rename self-heal (no interactive sudo) | ⚠️ **partial** | Cask `preflight` self-heals dangling `localaiengine`-tainted Caskroom symlinks (account-agnostic via `start_with?`, user-owned only, no sudo) — **M1-verified**. **Gap:** no standalone `brainlayer doctor --fix` CLI (the §1 MUST form). |
| 2 | sudo-free cask uninstall / upgrade | ✅ | brainbar installs **user** LaunchAgents (`gui/$uid`), no privileged helper → uninstall needs no sudo; `zap` stanza present; caveats document the two genuine sudo edge-cases (root-owned leftover after rename; clone-upgrade heuristic) |
| 3 | tap-add + trust documented | ✅ (shared runbook, orc-owned) | M1 install succeeded → tap trusted; brainbar resolves as `etanhey/layers/brainbar` |
| 4 | venv-python everywhere | ✅ | Daemon resolves the formula venv python via `resolveHomebrewVenvPython` (PR #528) before any bare-python fallback; hooks self-resolve src via `sys.path.insert` + degrade gracefully under `except` guards (never hard-fail on bare `python3`) |
| 5 | headless serve + daemon integrity | ✅ | **5a** canonical `/tmp/brainbar.sock`, zero stale `mcplayer-*` aliases in the bundle. **5b** `BrainBarDaemon` LaunchAgent (`RunAtLoad`/`KeepAlive`) brings the socket up headless. **5c** cask `postflight` re-bootstraps the daemon+UI LaunchAgents on reinstall (PR #525 / cask) — **M1-verified auto-bootstrap, PID assigned, no `setup --launchd` needed**. **5d** daemon is its own executable + LaunchAgent, not a GUI-app child |
| 6 | declared deps + correct cask names + adopt | ✅ | Agent MCP is **stdio** → structurally immune to the socat 3-address footgun; fleet socket access verified **connecting** on M1 (`brain_search` 5 results, no `-32000`) |
| 7 | read-only commands stay read-only | ⚠️ **review** | Action-verbs mutate appropriately (`reconcile-launchd`, `watch`, `wal-checkpoint`, `serve`); `health-check` is read-only. **Open:** `setup` ("create brainlayer.env…") and `init` (interactive wizard) mutate — but both are the **documented install entrypoints** (formula caveat: "Run setup after install"), not surprise-mutating inspect-verbs. Needs standard-owner ruling on whether documented-`setup`-as-install must be renamed `install`. |

## Remaining work

1. **§1 — `brainlayer doctor` / `doctor --fix` CLI (the one real MUST gap).**
   The *friction* is functionally fixed and M1-verified (cask preflight heals the
   reinstall case), but the standard prescribes a standalone non-interactive CLI:
   detect stale Caskroom artifacts (account-agnostic), `--fix` heals without bare
   sudo (`sudo -n` probe), root-owned → `STOP:` + distinct exit code, idempotent,
   SHOULD scan the sibling manifest (brainbar, karabiner-elements, aldente,
   wispr-flow, codex). The detection logic already exists in the cask preflight and
   would be lifted into the CLI. **Scoped follow-on; the M1 already works without it.**

2. **§7 — read-only ruling.** Confirm with the standard owner whether `setup`/`init`
   may remain the documented install entrypoints or must move mutation behind an
   explicit `install` subcommand.

Everything else (§0, §2, §3, §4, §5, §6) is conformant and M1-clone-verified.
