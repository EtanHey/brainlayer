# Tier-0 Meta-Watchdog Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a pure `/bin/sh` launchd watchdog that detects, alerts on, and heals a dead or stale BrainLayer health-check.

**Architecture:** A standalone POSIX-shell script checks the health-check launchd label and state-file mtime, fans bounded alerts out to a local log, macOS notification, and fail-open HTTP endpoint, then bootstraps or kickstarts the health-check. `TIER0_DOMAIN` defaults to `gui/$(id -u)`; inspection uses `launchctl print "$TIER0_DOMAIN/$TIER0_LABEL"`, absent-label recovery uses `launchctl bootstrap "$TIER0_DOMAIN" "$TIER0_HEALTH_PLIST_PATH"`, and all recovery ends with `launchctl kickstart -k "$TIER0_DOMAIN/$TIER0_LABEL"`. The existing launchd installer copies the script to a stable per-user path and renders a plist that calls `/bin/sh` directly. Pytest drives four isolated drills using temporary paths and executable overrides.

**Tech Stack:** POSIX `/bin/sh`, macOS launchd/osascript/curl/stat, plist XML, pytest.

---

### Task 1: Prove the four silent-death drills

**Files:**
- Create: `tests/test_tier0_drills.py`
- Create: `scripts/tier0-watchdog.sh`

1. Add a test harness that creates fake `launchctl`, `osascript`, `curl`, and
   `stat` executables and runs `/bin/sh scripts/tier0-watchdog.sh` with only
   temporary paths and a deterministic epoch.
2. Add D1 for an unloaded label and assert alert events precede bootstrap and
   kickstart events.
3. Add D2 for a stale state file and assert alert events precede kickstart, with
   no bootstrap.
4. Add D3 with a hanging curl stub and a one-second alert-process timeout; assert
   the Tier-0 log and osascript still alert before kickstart.
5. Add D4 for a loaded label and fresh state and assert exit zero with no alert,
   bootstrap, or kickstart.
6. Add a separate missing-state regression that asserts alerting and direct
   kickstart without bootstrap; do not count it as a fifth exit-gate drill.
7. Assert the exact user-domain `print`, `bootstrap`, and `kickstart` argument
   forms in the applicable drills.
8. Assert the notify request uses `POST http://localhost:3847/notify`,
   `Content-Type: application/json`, a constant `title`/`body`/`source` payload,
   curl fail-on-HTTP-error behavior, and a bounded timeout.
9. Run `pytest tests/test_tier0_drills.py -v` and confirm RED because the script
   does not exist.
10. Implement the minimal POSIX-shell detection, bounded alert fan-out, and
    healing behavior, with quoted executable/path overrides and no Python
    dependency.
11. Re-run the drill suite and confirm GREEN.

### Task 2: Prove the launchd and installer contract

**Files:**
- Create: `scripts/launchd/com.brainlayer.tier0-watchdog.plist`
- Modify: `scripts/launchd/install.sh`
- Modify: `pyproject.toml`
- Modify: `tests/test_installable_build.py`

1. Add failing plist assertions for the Tier-0 label, 300-second interval,
   `RunAtLoad`, and exact `ProgramArguments` beginning with `/bin/sh` and never
   using the environment runner or Python.
2. Add a failing packaged-installer test that invokes the `tier0-watchdog`
   target with fake launchctl, then asserts the script is installed executable
   and the plist contains its stable installed path. Assert installer bootstrap
   and post-load print use the same `gui/$UID` domain.
3. Run the focused tests and confirm RED because the plist/target do not exist.
4. Add the plist, a dedicated installer path that copies the runtime script,
   and `tier0-watchdog` handling in install/all/remove/usage flows.
5. Include the runtime script in wheel and sdist packaging.
6. Re-run the focused installer and drill tests and confirm GREEN.

### Task 3: Verify and publish the worker PR

**Files:**
- Create: `docs.local/handoffs/BL-C-REPORT.md`

1. Run `pytest tests/test_tier0_drills.py tests/test_installable_build.py -v`.
2. Run `ruff check src/ tests/` and `ruff format --check src/ tests/`.
3. Run the full `pytest` suite without any production DB fixture.
4. Read the complete diff and every deliverable file, then record exact drill,
   lint, and suite evidence in the report with the install command
   `brainlayer setup --launchd --target tier0-watchdog`.
5. Run the bounded local CodeRabbit review, address blocking findings, commit,
   and push with `BRAINLAYER_PREPUSH_SCOPE=changed-only`.
6. Open a ready-for-review PR, append the single required channel line, request
   Codex/Cursor/Bugbot reviews, and inspect CI plus actionable feedback.
7. Stop at the worker boundary (PR plus addressed review feedback); the lead
   owns merge and the live install.
8. Store the implementation decision and milestone in BrainLayer.
