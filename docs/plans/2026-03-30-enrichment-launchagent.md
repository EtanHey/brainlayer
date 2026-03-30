# Enrichment LaunchAgent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Install a macOS LaunchAgent that runs BrainLayer realtime Gemini enrichment automatically at login/boot with install-time API key resolution, low priority, stable logging, and simple start/stop commands.

**Architecture:** Reuse the existing `scripts/launchd/com.brainlayer.enrichment.plist` template and `scripts/launchd/install.sh` flow instead of adding a second launcher path. The plist will run Python directly and call `brainlayer.enrichment_controller.enrich_realtime`, while the installer will resolve `GOOGLE_API_KEY` once from the environment or `~/.zshrc`, write the rendered plist into `~/Library/LaunchAgents/`, and expose simple `load` / `unload` convenience actions for the enrichment agent.

**Tech Stack:** Python, launchd plist templates, bash installer script, pytest.

---

### Task 1: Tighten LaunchAgent contract with failing tests

**Files:**
- Modify: `tests/test_enrichment_controller.py`
- Test: `tests/test_enrichment_controller.py`

**Step 1: Write the failing tests**
- Assert the enrichment plist uses Python to call `enrich_realtime`.
- Assert the plist uses `Nice=10`.
- Assert the plist logs to `~/Library/Logs/brainlayer-enrichment.log`.
- Assert the installer script targets `~/Library/LaunchAgents/` and includes enrichment `load` / `unload` helpers.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_enrichment_controller.py -k 'enrichment_plist or launchagent' -v`

**Step 3: Implement the minimal template/script updates**
- Update plist placeholders and launch command.
- Update installer commands and API key resolution flow.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_enrichment_controller.py -k 'enrichment_plist or launchagent' -v`

### Task 2: Implement install-time API key resolution and load/unload helpers

**Files:**
- Modify: `scripts/launchd/install.sh`
- Modify: `scripts/launchd/com.brainlayer.enrichment.plist`

**Step 1: Resolve `GOOGLE_API_KEY` from env or `~/.zshrc`**
- Keep install-time substitution only.
- Fail clearly if no key can be resolved for enrichment install.

**Step 2: Add rendered install target + helper actions**
- Ensure install destination is `~/Library/LaunchAgents/com.brainlayer.enrichment.plist`.
- Add convenience actions for loading and unloading the enrichment LaunchAgent.

**Step 3: Keep launchd behavior aligned with requirements**
- `RunAtLoad=true`
- low priority via `Nice=10`
- log to `~/Library/Logs/brainlayer-enrichment.log`

### Task 3: Verify end-to-end and publish

**Files:**
- Modify: `docs/configuration.md`
- Modify: `~/Gits/orchestrator/collab/native-apps-march30.md`

**Step 1: Run focused tests**

Run: `pytest tests/test_enrichment_controller.py -k 'enrichment_plist or launchagent' -v`

**Step 2: Run broader repo verification**

Run: `pytest`

**Step 3: Install and exercise locally**

Run:
- `bash scripts/launchd/install.sh enrichment`
- `bash scripts/launchd/install.sh load enrichment`
- `launchctl list | grep com.brainlayer.enrichment`
- `bash scripts/launchd/install.sh unload enrichment`

**Step 4: Commit, push, and open PR**

Run:
- `git add ...`
- `cr review --plain`
- `git commit -m "feat: install enrichment launch agent"`
- `git push -u origin codex/p4-enrichment-launchagent`
- `gh pr create ...`
