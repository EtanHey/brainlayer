# Package Hotlane Worker Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make every supported BrainLayer installation ship, install, supervise, and verify the bounded hotlane embedding worker instead of generating a launchd job that points to a missing source-checkout file.

**Architecture:** Package the existing version-matched `hotlane_brainbar_daemon.py` beside the launchd installer, copy it during setup to `~/.local/lib/brainlayer/hotlane_brainbar_daemon.py`, and render the plist with that stable destination. This mirrors the established throughput-watchdog packaging pattern, survives Homebrew Cellar upgrades, keeps drain embedding disabled, and preserves the existing bounded batch-4 worker contract. Setup also waits for launchd to confirm the old job is absent before bootstrap; if another supervisor wins the reload race, setup accepts that outcome only after a fresh loaded-job check and the same stable-PID runtime gate.

**Tech Stack:** Python 3.11–3.13, Hatch/uv wheel builds, Bash launchd installer, plistlib, pytest, launchd, PyPI trusted publishing, Homebrew `homebrew-LAYERS`.

---

### Task 1: Prove the built-wheel and installer regression

**Files:**
- Modify: `tests/test_installable_build.py`
- Modify: `tests/test_engine_package_boundary.py`

**Step 1: Extend the package-boundary expectation**

Assert that wheel force-includes:

```python
assert (
    wheel_config["force-include"]["scripts/hotlane_brainbar_daemon.py"]
    == "brainlayer/launchd/hotlane_brainbar_daemon.py"
)
```

Also require `scripts/hotlane_brainbar_daemon.py` in the sdist `only-include` list.

**Step 2: Extend the real built-wheel test**

Add this assertion to the wheel ZIP listing:

```python
assert "brainlayer/launchd/hotlane_brainbar_daemon.py" in listing
```

**Step 3: Add a packaged-installer behavior test**

Create a fake packaged `brainlayer/launchd` directory, copy the launchd templates and hotlane daemon into it, provide fake `launchctl`, a fake executable `brainlayer`, and a complete environment file, then execute:

```bash
<packaged-launchd>/install.sh hotlane
```

Assert:

- `~/.local/lib/brainlayer/hotlane_brainbar_daemon.py` exists, is executable, and is byte-identical to the packaged daemon.
- The installed plist points argument 2 at that stable file.
- The rendered worker still uses `--backlog-batch 4`.
- Fake launchctl observed bootstrap and consecutive running-state checks for `com.brainlayer.hotlane-brainbar`.

Add a second installer test whose fake launchctl accepts bootstrap but reports a
waiting job with `last exit code = 2`. Assert setup exits nonzero and reports
that the hotlane failed its runtime verification. This is the regression for
“installed” being reported as “working.”

**Step 4: Run the tests and verify RED**

Run:

```bash
uv run pytest \
  tests/test_engine_package_boundary.py::test_pyproject_declares_pure_engine_package_boundary \
  tests/test_installable_build.py::test_packaged_launchd_installer_installs_hotlane_daemon \
  tests/test_installable_build.py::test_wheel_contains_cli_and_launchd_templates -q
```

Expected: failures because the daemon is not declared in package metadata, is not copied by the installer, and is absent from the wheel.

### Task 2: Package and install the stable hotlane daemon

**Files:**
- Modify: `pyproject.toml`
- Modify: `scripts/launchd/install.sh`
- Modify: `scripts/launchd/com.brainlayer.hotlane-brainbar.plist`

**Step 1: Ship the daemon**

Add:

```toml
"scripts/hotlane_brainbar_daemon.py" = "brainlayer/launchd/hotlane_brainbar_daemon.py"
```

to wheel force-includes and add `scripts/hotlane_brainbar_daemon.py` to sdist `only-include`.

**Step 2: Add a stable install destination**

Define:

```bash
HOTLANE_BRAINBAR_DST="$BRAINLAYER_LIB_DIR/hotlane_brainbar_daemon.py"
```

Add `install_hotlane_brainbar_daemon()` that resolves the daemon beside the packaged installer, falls back one directory up for a source checkout, fails clearly if absent, and installs it mode `0755` at the stable destination.

**Step 3: Make every hotlane installation copy before loading**

Inside `install_plist`, when `name=hotlane-brainbar`, call `install_hotlane_brainbar_daemon` before rendering or bootstrapping.

Extend the template render with:

```bash
-e "s|__HOTLANE_BRAINBAR_DAEMON__|$HOTLANE_BRAINBAR_DST|g"
```

**Step 4: Make reload safe under concurrent supervision**

After `bootout`, poll `launchctl print` with bounded attempts until the old
label is confirmed absent. If bootstrap then exits nonzero, tolerate it only
for the hotlane and only when a fresh `launchctl print` proves another
supervisor loaded the label after that confirmed-unloaded state. Preserve
bootstrap failure for every other case.

Add a regression test that simulates the fleet watchdog winning that race and
requires the installer to continue into runtime verification.

**Step 5: Verify the launchd worker actually stays running**

After bootstrapping the hotlane, poll `launchctl print` and require two
consecutive `state = running` samples for the same numeric PID. Use bounded
attempts and a configurable test-only interval. If the job is waiting,
crash-looping, missing a PID, or exits during the samples, return nonzero with a
clear runtime-verification error.

**Step 6: Point launchd at the stable file**

Replace:

```xml
<string>__BRAINLAYER_DIR__/scripts/hotlane_brainbar_daemon.py</string>
```

with:

```xml
<string>__HOTLANE_BRAINBAR_DAEMON__</string>
```

**Step 7: Run the focused tests and verify GREEN**

Run the Task 1 command. Expected: all selected tests pass.

### Task 3: Prepare BrainLayer v1.5.2

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/brainlayer/__init__.py`
- Modify: `server.json`
- Modify: `brain-bar/bundle/Info.plist`

**Step 1: Bump all release metadata**

Change `1.5.1` to `1.5.2` in the project version, package `__version__`, MCP
server version, MCP package version, and both checked-in BrainBar bundle version
keys.

**Step 2: Verify release consistency**

Run:

```bash
uv run pytest tests/test_release_version_sync.py tests/test_version_consistency.py -q
```

Expected: all tests pass with all release metadata synchronized.

### Task 4: Verify source, wheel, installer, and live branch build

**Files:**
- Verify only

**Step 1: Run focused packaging and launchd tests**

```bash
uv run pytest \
  tests/test_installable_build.py \
  tests/test_engine_package_boundary.py \
  tests/test_launchd_hygiene.py \
  tests/test_hotlane_brainbar_daemon.py \
  tests/test_stability_health_check.py -q
```

**Step 2: Build and inspect the wheel**

```bash
uv build --wheel --out-dir <temporary-directory>
python -m zipfile -l <wheel>
```

Confirm both the plist and `brainlayer/launchd/hotlane_brainbar_daemon.py` exist.

**Step 3: Test a throwaway installed wheel**

Install the wheel into a temporary virtual environment, run `brainlayer --help`, run the packaged hotlane daemon with `--help`, and execute `brainlayer setup --launchd --target hotlane` against fake HOME/fake launchctl. Confirm the rendered daemon target exists.

**Step 4: Run the full suite**

```bash
ulimit -n 4096
uv run pytest
```

Record exact pass/skip/xfail/failure counts. Do not claim green if failures remain.

**Step 5: Complete the non-VoiceLayer daemon gate**

From a fresh real client session, verify BrainLayer MCP tools are available and return real results against the branch-installed worker before merge.

### Task 5: Complete the PR and review loop

**Files:**
- Commit all files above

**Step 1: Run bounded local CodeRabbit review**

```bash
coderabbit review --agent
```

Address critical findings before commit.

**Step 2: Commit and push**

```bash
git add \
  docs/plans/2026-07-24-package-hotlane-worker.md \
  pyproject.toml \
  scripts/launchd/install.sh \
  scripts/launchd/com.brainlayer.hotlane-brainbar.plist \
  brain-bar/bundle/Info.plist \
  src/brainlayer/__init__.py \
  server.json \
  tests/test_installable_build.py \
  tests/test_engine_package_boundary.py
git commit -m "fix: package the persistent hotlane worker"
git push -u origin fix/package-hotlane-worker
```

**Step 3: Open a ready PR for issue 626**

Include the wheel reproduction, RED/GREEN evidence, built-wheel install proof, and live daemon/MCP receipt.

**Step 4: Invoke all available reviewers**

Request `@codex review`, `@cursor @bugbot review`, and CodeRabbit. Complete at least two review rounds, explicitly replying to every CRITICAL/HIGH/MAJOR finding.

**Step 5: Merge only when CI, review, and daemon gates are clean**

Use a merge commit. Verify the remote merge commit contains the latest pushed tree.

### Task 6: Publish and verify the supported distribution

**Files:**
- Modify in `homebrew-LAYERS`: `Formula/brainlayer.rb`
- Modify in `homebrew-LAYERS` if the tag creates a new BrainBar asset: `Casks/brainbar.rb`

**Step 1: Tag the verified merge**

Create and push `v1.5.2`. This triggers PyPI trusted publishing and the signed/notarized BrainBar release workflow.

**Step 2: Verify release artifacts**

Confirm the GitHub workflows are green, PyPI serves `brainlayer==1.5.2`, the wheel contains the daemon, and the GitHub release contains the expected BrainBar asset.

**Step 3: Update the Homebrew tap**

Update version URLs and SHA-256 values from the published PyPI and GitHub artifacts. Run the tap’s formula/cask tests, commit, push, review, and merge through its PR workflow.

**Step 4: Perform a clean supported install**

In an isolated Homebrew test prefix or clean fixture:

- Install/upgrade `brainlayer` from the tap.
- Run the supported hotlane setup command.
- Confirm launchd’s rendered daemon target exists inside the stable per-user library.
- Confirm the worker remains running and produces an embedding for a newly stored chunk.
- Confirm Etan’s real BrainLayer MCP client reconnects.

**Step 5: Upgrade the M1 off the workaround**

Upgrade the M1 to v1.5.2, rerun supported setup, and verify the installed stable daemon hash comes from the released wheel, not the manual v1.5.1 recovery copy. Re-run coverage, flow, launchd, error-log, WAL, and MCP checks before declaring the release complete.
