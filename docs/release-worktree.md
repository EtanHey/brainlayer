# Release worktrees

Use the tracked runner instead of a fresh repository clone for a local release step:

```bash
python3 "$B/scripts/release_worktree.py" "$M" -- bash "$STEP" "$V" "$M" '{worktree}'
```

`B` is the canonical checkout, `M` is the already-fetched full 40-character
release commit SHA, and `STEP` is the operator's foreground release script.
Copy and read that script into an operator-owned path such as
`$B/.build.noindex/release-tag-step.sh`; do not depend on a worker's temporary
worktree. Its arguments are the version, full release SHA, and detached worktree path. The command runs inside
that worktree. It must wait for its own children before returning; do not
background the tag push and let cleanup race it. Run verification and release
gates as usual before this command, and run the release tail only after it succeeds.
Use the normal heavy-slot admission and evening cutoff policy for the foreground step.
Fetch release tags, including the predecessor, in the outer chain before invoking
the runner: shared refs must contain the tags the pre-push range needs. The runner
itself also accepts an existing tag as its reference when the command supports it;
the example release step requires the full SHA and rejects invalid values loudly.

The runner resolves the reference to an exact commit and creates a uniquely named
detached checkout under the canonical `.worktrees/`, even when invoked from another
worktree. Git objects, refs and configuration are shared. The runner never sets
`core.hooksPath`: the existing boundary wrapper remains in force and can delegate
to the detached checkout's `.githooks/pre-push`, including its tag-range/crash-sweep
gates. Do not copy the old clone's `git config core.hooksPath .githooks` line.

The foreground script owns its build or venv setup. It can also read
`BRAINLAYER_RELEASE_WORKTREE` and the resolved `BRAINLAYER_RELEASE_SHA` from its
environment. The runner performs no fetch, tag creation, push or installation itself.

The command gets its own process group. SIGINT or SIGTERM sent to the runner
alone is forwarded to that whole group. The runner waits up to five seconds,
then sends SIGKILL if needed and waits up to two more seconds. It reaps the direct
child and checks for live descendants before cleanup; zombies cannot execute and
do not hold cleanup open. If the group is still live or cannot be measured, the
worktree is preserved and the runner returns nonzero. Foreground commands must
not daemonize or move descendants to another process group.

After the command finishes and its group has stopped, plain `git worktree remove` removes a clean checkout
and its ignored build/venv output. Dirty tracked files or untracked artifacts
cause cleanup to refuse; the runner prints the retained path and preserves it
without force. The original nonzero command status is retained; an interrupted
command returns 130 for SIGINT or 143 for SIGTERM. A successful command whose
cleanup is refused returns **3**, meaning the release step succeeded but its
worktree was retained: inspect it before rerunning a tag push. Setup errors return
2 and failure to start the command returns 127. A child may itself return any
nonzero status, so also read the printed outcome. Existing checkouts are never
reused or deleted. A hard kill of the runner can leave a worktree for inspection.

Scratch-Git tests exercise an actual local push and inherited hook invocation,
shared common directory, detached HEAD, unchanged source branch/config,
successful/failed cleanup, dirty preservation, refusing pre-push hooks,
runner-only signal forwarding/waiting and invocation from another worktree.
They perform no network push, fleet release, build or installation.
