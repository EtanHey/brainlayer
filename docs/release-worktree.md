# Release worktrees

Use the tracked runner instead of a fresh repository clone for a local release step:

```bash
python3 "$B/scripts/release_worktree.py" "$M" -- bash "$STEP" "$V" "$M" '{worktree}'
```

`B` is the canonical checkout, `M` is an already-fetched release tag or SHA,
and `STEP` is the operator's foreground release script. Its arguments are the
version, release reference, and detached worktree path. The command runs inside
that worktree. It must wait for its own children before returning; do not
background the tag push and let cleanup race it. Run verification and release
gates as usual before this command, and run the release tail only after it succeeds.
Use the normal heavy-slot admission and evening cutoff policy for the foreground step.

The runner resolves the reference to an exact commit and creates a uniquely named
detached checkout under the canonical `.worktrees/`, even when invoked from another
worktree. Git objects, refs and configuration are shared. The runner never sets
`core.hooksPath`: the existing boundary wrapper remains in force and can delegate
to the detached checkout's `.githooks/pre-push`, including its tag-range/crash-sweep
gates. Do not copy the old clone's `git config core.hooksPath .githooks` line.

The foreground script owns its build or venv setup. It can also read
`BRAINLAYER_RELEASE_WORKTREE` and the resolved `BRAINLAYER_RELEASE_SHA` from its
environment. The runner performs no fetch, tag creation, push or installation itself.

After the command finishes, plain `git worktree remove` removes a clean checkout
and its ignored build/venv output. Dirty tracked files or untracked artifacts
cause cleanup to refuse; the runner prints the retained path and preserves it
without force. The original nonzero command status is retained; a successful
command whose cleanup fails returns nonzero. Existing checkouts are never reused
or deleted. A hard kill can leave a worktree for operator inspection.

Scratch-Git tests exercise an actual local push and inherited hook invocation,
shared common directory, detached HEAD, unchanged source branch/config,
successful/failed cleanup, dirty preservation and invocation from another worktree.
They perform no network push, fleet release, build or installation.
