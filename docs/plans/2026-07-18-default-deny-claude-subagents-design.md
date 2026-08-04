# Default Claude Subagent Denylist Design

## Goal

Make Claude subagent transcripts opt-out of ingestion by default by adding
`~/.claude/projects/**/subagents/**` to `DEFAULT_INGEST_DENYLIST`.

## Design

Keep the existing denylist matcher and environment-override contract unchanged.
The default tuple gains the subagent glob alongside the workflow glob, so direct
Claude sessions and other providers remain allowed. An explicitly configured
`BRAINLAYER_INGEST_DENYLIST`, including an empty value, continues to replace the
defaults. Existing attribution code remains in place for compatibility; this
patch does not refactor adjacent policy machinery.

## Verification

Update the focused policy test first so it expects ordinary attributed Claude
subagents to be denied. Confirm that test fails against the old default, make
the one-line tuple change, then run the complete denylist tests and the full
project gate before publishing the PR.
