## Change and evidence

Describe the change and the exact-head checks run.

## Crash sweep / feature batch

- Add the `crash-sweep` label to force all 128 crash shards on this PR (including a single-PR batch).
- Migration, revert, launchd, maintenance, supervisor, watcher, vector-store, pipeline and daily-backup paths require sweeps automatically; CI and sweep harness changes do too.
- Before a batch release tag, install or done claim, require a successful **CI** run on the exact batch commit SHA, from a main push or **Run workflow** on the batch ref. PR merge-ref runs are insufficient.
- `python3 scripts/ci/crash_sweep_gate.py verify-release EtanHey/brainlayer <full-batch-SHA>` prints the qualifying job URL or fails closed. The tag pre-push hook and both publishing workflows enforce it. After any new commit, run the gate again for that SHA.
- A single-PR batch runs this gate immediately. All shards, artifacts and deadlines remain unchanged. A PR skip is not batch acceptance.

Loop-record: implementer=<vendor/model/effort>; reviewer=none; rounds=0; defects=0
