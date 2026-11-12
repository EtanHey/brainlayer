# Local JSONL retention safety

Run `sh formal/jsonl-retention/check.sh` with Java and `TLC_JAR` (defaults to
`~/.local/share/tlaplus/tla2tools-1.7.4.jar`), like the adjacent policy-reconcile
cases. Run `PYTHONPATH=src python3 formal/jsonl-retention/check_mutations.py`
for the production-function mutation checks, plus
`pytest -q tests/test_jsonl_local_retention.py` for the seeded randomized oracle.
Hypothesis is not in the dev dependencies.

TLC exhaustively explores two archives, ages 0 through 14, Upload, Verify,
Delete, Crash and Tick. Newest is fixed: creating another archive can only
release an older archive from that protection. Verify represents durable
exact-name/size/hash receipt persistence; Crash stutters because receipts
survive and unlink is atomic. No action removes Drive copies: independent
remote deletion is outside this local-deletion proof. Production invalidates
local-deletion receipts before its optional Drive pruning, restoring only
survivors after successful pruning. A crash there conservatively retains locals.

Invariants: NeverDeleteWithoutVerifiedCopy, NeverDeleteYoungerThan14Days,
NeverDeleteNewest, AlwaysAtLeastOneCopy. The last applies per archive at local
deletion, not forever after future Drive retention. Changing the configured
age above 14 only restricts Delete further.

Both dropped-verified and dropped-age model mutations produce the named
invariant violations. Both corresponding in-memory mutations of the REAL
retention function fail the randomized scratch oracle (120 sets, mismatched
sizes, unverified receipts, boundary ages, crashes before/after unlink, retry).
Legacy receipts without archive-level identity fail closed; no home backup
archives were read or modified.
