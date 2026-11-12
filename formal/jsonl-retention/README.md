# Local JSONL retention safety

Run `sh formal/jsonl-retention/check.sh` with Java and `TLC_JAR` (defaults to
`~/.local/share/tlaplus/tla2tools-1.7.4.jar`). CI uses the existing pinned jar
and uploads the TLC logs. Run `PYTHONPATH=src python3
formal/jsonl-retention/check_mutations.py` for production-function mutation
checks; pytest runs those too. Hypothesis is absent from dev dependencies.

Two archives arrive successively; newest changes. Upload can produce different
bytes under the same name, Verify creates durable receipts, and live Drive
objects may disappear or change while local bytes survive. Revoke, Prune,
Restore and Crash model the durable pending-revocation window. Tick and
ClockJump advance the observed wall clock through boundary ages 0/14/28/31.
Both mtime and verified-at ages must pass. Re-verification resets receipt age.

Invariants: NeverDeleteWithoutVerifiedCopy (live matching bytes at deletion),
NeverDeleteYoungerThan14Days (both observed ages), NeverDeleteNewest,
RestoreOnlySurvivors and AlwaysAtLeastOneCopy (each archive ever created,
until its measured age reaches the 30-day abstract Drive-policy horizon).
Production Drive retention is count-based (30), not a 30-day guarantee: the
model's horizon abstracts a daily-backup policy and is not a retention SLA.

The current case assumes external actors do not remove or corrupt the LAST
copy before that horizon. RemoteDelete/RemoteReplace still invalidate receipts
while local files exist, exercising the live-check veto. The separate external
case REMOVES that assumption and fails AlwaysAtLeastOneCopy: after authorized
local deletion, external cleanup removes the final Drive copy. This is an
unavoidable environmental counterexample, not proof that arbitrary external
cleanup is safe. A fresh listing also cannot prevent a remote delete racing
between the listing and unlink; no transaction spans Drive and local storage.

A forward clock jump is modeled as observed age advancing. The two wall-clock
ages cannot prove 14 days of physical elapsed time across arbitrary jumps.
They prevent a backdated file from bypassing a recent receipt, not clock fraud.
A trusted elapsed clock/remote hold would be a different requirement. These
limits are explicitly reported for the round-2 reviewer.

Five rule mutations must fail TLC: drop live check, drop content MD5 check,
drop newest protection, drop age guards, restore revoked receipts blindly.
The first four plus dropped verified are also rejected by in-memory mutations
of the REAL retention function with the seeded randomized scratch oracle.
Real run_backup tests exercise revoke/restore and the interrupted-prune note.

| Model action | Production code (src/brainlayer/jsonl_backup.py) |
|---|---|
| New | run_backup bundle creation (1784); _local_archive_paths (428) discovers new newest |
| Upload | upload_file_to_drive_raw call (1908) |
| Verify | _drive_archive_receipt (415), then durable _atomic_write_json (1940) |
| Delete | prune_local_archives (445), gates (463-483), unlink (498) |
| RemoteDelete / RemoteReplace | external actors; _list_surviving_archives (352) observes their effect |
| Revoke / Prune / Restore | retained_state pending marker, revoke, prune and survivor restore (1959-1980) |
| Crash | atomic receipt persistence (210), pending marker recovery (1748) |
| Tick / ClockJump | run_backup now (1683), mtime and verified_at age checks (461-470) |

The source static guard inspects rejection gates before unlink; randomized
checks and real run tests are executable conformance checks. This model is an
abstraction, not a proof of every Python statement or OS/Drive implementation.
Legacy receipts do not authorize deletion. No real archives were modified.
