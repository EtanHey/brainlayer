# Ingest coverage verification (TLA+)

Status: **the property does NOT hold for today's code in 5 failure shapes:** lag, coverage, DB write
failure, a missing or moved root, and a rewrite the #732 fingerprint cannot see. It holds for a dead
or stuck watcher, a blind monitor (stale or unreadable health file), a rewrite the fingerprint does
see (every rewritten record is re-ingested, checked record by record), and sleep. A follow-up fix
for the first four is modelled here (`Rule = "proposed"`) and passes. It is not in the code yet. The
lossy rewrite needs a different fix (below).

Etan's condition for retiring the chart icon (13 Nov): "any type of agent that is running locally
that can use brain layer ... would automatically for sure make an error if brain bar is connected
but not indexing their Jason L's or their memories".

**The property, `IngestFailureLeavesGreen`:** if a wired agent writes and ingest makes no progress
for more than N health ticks, or the monitor cannot see a new watcher heartbeat for more than N
ticks, then the `ingest` check is amber or red. It stays non-green until progress resumes. Also:
`SleepIsNotFailure` (closing the lid never leaves green by itself) and
`StallAcrossWakeStillCounts` (a stall that was running when the lid closed still counts after wake).

## Run

From the repository root, with Python 3.11+ and Java 17:

```sh
formal/ingest-coverage/check.sh                  # every case and the trace freshness, about 3 minutes
formal/ingest-coverage/check.sh --case finding   # cases whose name starts with this
formal/ingest-coverage/check.sh --list           # the case table
formal/ingest-coverage/check.sh --require-sync   # also fail when a covered source changed (CI runs this)
python3 formal/ingest-coverage/check.py --write-configs   # after editing CASES
python3 formal/ingest-coverage/replay.py --generate       # after changing the spec or SCENARIOS
pytest tests/test_ingest_coverage_replay.py               # the model against the real code
```

TLC is `TLC_JAR=<jar>` or `~/.local/share/tlaplus/tla2tools-1.7.4.jar`. Exit **0**: every case matched.
**1**: a case did not, a config or trace is stale, or `--require-sync` saw a changed source. **2**:
incomplete (no TLC, a crash). Output: `.build.noindex/ingest-coverage/<UTC>-<pid>/` (per-case `.log`,
`.trace.txt` with changed variables only, `results.json`). CI: job `TLA+ ingest coverage`; the replay
test runs in the `test` job.

## What the model is

`IngestCoverage.tla`, one step per health tick (300 s) or per tick asleep. It follows the real code:

- `watcher.py` `poll_once` writes `watcher-health.json` in its `finally` block on every completed
  poll, whatever the poll ingested. `updated_at` and `poll_count` are in the same payload, so they
  freeze together.
- `health_facts.watcher_fact` reads only `updated_at`. That is the ingest check's only input.
- `health_verdict.evaluate` turns that one clock into the state. Amber after 10 min, red after
  30 min. Unknown turns amber on the 3rd tick in a row. Red holds through amber, and recovery
  needs 2 observed good ticks. On a confirmed clock, time asleep is credited. Otherwise there is
  one wake-grace window per clock value.
- With a verdict, the dot is the verdict (`badge_state.py:127`). An `unknown` check counts as
  green in the dot.
- The watcher's coverage watchdog, offset lag, DB probe and file failures go into
  `watcher-health.json`, and `watcher_stalled` goes into ingest *context*. **No verdict input reads
  them.** The `watcher_zero_writes_while_active` alarm goes to stderr and Axiom only.

State: the fault this tick (none, or one of the 8 shapes, with shape 8 split into a detected and a lossy rewrite), awake or asleep, whether the agent
writes, heartbeat and persisted-clock ages (awake and wall), resume and grace windows, the
verdict (state, unknown ticks, good ticks), and the stall and blind counters. `coverage.json`
maps each model action to a source line and pins each source by sha256.

The wired agents are `default_watch_roots` (`watcher.py:357`): claude, codex, cursor,
cursor-agent-transcripts and gemini. The verdict does not depend on which agent writes. Agent
identity matters only in the replay, where each agent writes its own format.

## Results

| Case | Kind | Expect | Got | Meaning |
|---|---|---|---|---|
| `real-heartbeat-shapes` | guarantee | pass | pass | Today: dead, stuck and blind (stale or unreadable) leave green within N = 2 ticks (10 min) and stay non-green until progress. A rewrite the fingerprint sees is real progress (every record re-ingested). Lid closes never leave green. `StallClockIsTruthful` holds. |
| `real-unconfirmed-clock` | guarantee | pass | pass | Without a boot-session clock, wake grace stretches the bound to N = 7 model ticks. A stall still never stays green. |
| `real-bound-tight` | bound | violated | violated | N = 1 fails, so 2 is the real bound. |
| `finding-real-lag` | **finding** | violated | violated | Offset lag grows, the heartbeat stays fresh, and the check stays green forever. |
| `finding-real-coverage` | **finding** | violated | violated | Records stop being indexed (under 25 % coverage, or format drift). Green forever. |
| `finding-real-dbfail` | **finding** | violated | violated | Every flush fails and is retained. Green forever. |
| `finding-real-rootgone` | **finding** | violated | violated | The root is renamed and the agent writes elsewhere. Green forever. |
| `finding-real-rewrite-lossy` | **finding** | violated | violated | A same-length edit past the 4 KiB head digest and before the last 256 bytes is invisible to the #732 fingerprint. The watcher resumes at its offset, the edited records are never ingested, and the check stays green. |
| `finding-proposed-rewrite-lossy` | **finding** | violated | violated | The proposed alert rule does not fix that: the watcher never knows anything was lost. |
| `finding-unconfirmed-short-sleep` | **finding** (minor) | violated | violated | Unconfirmed clock, a 10–15 min lid close, monitor reads before the first poll: amber. |
| `proposed-all-shapes` | guarantee | pass | pass | With the follow-up fix (below), every shape except the lossy rewrite holds. |
| `mut-stale-green` | mutation | violated | violated | Treating a readable but stale health file as green breaks it. |
| `mut-trust-alive` | mutation | violated | violated | Ignoring a frozen poll_count (trusting a live process) breaks it. |
| `mut-no-coverage` | mutation | violated | violated | Dropping the coverage check from the fixed rule breaks it. |
| `mut-sleep-resets` | mutation | violated | violated | Letting a wake forgive the heartbeat clock breaks `StallAcrossWakeStillCounts`. |
| `mut-rewrite-progress` | mutation | violated | violated | Counting every rewrite as progress, lossy or not (round 1's model), breaks `StallClockIsTruthful`: records dropped with the stall clock at 0. |

The exhaustive cases explore every interleaving of faults, writes, sleeps and wake order within
the bounds. That is up to about 40,000 distinct states each.

## Finding: five shapes stay green today

The counterexample is three states long (`finding-real-coverage.trace.txt`):

```
State 1: fault="none"  v="green"  stall=0
State 2: fault="coverage"  backlog=TRUE  stall=1
State 3: stall=2
State 4: stall=3        <- v is still "green"; the heartbeat is fresh every tick
```

The real code reproduces this. The replay (`coverage`, `lag`, `dbfail`, `rootgone`, all five agents)
writes a record every tick for 9 ticks (45 min). None reaches the scratch DB, and the real
`evaluate()` reports ingest green on every tick. `test_the_findings_are_real_today` pins it.

**The lossy rewrite.** The running tailer's `_verify_anchor` and the resume fingerprint compare only
the 256 bytes before the offset and a digest of the first 4 KiB. In `rewrite_lossy`, the agent edits
one record in place each tick, past 4 KiB, keeping the same length, and appends a new turn. For all
five agents the appended turn is ingested, the edited record never is, and ingest stays green on a
fresh heartbeat. `test_a_lossy_rewrite_lands_the_newest_record_and_loses_the_edited_one` pins the
split. A newest-record-only check, which is what round 1's replay used, would have reported no
backlog. Fixing this needs a fingerprint that covers every byte before the offset (a running digest
of consumed bytes), not an alert, because the watcher cannot report a change it never sees.
Agents append in practice, and Claude Code's rewinds shrink the file, which `check_rewind` handles.
So this matters only for an agent that edits transcript records in place.

**Proposed follow-up (separate PR, not here):** let the ingest check read what the watcher already
publishes. Amber on the first tick, red after 2, while `watcher-health.json` shows
`alert_reasons` ∋ `coverage_drop` / `offset_lag` / `file_ingestion_failure`, or `db_probe_failed`
with pending input. Add a `roots_missing` field for a wired root that existed and is gone. The
alert has to persist in the file a dead watcher leaves behind (the model needed that; see
`fileAlert`). `proposed-all-shapes` is that rule. `mut-no-coverage` shows the coverage part is
load-bearing.

**Minor finding (unconfirmed clock).** With no `kern.bootsessionuuid` pair (first tick after a reboot,
or wall lag ≥ 600 s), a lid close of 10–15 min is not a gap (it must be > `sleep_gap_s` = 900 s), so
there is no grace and no credit. If the health tick runs before the watcher's first poll after
wake, the heartbeat is 900 s old, which is over 600 s, so amber. Recovery then takes 2 good ticks.
The replay reproduces this on the real code (`unconfirmed-short-sleep`). With a confirmed clock,
the normal macOS path, it cannot happen (`real-heartbeat-shapes`).

## Replay against the real code

`replay.py --generate` runs 15 scripted scenarios through TLC at production thresholds (300 s
ticks) and commits the model's per-tick behaviour as `traces/*.json`. The scenarios are each of
the 10 fault kinds, sleep, sleep with monitor-first wake, a stall across a wake, and unconfirmed
short and long sleeps. `tests/test_ingest_coverage_replay.py` drives the real code with each trace
for each of the 5 agents (75 runs):

- a real `JSONLWatcher` over the real `default_watch_roots` of a scratch home;
- the real `create_flush_callback` into a scratch DB;
- the real `watcher_fact` and `evaluate`;
- JSONL in each agent's own format.

Every tick must match on three things: the ingest state, what the monitor saw
(fresh / old / unreadable), and whether **every** record the agent wrote has reached the DB. A
rewrite's records are checked one by one, not just the newest. Records a fault really drops
(format drift, an in-place edit the fingerprint cannot see) count as backlog only while that
fault lasts. They stay out of the DB afterwards, which is the finding.

Faults are real where the OS allows it:
- a write-only transcript (lag);
- a renamed root;
- a read-only health directory (stale);
- a file rewritten in place, longer, under the stored offset (detected), and a same-length edit
  past the first 4 KiB (not detected);
- records in a shape no parser takes (coverage).

Three are stand-ins at a boundary:
- a dead or stuck watcher is "no completed poll";
- a DB failure is raised from the flush callback;
- an unreadable health file is one failed read.

The replay can fail, and each of these is a test:
- making the real `_age_state` always green gives 8–9 mismatching ticks;
- flipping one model tick is caught;
- disabling the real `_verify_anchor` makes the detected-rewrite scenario mismatch.

## What this cannot cover

- **Agents BrainLayer is not wired to**, including future labs' agents. Sources are wired on
  purpose ("nothing is auto-detected"), so the guarantee is per wired root. The known-agent census
  (separate stacked PR) catches running agents from a fixed list whose transcripts land outside
  the watched roots. It cannot catch an agent that is not on the list. Today, for example, Gemini
  CLI writes `~/.gemini/tmp/<hash>/chats/session-*.json`, which is not
  `~/.gemini/sessions/**/*.jsonl`.
- **Wrong parsing or content**: a record ingested as the wrong text. (The lossy in-place rewrite
  used to be listed here; it is now a modelled, replayed finding.)
- **The Swift icon drawing.** That is BrainBar's app tests, not this.
- **Maintenance pause** (red softened to amber, still non-green) and the `t3_ingest_unhealthy` path are
  not modelled.

## Keeping it in step

CI runs `check.py --require-sync`. A change to any pinned source fails this job until someone does
three things: re-reads the mapped lines in `coverage.json`, updates the model or adds a switch, and
reruns `check.py`, `replay.py --generate` and the replay test. Only then is the pin moved.
