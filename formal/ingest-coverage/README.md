# Ingest coverage verification (TLA+)

Status: **the property does NOT hold for today's code in 4 of the 8 failure shapes.** It holds for
the other 4 (dead, stuck, blind, rewritten file) and for sleep. A follow-up fix closes the 4 gaps;
it is modelled here (`Rule = "proposed"`) and passes. It is not in the code yet.

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

State: the fault this tick (none, or one of the 8 shapes), awake or asleep, whether the agent
writes, heartbeat and persisted-clock ages (awake and wall), resume and grace windows, the
verdict (state, unknown ticks, good ticks), and the stall and blind counters. `coverage.json`
maps each model action to a source line and pins each source by sha256.

The wired agents are `default_watch_roots` (`watcher.py:357`): claude, codex, cursor,
cursor-agent-transcripts and gemini. The verdict does not depend on which agent writes. Agent
identity matters only in the replay, where each agent writes its own format.

## Results

| Case | Kind | Expect | Got | Meaning |
|---|---|---|---|---|
| `real-heartbeat-shapes` | guarantee | pass | pass | Today: dead, stuck, blind (stale or unreadable) and a rewritten file leave green within N = 2 ticks (10 min) and stay non-green until progress. Lid closes never do. |
| `real-unconfirmed-clock` | guarantee | pass | pass | Without a boot-session clock, wake grace stretches the bound to N = 7 model ticks. A stall still never stays green. |
| `real-bound-tight` | bound | violated | violated | N = 1 fails, so 2 is the real bound. |
| `finding-real-lag` | **finding** | violated | violated | Offset lag grows, the heartbeat stays fresh, and the check stays green forever. |
| `finding-real-coverage` | **finding** | violated | violated | Records stop being indexed (under 25 % coverage, or format drift). Green forever. |
| `finding-real-dbfail` | **finding** | violated | violated | Every flush fails and is retained. Green forever. |
| `finding-real-rootgone` | **finding** | violated | violated | The root is renamed and the agent writes elsewhere. Green forever. |
| `finding-unconfirmed-short-sleep` | **finding** (minor) | violated | violated | Unconfirmed clock, a 10–15 min lid close, monitor reads before the first poll: amber. |
| `proposed-all-shapes` | guarantee | pass | pass | With the follow-up fix (below), all 8 shapes hold. |
| `mut-stale-green` | mutation | violated | violated | Treating a readable but stale health file as green breaks it. |
| `mut-trust-alive` | mutation | violated | violated | Ignoring a frozen poll_count (trusting a live process) breaks it. |
| `mut-no-coverage` | mutation | violated | violated | Dropping the coverage check from the fixed rule breaks it. |
| `mut-sleep-resets` | mutation | violated | violated | Letting a wake forgive the heartbeat clock breaks `StallAcrossWakeStillCounts`. |

The exhaustive cases explore every interleaving of faults, writes, sleeps and wake order within
the bounds. That is up to about 40,000 distinct states each.

## Finding: four shapes stay green today

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

`replay.py --generate` runs 14 scripted scenarios through TLC at production thresholds (300 s
ticks) and commits the model's per-tick behaviour as `traces/*.json`. The scenarios are each of
the 9 fault kinds, sleep, sleep with monitor-first wake, a stall across a wake, and unconfirmed
short and long sleeps. `tests/test_ingest_coverage_replay.py` drives the real code with each trace
for each of the 5 agents (70 runs):

- a real `JSONLWatcher` over the real `default_watch_roots` of a scratch home;
- the real `create_flush_callback` into a scratch DB;
- the real `watcher_fact` and `evaluate`;
- JSONL in each agent's own format.

Every tick must match on three things: the ingest state, what the monitor saw
(fresh / old / unreadable), and whether the newest record reached the DB.

Faults are real where the OS allows it:
- a write-only transcript (lag);
- a renamed root;
- a read-only health directory (stale);
- a file rewritten in place, longer, under the stored offset;
- records in a shape no parser takes (coverage).

Three are stand-ins at a boundary:
- a dead or stuck watcher is "no completed poll";
- a DB failure is raised from the flush callback;
- an unreadable health file is one failed read.

The replay can fail. Making the real `_age_state` always green gives 8–9 mismatching ticks, and
flipping one model tick is caught. Both are tests.

## What this cannot cover

- **Agents BrainLayer is not wired to**, including future labs' agents. Sources are wired on
  purpose ("nothing is auto-detected"), so the guarantee is per wired root. The known-agent census
  (`src/brainlayer/agent_census.py`, below) catches running agents from a fixed list whose
  transcripts land outside the watched roots. It cannot catch an agent that is not on the list. Today, for example, Gemini
  CLI writes `~/.gemini/tmp/<hash>/chats/session-*.json`, which is not
  `~/.gemini/sessions/**/*.jsonl`.
- **Wrong parsing or content**: a record ingested as the wrong text, or a rewrite whose head
  digest and tail happen to match (`#732`'s fingerprint).
- **The Swift icon drawing.** That is BrainBar's app tests, not this.
- **Maintenance pause** (red softened to amber, still non-green) and the `t3_ingest_unhealthy` path are
  not modelled.

## Known-agent census (the add-on for unwired agents)

`agent_census.KNOWN_AGENTS` lists claude, codex, cursor-agent, gemini and opencode, each with its
process name and transcript roots. On every health check, `health_check` reuses the `ps`
snapshot it already takes. When a listed agent is running (matched on the argv[0] or argv[1]
basename) and has written a file in the last 24 h under a root that no `default_watch_roots`
entry covers, three things happen:

- an `unindexed_agent_running` warning issue;
- the `ingest` check is floored at **amber** with the note "an agent BrainLayer does not index is
  running: <agent>: <root>";
- a typed `unindexed_agents` list appears on the check, with the code in `context_codes` (seam F21).

The root probe is bounded at depth 4 and 2,000 entries. An empty census changes nothing. A census
that was not taken (the `ps` read failed) is `UNOBSERVED` and changes nothing; the monitor
check already reports `process_snapshot_failed`. The model does not include the census, because
it can only add non-green ticks.

## Keeping it in step

CI runs `check.py --require-sync`. A change to any pinned source fails this job until someone does
three things: re-reads the mapped lines in `coverage.json`, updates the model or adds a switch, and
reruns `check.py`, `replay.py --generate` and the replay test. Only then is the pin moved.
