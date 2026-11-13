# Ingest coverage verification (TLA+)

Status: **with the stall-code fix (INGEST-FIX, stacked on #1780) the property holds for each of the
8 failure shapes on its own**: a dead or stuck watcher, lag (an unreadable transcript), coverage (writes
that never reach the DB the watcher probes), DB write failure, a blind monitor (stale or unreadable
health file, or a payload whose stall fields are absent or malformed), a missing or moved root, and a
rewrite the #732 fingerprint sees. Sleep never leaves
green by itself. **Three open findings still fail it:** records in a shape no parser takes (`drift`),
a same-length in-place edit the fingerprint cannot see (`rewrite_lossy`, #1780's 5th finding; it
needs a full-content checksum, not an alert), and a heartbeat stall that hands straight over to a
debounced shape (`finding-real-flapping`). `before-fix-*` keeps #1780's finding: the heartbeat-only
verdict (`Rule = "heartbeat"`) leaves lag, coverage, DB failure and root-gone green forever.

Etan's condition for retiring the chart icon (13 Nov): "any type of agent that is running locally
that can use brain layer ... would automatically for sure make an error if brain bar is connected
but not indexing their Jason L's or their memories".

**The property, `IngestFailureLeavesGreen`:** if a wired agent writes and ingest makes no progress
for more than N health ticks, or the monitor cannot see a new watcher heartbeat for more than N
ticks, or it reads a fresh heartbeat whose stall fields are unreadable for more than N ticks, then
the `ingest` check is amber or red. It stays non-green until progress resumes. Also:
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
- `health_facts.watcher_health_fact` reads `updated_at` and, from the same read, the watcher's
  stall codes (`watcher_stall_codes`): `alert_reasons` ∋ `coverage_drop` / `offset_lag` /
  `file_ingestion_failure`, `db_probe_failed` once `db_probe_failed_for_s` ≥ 300 s (one failed probe
  is a momentary lock), and `root_missing:<provider>` from `roots_missing`. If any of those fields
  is absent, malformed or inconsistent (a negative probe duration, a duration while the probe is
  fine, a non-int `poll_count`), the codes are UNOBSERVED (fault `partial`), never "no stall".
- `health_verdict.evaluate` turns the heartbeat into the state. Amber after 10 min, red after
  30 min. Unknown turns amber on the 3rd tick in a row. Red holds through amber, and recovery
  needs 2 observed good ticks. On a confirmed clock, time asleep is credited. Otherwise there is
  one wake-grace window per clock value. Stall codes (`_ingest_stall`) are counted once per
  snapshot: only a new snapshot (`updated_at` or `poll_count` changed, so a backwards clock
  correction still counts) adds an alert check, so a frozen file
  re-read after sleep or from a dead watcher keeps its count instead of escalating. One or more
  checks: at least amber; the 6th (`ingest_alert_red_ticks`): red. UNOBSERVED codes are never a good
  check: the count holds, and after `unknown_amber_ticks` (3) such checks in a row the state is at
  least amber. `coverage_drop` is ignored while maintenance pauses the drain.
- With a verdict, the dot is the verdict (`badge_state.py:127`). An `unknown` check counts as
  green in the dot.
- What the watcher publishes, and when (`watcher.py`): an unreadable transcript is a
  `file_ingestion_failure` on the same poll. `coverage_drop` needs 300 s of monotonic (awake)
  time with inserts/active < 0.25; retained failed flushes count as active, and the window stays
  open while it writes nothing, so the alert outlives the fault by one poll (`covLinger`).
  `roots_missing` lists a root with a transcript write in the last 14 days (its newest offsets.json
  confirmation; prune keeps an unavailable root's offsets; being present is not activity) that has
  been gone for 300 s. So a root that vanished while the watcher was stopped is reported after a
  restart, a stale root deleted on purpose never alarms, and a recently used one stops alarming 14
  days after its last transcript write. The file a
  dead watcher leaves behind keeps its last alert (`fileAlert`).

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
| `real-<shape>` × 10 | guarantee | pass | pass | Each shape alone (dead, stuck, lag, coverage, dbfail, blind_stale, blind_unreadable, partial, rootgone, rewrite) leaves green within N = 2 ticks (10 min) and stays non-green until progress. Lid closes never leave green. `StallClockIsTruthful` holds. |
| `real-heartbeat-mix` | guarantee | pass | pass | Any mix of the heartbeat shapes holds with N = 2. |
| `real-unconfirmed-clock` | guarantee | pass | pass | Heartbeat shapes without a boot-session clock: wake grace stretches the bound to N = 7. |
| `real-unconfirmed-<alert>` × 4 | guarantee | pass | pass | lag, coverage, dbfail and rootgone hold with N = 2 without a boot-session clock (the alert path has no clock). |
| `real-bound-tight` | bound | violated | violated | N = 1 fails for the heartbeat shapes. |
| `real-bound-tight-alert` | bound | violated | violated | N = 1 fails for a debounced alert: coverage is still green on its 2nd tick. |
| `finding-real-flapping` | **finding** (open) | violated | violated | A dead watcher restarts into failing writes: one green tick at stall 3 before `coverage_drop` debounces. If it then dies again, recovery hysteresis can read green while still stalled. |
| `finding-real-drift` | **finding** (open) | violated | violated | Records no parser takes publish no watcher signal. Green forever. |
| `finding-real-rewrite-lossy` | **finding** (open) | violated | violated | A same-length edit past the 4 KiB head digest is invisible to the #732 fingerprint; the stall codes cannot see it either. Green forever. |
| `finding-unconfirmed-short-sleep` | **finding** (minor) | violated | violated | Unconfirmed clock, a 10–15 min lid close, monitor reads before the first poll: amber. |
| `before-fix-<alert>` × 4 | history | violated | violated | #1780's finding: with the heartbeat-only verdict, lag, coverage, dbfail and rootgone stay green forever. |
| `mut-stale-green` | mutation | violated | violated | Treating a readable but stale health file as green breaks it. |
| `mut-trust-alive` | mutation | violated | violated | Ignoring a frozen poll_count (trusting a live process) breaks it. |
| `mut-no-coverage` | mutation | violated | violated | Dropping `coverage_drop` from the stall codes breaks it. |
| `mut-sleep-resets` | mutation | violated | violated | Letting a wake forgive the heartbeat clock breaks `StallAcrossWakeStillCounts`. |
| `mut-partial-clear` | mutation | violated | violated | Reading a payload without its stall fields as "no stall" breaks it. |
| `mut-rewrite-progress` | mutation | violated | violated | Counting every rewrite as progress, lossy or not, breaks `StallClockIsTruthful`. |

The exhaustive cases explore every interleaving of faults, writes, sleeps and wake order within
the bounds. That is up to about 40,000 distinct states each.

## The fix, and what is still open

Before the fix, the `before-fix-coverage` counterexample is three states: the fault starts, the
heartbeat stays fresh every tick, and `v` is still green at stall 3. The fixed traces
(`traces/{lag,coverage,dbfail,rootgone}.json`, matched by the real code for all five agents) read:
lag amber on the first fault tick; coverage, dbfail and rootgone amber on the 2nd (the watcher's
300 s debounce); red on the 6th alert check; then red → recovering → green on the 2 good ticks after
the fault. `alert-across-sleep` (Codex r1 #4): an alert from before a 30 min lid close, re-read
before the first poll after wake, holds amber without escalating, then clears on the next new
snapshot. `dbfail-then-partial` (Codex r1 #2): truncated payloads after a DB failure hold amber. `tests/test_ingest_stall_alerts.py` pins the same rule through the
real `collect_facts` → `evaluate` path and the real watcher payload.

Timing caveat: the model has one watcher poll per health tick. Production polls every 30 s, so a
300 s episode that starts within ~30 s of a health read surfaces one tick later than modelled.

**Open: drift.** A record in a shape no parser takes is read and dropped. For Claude it is counted as
a seen entry and yields no chunk; for the other agents the normaliser drops it. No watcher number
moves: active entries exclude it, so `coverage_drop` cannot fire. Catching it needs a new detector
(for example a per-provider "records arrive, nothing is indexed" episode), and that risks false
alarms during long tool runs. That is a separate decision.

**Open: the lossy rewrite (#1780's 5th finding).** The running tailer's `_verify_anchor` and the
resume fingerprint compare only the 256 bytes before the offset and a digest of the first 4 KiB. A
same-length edit between them is never seen, so the watcher has nothing to report and no alert-based
rule can fix it. It needs a fingerprint over every consumed byte. Agents append in practice, and
Claude Code's rewinds shrink the file, which `check_rewind` handles.

**Open: a heartbeat stall handing over to a debounced shape.** If the watcher was dead long enough to
turn amber or red and then comes back with writes still failing, its fresh heartbeat starts recovery
before `coverage_drop` has lasted 300 s. A crash-looping watcher whose processes each live under
300 s never debounces at all, because each new process starts a new episode (not modelled: in the
model "dead" keeps the same process).

**Minor finding (unconfirmed clock).** With no `kern.bootsessionuuid` pair (first tick after a reboot,
or wall lag ≥ 600 s), a lid close of 10–15 min is not a gap (it must be > `sleep_gap_s` = 900 s), so
there is no grace and no credit. If the health tick runs before the watcher's first poll after
wake, the heartbeat is 900 s old, which is over 600 s, so amber. Recovery then takes 2 good ticks.
The replay reproduces this on the real code (`unconfirmed-short-sleep`). With a confirmed clock,
the normal macOS path, it cannot happen.

## Replay against the real code

`replay.py --generate` runs 19 scripted scenarios through TLC at production thresholds (300 s
ticks) and commits the model's per-tick behaviour as `traces/*.json`. The scenarios are each of
the 12 fault kinds, an alert across a sleep, a DB failure followed by truncated payloads, sleep, sleep with monitor-first wake, a stall across a wake, and unconfirmed
short and long sleeps. `tests/test_ingest_coverage_replay.py` drives the real code with each trace
for each of the 5 agents (95 runs):

- a real `JSONLWatcher` over the real `default_watch_roots` of a scratch home;
- the real `create_flush_callback` into a scratch DB;
- the real `watcher_health_fact` and `evaluate`;
- JSONL in each agent's own format.

Every tick must match on three things: the ingest state, what the monitor saw
(fresh / old / unreadable), and whether **every** record the agent wrote has reached the DB. A
rewrite's records are checked one by one, not just the newest. Records a fault really drops
(writes landing elsewhere, format drift, an in-place edit the fingerprint cannot see) count as backlog only while that
fault lasts. They stay out of the DB afterwards, which is the finding.

Faults are real where the OS allows it:
- a write-only transcript (lag);
- a renamed root;
- a read-only health directory (stale);
- a file rewritten in place, longer, under the stored offset (detected), and a same-length edit
  past the first 4 KiB (not detected);
- records in a shape no parser takes (drift);
- writes that succeed into a second DB the watcher does not probe (coverage).

Five are stand-ins at a boundary:
- a dead or stuck watcher is "no completed poll";
- a DB failure is raised from the flush callback;
- an unreadable health file is one failed read;
- a partial payload is the health file rewritten with only `updated_at` and `poll_count` before the read;
- the watcher's and flush path's `time` module is `SimTime`: monotonic = awake seconds, wall = the
  tick time (plus 1 s for the watcher once its first window is open, so a window boundary follows
  its poll's own inserts, as it does in production). The watchdog's `now_fn` is the same clock.

The replay can fail, and each of these is a test:
- making the real `_age_state` always green gives 8–9 mismatching ticks;
- flipping one model tick is caught;
- disabling the real `_verify_anchor` makes the detected-rewrite scenario mismatch.

## What this cannot cover

- **Agents BrainLayer is not wired to**, including future labs' agents. Sources are wired on
  purpose ("nothing is auto-detected"), so the guarantee is per wired root. The known-agent census
  (`src/brainlayer/agent_census.py`, below) catches running agents from a fixed list whose
  transcripts land outside the watched roots. It cannot catch an agent that is not on the list. Today, for example, Gemini
  CLI writes `~/.gemini/tmp/<hash>/chats/session-*.json`, which is not
  `~/.gemini/sessions/**/*.jsonl`.
- **Wrong parsing or content**: a record ingested as the wrong text.
- **A provider folder deleted within 14 days of its last transcript write alarms until then**: the
  watcher cannot tell removal from loss. A root with no offsets confirmed in the last 14 days that
  vanished before this process started is not reported.
- **Restarts** are not modelled (in the model "dead" keeps the same process); the restart path of
  `roots_missing` is a unit test on the real watcher.
- **The Swift icon drawing.** That is BrainBar's app tests, not this.
- **Maintenance pause** (red softened to amber, still non-green) and the `t3_ingest_unhealthy` path are
  not modelled.

## Known-agent census (the add-on for unwired agents)

`agent_census.KNOWN_AGENTS` lists claude, codex, cursor-agent, gemini and opencode. Each entry has
its own executable, the entry script an interpreter would run, and its transcript roots with a
filename pattern. Each health check reuses the `ps axo pid=,command=` snapshot.

- **Matching.** An agent is running only when argv[0] is its binary, or argv[0] is node, bun,
  deno or python and the *script* that interpreter runs is the agent's entry script. The
  interpreter's options are interpreted, not skipped: inline code or a module (`python -c`/`-m`
  including clusters like `-uc`, `node -e/--eval/-p`, `bun -e`, `bun x`, `deno eval`) means no
  script at all, and option values (`-W`, `-X`, `-r`, `--require`, ...) are never the script. An
  argument never counts: `grep gemini`, `python3 -c gemini`, `brainlayer … gemini` and
  `--agent gemini` do not match.
- **Tied to the current run.** Start times come from `ps -o pid=,etime=` for the matched PIDs
  only. Only a file matching the agent's transcript pattern and modified after a live
  instance's start counts. For Gemini that pattern is `tmp/*/chats/session-*.json`.
- **When it fires.** If such a file sits under a root that no `default_watch_roots` entry
  covers, three things happen: an `unindexed_agent_running` warning issue, the `ingest` check
  floored at **amber** with "an agent BrainLayer does not index is running: <agent>: <root>", and a
  typed `unindexed_agents` list (code in `context_codes`, seam F21).
- **Per agent, not measured is never "none".** For each agent, a start time that `ps` cannot give,
  or a scan cut short by the 5,000-entry or depth-6 cap or a directory read error (with nothing
  found), makes *that agent* not measured. Any such agent marks the census incomplete:
  `agent_census_incomplete` names it as ingest context, and the typed field `census_incomplete`
  lists it. A finding for one agent never makes another agent's cut-short scan count as measured.
- **Merged per agent; unmeasured holds.** The verdict persists the last census on the ingest check.
  An agent measured this check replaces its entry: a finding is added, a measured negative (not
  running, or no new transcript) removes it. An agent not measured keeps its last entry. If `ps`
  failed, the whole census is unmeasured and everything holds. An amber is therefore cleared only by
  a measured negative for each agent in it.

An empty census changes nothing. The model does not include the census, because it can only add
non-green ticks.

## Keeping it in step

CI runs `check.py --require-sync`. A change to any pinned source fails this job until someone does
three things: re-reads the mapped lines in `coverage.json`, updates the model or adds a switch, and
reruns `check.py`, `replay.py --generate` and the replay test. Only then is the pin moved.
