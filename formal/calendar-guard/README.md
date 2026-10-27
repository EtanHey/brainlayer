# Calendar guard verification

Status: **NOT_GREEN** for the full calendar contract. The guarded-worker fix
passes the bounded cases; lifecycle and journal gaps remain. Runtime code is unchanged.

Run from the repository root with Python 3.11+, Java and TLC 1.7.4 installed at
`~/.local/bin/tlc` (the wrapper uses `~/.local/share/tlaplus/tla2tools-1.7.4.jar`):

```sh
formal/calendar-guard/check.sh --scratch-break
```

Exit **1** means the documented current-source findings remain; **2** means
an unexpected result. `--expect-known --scratch-break` returns 0 only when all
expected passes and counterexamples reproduce, and still prints `NOT_GREEN`.
With `--case`, a passing selection returns 0 and a documented failure returns 1.
The full current-source contract remains `NOT_GREEN` in either case.
Logs, run configs, state files, scratch code and `results.json` live in
`.build.noindex/calendar-guard/<UTC timestamp>-<unique suffix>/`.
The generated reference configs in `configs/` are committed. The checker
compares them with the source projections before starting TLC.
Each TLC process uses `-workers 1` and a 512 MiB heap, sequentially. Before
every process, the checker reads `sysctl -n vm.loadavg`; one-minute load above
40 delays startup for at most ten minutes, then fails without starting TLC.
Each result records the observed load and wait. This host-pressure runner
requires macOS's `vm.loadavg` sysctl.

The pinned target is main `f01469cfdac1368d54a9f9d47bf743f5fc49ca8d`
(#1517/#1518/#1520) with calendar source from #1519
`a9c3f184821006dd29b40f2758bee1ba2df4afc5`. The checker reads Git objects, verifies
their SHA-256 fingerprints and refuses unexpected checkout source drift. It does
not silently retarget to a later main. Fetch the named historical commits if absent.

`coverage.json` maps every modeled action to the pinned source's path:line and
records bounds and atomicity. When covered code changes, review the mapping,
source projection and findings before advancing the pins. Then rerun the scratch
break. CI wiring is a follow-up, outside this PR.

| Promise | Check | Scope / finding |
|---|---|---|
| A run terminates and releases custody within budget | `BoundedTermination`, `EventuallyReleased` | Guarded workers pass; pre-guard and post-guard wedges fail |
| Healthy operations do not provoke SIGKILL | `NoFalseKill` | Finite off-loop stalls pass; historical synchronous I/O fails |
| One child, after durable custody and registration | `ExactlyOneChild` | Registration **write** precedes release; actual guard receipt is not acknowledged |
| Each started run has one reported logical outcome | `Custody`, `AtMostOneTerminal` | Recent nightly passes; recovery, older and overwritten pending runs fail |
| Each run has at most one physical terminal record | `TerminalRecordUnique` | Fails: recovery writes both abandoned and synthetic summary |
| A torn journal cannot stop the next run | `TornJournalLive` | Passes with skipped torn rows and separated appends |
| Recovery does not signal a reused PID | `PidReuseSafe` | Two birth generations; actual recovery function probed too |

`CalendarGuard.tla` explores one step with independently scheduled durable
record, registration write, guard receipt, gate release, child finish, H/Q,
parent death and guard reaping. `CalendarJournal.tla` starts after custody can
be reacquired, explores normal/nightly and hourly records, two pending identities,
one interrupted abandonment publication, and a subsequent run. These are
complementary abstractions, **not a composed refinement proof**.

Time is a bounded logical clock: operation budget 4, H budget 2, Q budget 3,
finite stalls 0/1/3 and horizon 40 ticks. These preserve deadline ordering, not
production seconds (120/30/15 defaults). Advancing time and internal actions are
weakly fair; loop/guard scheduling and successful physical reap are assumed within
their represented budgets. Unbounded host descheduling, failed signals,
uninterruptible kernel tasks, filesystem durability after power loss, reboot,
multiple steps and real SQL/backup semantics are outside passing claims.

The runner executes AST-extracted **real Python function bodies** against
disposable files and fake OS boundaries: `pulse`, `record`, `_recover_abandoned`,
`run_guard`, `Custody.recover`. It never imports BrainLayer or connects to live
services. The scratch test copies the actual calendar module, removes pulse's
overdue condition there, observes continued H from that body, derives `Budgeted=FALSE`,
and reruns the unchanged model. This is source-coupled abstraction evidence,
not execution of the full daemon or a native installed acceptance test.

See [COUNTEREXAMPLES.md](COUNTEREXAMPLES.md) for source-grounded traces and
[RESULTS.md](RESULTS.md) for the measured run.

To run the M1 verification, fetch both branches on the machine assigned by the
lead. Keep the checkout on `fix/tla-calendar-guard`; the scratch branch is input
data and must never merge to main.

```sh
git fetch origin fix/tla-calendar-guard scratch/t2-calendar-guard-budget-20261027
```

Run these commands sequentially from the modeling checkout:

| Case | Command | Expected result |
|---|---|---|
| Historical a | `formal/calendar-guard/check.sh --case historical-sync` | `NoFalseKill`, TLC 12, wrapper 1 |
| Historical b | `formal/calendar-guard/check.sh --case historical-missing-terminal` | `Custody`, TLC 12, wrapper 1 |
| Historical c | `formal/calendar-guard/check.sh --case historical-worker-wedge --case historical-worker-liveness` | `BoundedTermination` and `EventuallyReleased`, TLC 12 and 13, wrapper 1 |
| Current | `formal/calendar-guard/check.sh --case current` | Nine bounded passes and nine retained current findings, wrapper 1 / `NOT_GREEN` |
| Scratch break | `formal/calendar-guard/check.sh --case scratch-worker-wedge --scratch-break --scratch-ref origin/scratch/t2-calendar-guard-budget-20261027` | `BoundedTermination`, TLC 12, wrapper 1 |

Add `--prepare-only` to validate source probes, config selection and committed
configs without starting TLC. The scratch ref carries the full real calendar
module at `formal/calendar-guard/scratch/maintenance_calendar.py`, with only
`if not overdue:` replaced by `if True:`. Runtime paths remain unchanged on
both branches. The checker
copies and executes that function body, records both source hashes, and runs
the unchanged TLA+ model. Any unexpected source, config or verdict returns 2.

Return each run directory's `cases.json`, `results.json`, `probes.json`, configs
and `.log` files to the lead. Generated state files can stay on the runner.
