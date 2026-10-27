# Calendar guard verification

Status: **NOT_GREEN** for the full calendar contract. The guarded-worker fix
passes the bounded cases; lifecycle and journal gaps remain. Runtime code is unchanged.

Run from the repository root with Python 3.11+, Java and TLC 1.7.4 installed at
`~/.local/bin/tlc` (the wrapper uses `~/.local/share/tlaplus/tla2tools-1.7.4.jar`):

```sh
formal/calendar-guard/check.sh --scratch-break
```

Set `TLC` to use a different executable path (including a rig-local wrapper):

```sh
TLC=/path/to/rig/bin/tlc formal/calendar-guard/check.sh --scratch-break
```

When unset, `TLC` defaults to `~/.local/bin/tlc`. A leading `~` is expanded;
relative paths resolve against the caller’s working directory before the launcher
changes directory. The value is one executable path, not a shell command with arguments.
`check.sh` forwards the environment unchanged. To probe executable selection
without starting TLC, run `pytest -q formal/calendar-guard/test_check.py`.

Exit **1** means the documented current-source findings remain; **2** means
an incomplete or unexpected result, including a probe crash. The source banner
does not assert a tool version; each result records `tlc_version` from actual
TLC output, or null if absent. Different versions are recorded, not rejected. `--expect-known --scratch-break` returns 0 only when all
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
| Recovery does not signal a reused PID | `PidReuseSafe` | Probe-verified: actual recovery function executes against a mismatched birth; TLA `BirthChecked=TRUE` is a hard-coded constant |

`CalendarGuard.tla` explores one step with independently scheduled durable
record, registration write, guard receipt, gate release, child finish, H/Q,
parent death and guard reaping. `CalendarJournal.tla` starts after custody can
be reacquired, explores normal/nightly and hourly records, two pending identities,
one interrupted abandonment publication, and a subsequent run. These are
complementary abstractions, **not a composed refinement proof**.

`ExactlyOneChild` is true by construction in these models and has not been
shown failing; a future source-order projection/negative test is needed.
`registration-ack` checks a stronger receipt-before-exec requirement, not a
proven runtime cleanup defect. `finding-two-pending` injects two pending rows
without the intervening pre-run recovery scan; it is an abstraction finding
(reachable in the real sequence only with a skipped or failed earlier scan).
The literal terminal-record check also differs from the designed one-logical-outcome
contract. These traces must not automatically become runtime fix PRs.

Outer process bounds are **not modeled**: the nightly is a plain launchd
`StartCalendarInterval` job with no run timeout (`scripts/launchd/com.brainlayer.maintenance.plist`;
`ExitTimeOut=30` bounds shutdown, not runtime). Hourly transcript recovery runs
under the supervisor with a **7,500 s** run timeout (`supervisor_jobs.py`),
which reaps its group on timeout (`service_supervisor.py`). The unbounded-wedge
findings therefore have different external limits on the two entry points.

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
| Current | `formal/calendar-guard/check.sh --case current` | Ten bounded passes and nine retained current findings, wrapper 1 / `NOT_GREEN` |
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

Round-1 model regression configs: `current-completed-then-pending` must pass
`Custody` after scanning run 1’s summary and publishing run 2’s abandonment.
`probe-parent-before-receipt` deliberately violates `ParentDeathAfterReceipt`
to prove reachability after gate release, before guard receipt. It is a
reachability probe, **not an additional product finding**. The expanded full
run contains 25 outcomes. Parent-death cleanup drains the sent registration
before reap; the stronger receipt-before-exec assertion still fails.
