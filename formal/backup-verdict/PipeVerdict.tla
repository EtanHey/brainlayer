----------------------------- MODULE PipeVerdict -----------------------------
(***************************************************************************)
(* evaluate() for the latched and hysteresis checks: drive_auth, ingest,    *)
(* stores (pending queue + nightly store round-trip receipt) and embedding. *)
(*                                                                          *)
(* The world: the watcher may stop and restart, the store path may break   *)
(* and heal, Drive access may be lost and reconnected, the embed backlog    *)
(* may stall or progress. Each tick sees any subset of facts (the rest are  *)
(* UNOBSERVED), and the monitor may skip hours.                             *)
(*                                                                          *)
(* All-fixed values: CanaryLatch, GraceNotEvidence, DriveLatch,             *)
(* ResetRecovery, ReceiptReplaced, AnchoredBacklog, ObservedHysteresis,   *)
(* GraceOnce all TRUE.                                                      *)
(*                                                                          *)
(* ingest, embedding and the pending-store queue are aging clocks served by *)
(* _stall_age -> _graced_age: after a monitoring gap the age counts from    *)
(* resumed_at (wake grace), and a graced age is never recovery evidence.   *)
(***************************************************************************)
EXTENDS Integers

CONSTANTS
    Horizon,
    IngestAmber,       \* ingest_amber_s (10 min)
    IngestRed,         \* ingest_red_s (30 min)
    EmbedAmber,        \* embedding_amber_s (1 h)
    EmbedRed,          \* embedding_red_s (6 h)
    StoreAmber,        \* stores_amber_s (15 min)
    StoreRed,          \* stores_red_s (45 min)
    Gap,               \* sleep_gap_s (15 min)
    Grace,             \* wake_grace_s (2 h)
    CanaryRed,         \* canary_red_failures (3)
    GoodTicks,         \* recovery_good_ticks (2)
    UnknownAmber,      \* unknown_amber_ticks (3)
    CanaryLatch,       \* #1583 B1: an unrun canary keeps its failure; an empty queue never repays it
    GraceNotEvidence,  \* #1583 B2: an age that looks fresh only because of a resume is not a good tick
    DriveLatch,        \* #1583 B4: confirmed Drive loss holds until an observed ok
    ResetRecovery,     \* #1595: any non-green raw tick resets good_ticks
    ObservedHysteresis, \* S1a-5 (4e07651b): a tick that is not observed-good resets good_ticks
                       \* (raw not green, no evidence, or graced), and red that observes
                       \* amber stays red: red clears only on consecutive observed good ticks
    ReceiptReplaced,   \* #1587 B5: every round-trip attempt replaces the receipt
    AnchoredBacklog,   \* #1587 B4: rows leave the backlog only by getting a vector
    GraceOnce,         \* S1a-6 (plan 3d wake-grace ruling): one grace window per stall
                       \* episode (keyed by its clock value), never renewed by a later gap;
                       \* FALSE = today's renewable grace (resumed_at resets on every gap)
    QueueStalls,       \* may the pending-store queue stall (FALSE isolates the canary)
    TickAt,            \* hours at which the monitor may tick
    MustTick,          \* scripted schedules: the monitor must tick at every TickAt hour
    AlwaysObserve,     \* every tick observes the focused fact (scripted schedules)
    Focus              \* the check whose world moves and is observed; the checks share
                       \* only tick timing in evaluate(), so each is explored on its own

Checks == {"drive", "ingest", "stores", "embedding"}
Hyst == {"ingest", "stores"}
Stall == {"ingest", "embedding", "stores"}    \* the clocks _graced_age serves
NoTime == -1
Rank(s) == CASE s = "green" -> 0 [] s = "unknown" -> 1 [] s = "amber" -> 2
             [] s = "red" -> 3 [] OTHER -> 0
Lower(a, b) == Rank(a) < Rank(b)

VARIABLES
    now, alive, hb, storeOK, attempt, success, pendStart, access,
    backlog, stallSince,                                 \* the world (pendStart: oldest pending store)
    lastTick, resumed, st, uk, good, fails, owed, lost, ingClk, embClk, pendClk, ep, \* the monitor
    prevSt, streak, loose, lastCanary, lastObs, ovr       \* ghosts (lastObs: what the last tick saw)

world == <<now, alive, hb, storeOK, attempt, success, pendStart, access,
           backlog, stallSince>>
mon == <<lastTick, resumed, st, uk, good, fails, owed, lost, ingClk, embClk, pendClk, ep>>
ghost == <<prevSt, streak, loose, lastCanary, lastObs, ovr>>
vars == <<world, mon, ghost>>

-----------------------------------------------------------------------------
(* The world.                                                               *)
Advance ==
    /\ now < Horizon
    /\ MustTick /\ now \in TickAt => lastTick = now
    /\ now' = now + 1
    /\ hb' = IF alive /\ Focus = "ingest" THEN now + 1 ELSE hb
    /\ UNCHANGED <<alive, storeOK, attempt, success, pendStart, access, backlog,
                   stallSince, mon, ghost>>

WatcherStops == alive /\ alive' = FALSE /\ UNCHANGED <<now, hb, storeOK, attempt, success,
                    pendStart, access, backlog, stallSince, mon, ghost>>
WatcherStarts == ~alive /\ alive' = TRUE /\ hb' = now
                 /\ UNCHANGED <<now, storeOK, attempt, success, pendStart, access,
                                backlog, stallSince, mon, ghost>>
StorePath == storeOK' = ~storeOK
             /\ UNCHANGED <<now, alive, hb, attempt, success, pendStart, access, backlog,
                            stallSince, mon, ghost>>
QueueFlips == \/ QueueStalls /\ pendStart = NoTime /\ pendStart' = now   \* a store starts waiting
              \/ pendStart # NoTime /\ pendStart' = NoTime                  \* the queue drains
              /\ UNCHANGED <<now, alive, hb, storeOK, attempt, success, access, backlog,
                             stallSince, mon, ghost>>
\* The nightly round trip (maintenance light/full). With ReceiptReplaced, every
\* attempt replaces store-roundtrip.json; without it a failed write leaves the
\* previous success readable.
RoundTrip ==
    /\ attempt' = IF storeOK THEN "ok" ELSE "fail"
    /\ success' = IF storeOK THEN "ok"
                  ELSE IF ReceiptReplaced THEN "fail" ELSE success
    /\ UNCHANGED <<now, alive, hb, storeOK, pendStart, access, backlog, stallSince,
                   mon, ghost>>
DriveLost == access = "ok" /\ access' = "lost"
             /\ UNCHANGED <<now, alive, hb, storeOK, attempt, success, pendStart, backlog,
                            stallSince, mon, ghost>>
Reconnect == access = "lost" /\ access' = "ok"
             /\ UNCHANGED <<now, alive, hb, storeOK, attempt, success, pendStart, backlog,
                            stallSince, mon, ghost>>
BacklogStalls == ~backlog /\ backlog' = TRUE /\ stallSince' = now
                 /\ UNCHANGED <<now, alive, hb, storeOK, attempt, success, pendStart, access,
                                mon, ghost>>
VectorsWritten == backlog
                  /\ \E drained \in BOOLEAN :
                        /\ backlog' = ~drained
                        /\ stallSince' = IF drained THEN NoTime ELSE now
                  /\ UNCHANGED <<now, alive, hb, storeOK, attempt, success, pendStart, access,
                                 mon, ghost>>

World == \/ Focus = "ingest" /\ (WatcherStops \/ WatcherStarts)
         \/ Focus = "stores" /\ (StorePath \/ QueueFlips \/ RoundTrip)
         \/ Focus = "drive" /\ (DriveLost \/ Reconnect)
         \/ Focus = "embedding" /\ (BacklogStalls \/ VectorsWritten)

-----------------------------------------------------------------------------
(* What one tick can observe.                                               *)
Only(c, S) == IF Focus # c THEN {"unobs"} ELSE IF AlwaysObserve /\ S # {"unobs"} THEN S \ {"unobs"} ELSE S
DriveObs == Only("drive", IF access = "ok" THEN {"unobs", "ok", "refresh_failed"}
                          ELSE {"unobs", "refresh_failed", "consent_needed"})
CanaryObs == Only("stores", {"unobs", success} \ {"none"})
QueueObs == Only("stores", {"unobs", IF pendStart # NoTime THEN "pending" ELSE "empty"})
HbObs == Only("ingest", {"unobs", "seen"})
\* Without the anchored window, a stalled backlog's rows can age out of the
\* 24 h window and read as an empty backlog with no vector written.
EmbObs == Only("embedding", {"unobs", IF backlog THEN "stalled" ELSE "clear"}
                            \cup (IF ~AnchoredBacklog /\ backlog THEN {"clear"} ELSE {}))

-----------------------------------------------------------------------------
(* evaluate()'s shared state machine for one check.                        *)
Resumed1 == IF lastTick # NoTime /\ now - lastTick > Gap THEN now ELSE resumed
ResumedNow == IF Resumed1 # NoTime /\ now - Resumed1 > Grace THEN NoTime ELSE Resumed1

Hold(c, raw, ev, graced) ==
    LET prior == st[c]
        g0 == IF ObservedHysteresis
              THEN (IF raw # "green" \/ ~ev \/ graced THEN 0 ELSE good[c])
              ELSE IF ResetRecovery /\ raw # "green" THEN 0 ELSE good[c]
        uk1 == IF raw = "unknown" THEN uk[c] + 1 ELSE 0
    IN IF raw = "unknown"
       THEN [state |-> IF prior = "red" THEN "red"
                       ELSE IF uk1 >= UnknownAmber THEN "amber"
                       ELSE IF prior \in {"none", "unknown"} THEN "unknown" ELSE prior,
             good |-> g0, unknown |-> uk1]
       ELSE IF ~ev /\ prior \in {"amber", "red"} /\ Lower(raw, prior)
       THEN [state |-> prior, good |-> g0, unknown |-> 0]
       ELSE IF ObservedHysteresis /\ c \in Hyst /\ prior = "red" /\ raw = "amber"
       THEN [state |-> "red", good |-> g0, unknown |-> 0]
       ELSE IF c \in Hyst /\ raw = "green" /\ prior \in {"amber", "red"}
       THEN IF g0 + 1 < GoodTicks
            THEN [state |-> prior, good |-> g0 + 1, unknown |-> 0]
            ELSE [state |-> "green", good |-> 0, unknown |-> 0]
       ELSE [state |-> raw, good |-> 0, unknown |-> 0]

(* _drive_auth                                                              *)
DriveRaw(o) ==
    CASE o = "unobs" -> [raw |-> "unknown", ev |-> FALSE, lost |-> lost]
      [] o = "ok" -> [raw |-> "green", ev |-> TRUE, lost |-> FALSE]
      [] o = "consent_needed" \/ (DriveLatch /\ lost) -> [raw |-> "red", ev |-> TRUE, lost |-> TRUE]
      [] OTHER -> [raw |-> "amber", ev |-> TRUE, lost |-> FALSE]

AgeSt(a, amber, red) == IF a > red THEN "red" ELSE IF a > amber THEN "amber" ELSE "green"
Amber(c) == CASE c = "ingest" -> IngestAmber [] c = "embedding" -> EmbedAmber [] OTHER -> StoreAmber
Red(c) == CASE c = "ingest" -> IngestRed [] c = "embedding" -> EmbedRed [] OTHER -> StoreRed

(* Wake grace for clock k of check c, as _graced_age computes it.          *)
(* Renewable (today, GraceOnce = FALSE): the age counts from the global     *)
(* resumed_at, which every monitoring gap renews.                          *)
(* GraceOnce (S1a-6, 8eecb5ee): a window [from, from + Grace] persisted per *)
(* check as (grace_clock, grace_started_at, grace_until). The prior record  *)
(* is reused when grace_clock = k, expired or not; otherwise a window opens *)
(* only if a resume is in effect and k < resumed_at. The record persists    *)
(* only on ticks that run this path, so a cleared or unknown clock drops it.*)
NoRec == [clk |-> NoTime, from |-> NoTime]
GraceRec(c, k) ==
    IF ~GraceOnce THEN NoRec
    ELSE IF ep[c].clk = k THEN ep[c]
    ELSE IF ResumedNow # NoTime /\ k < ResumedNow THEN [clk |-> k, from |-> ResumedNow]
    ELSE NoRec

(* _graced_age: state from the effective age, graced when wake grace made   *)
(* it milder than the true age; a graced age is never evidence.            *)
GracedAge(c, start, observed) ==
    LET rec == GraceRec(c, start)
        eff == IF ~GraceOnce
               THEN (IF ResumedNow # NoTime /\ ResumedNow > start THEN ResumedNow ELSE start)
               ELSE IF rec.from # NoTime /\ now <= rec.from + Grace /\ rec.from > start
                    THEN rec.from ELSE start
        raw == AgeSt(now - eff, Amber(c), Red(c))
        graced == Lower(raw, AgeSt(now - start, Amber(c), Red(c)))
    IN [raw |-> raw, clk |-> start, graced |-> graced, rec |-> rec,
        ev |-> IF GraceNotEvidence THEN observed /\ ~graced ELSE TRUE]

(* _stall_age: observed None = nothing pending (green, evidence); an        *)
(* unobserved fact keeps aging the persisted start, or is unknown.          *)
StallAge(c, obs, start, persisted) ==
    CASE obs = "unobs" -> IF persisted = NoTime
                          THEN [raw |-> "unknown", ev |-> FALSE, clk |-> NoTime, graced |-> FALSE, rec |-> NoRec]
                          ELSE GracedAge(c, persisted, FALSE)
      [] start = NoTime -> [raw |-> "green", ev |-> TRUE, clk |-> NoTime, graced |-> FALSE, rec |-> NoRec]
      [] OTHER -> GracedAge(c, start, TRUE)

(* ingest: watcher heartbeat = max(observed, persisted), via _graced_age   *)
IngestRaw(o) ==
    LET clock == IF o = "seen" THEN (IF hb > ingClk THEN hb ELSE ingClk) ELSE ingClk
    IN IF clock = NoTime THEN [raw |-> "unknown", ev |-> FALSE, clk |-> NoTime, graced |-> FALSE, rec |-> NoRec]
       ELSE LET g == GracedAge("ingest", clock, TRUE)
            IN [g EXCEPT !.ev = IF GraceNotEvidence THEN g.ev /\ o = "seen" ELSE TRUE]

(* stores: oldest pending store (_stall_age), plus the latched canary      *)
StoresRaw(q, k) ==
    LET f1 == CASE k = "ok" -> 0 [] k = "fail" -> fails + 1 [] OTHER -> fails
        canary == CASE k = "ok" -> "green"
                    [] k = "fail" -> (IF f1 >= CanaryRed THEN "red" ELSE "amber")
                    [] CanaryLatch /\ fails > 0 -> (IF fails >= CanaryRed THEN "red" ELSE "amber")
                    [] OTHER -> "unknown"
        p == StallAge("stores", q, IF q = "pending" THEN pendStart ELSE NoTime, pendClk)
        worst == IF canary = "unknown" THEN p.raw
                 ELSE IF p.raw = "unknown" THEN canary
                 ELSE IF Lower(p.raw, canary) THEN canary ELSE p.raw
        ow == CanaryLatch /\ (owed \/ f1 > 0)
        ev == IF ~CanaryLatch THEN TRUE
              ELSE ~p.graced /\ (IF ow THEN k = "ok" ELSE p.ev \/ k = "ok")
    IN [raw |-> worst, ev |-> ev, fails |-> f1, owed |-> ow, clk |-> p.clk, graced |-> p.graced,
        rec |-> p.rec]

(* embedding: backlog stall start from the anchored window, via _stall_age *)
EmbRaw(o) == StallAge("embedding", o, IF o = "stalled" THEN stallSince ELSE NoTime, embClk)

-----------------------------------------------------------------------------
(* Positive evidence, per §3d "What clears red", judged from the world.     *)
Demo(c, o) ==
    CASE c = "drive" -> o.drive = "ok"
      [] c = "ingest" -> o.hb = "seen" /\ now - hb <= IngestAmber
      [] c = "stores" -> LET qok == \/ o.queue = "empty"
                                    \/ o.queue = "pending" /\ now - pendStart <= StoreAmber
                         IN /\ o.queue # "pending" \/ qok
                            /\ \/ o.canary = "ok" /\ attempt = "ok"
                               \/ o.canary = "unobs" /\ qok /\ lastCanary # "fail"
      [] c = "embedding" -> o.emb # "unobs" /\ (~backlog \/ stallSince > embClk)

Observed(c, o) == CASE c = "drive" -> o.drive # "unobs"
                    [] c = "ingest" -> o.hb # "unobs"
                    [] c = "stores" -> o.canary # "unobs" \/ o.queue # "unobs"
                    [] OTHER -> o.emb # "unobs"

Tick ==
    /\ now \in TickAt
    /\ lastTick < now
    /\ \E o \in [drive : DriveObs, hb : HbObs, canary : CanaryObs, queue : QueueObs,
                 emb : EmbObs] :
        LET d == DriveRaw(o.drive)
            i == IngestRaw(o.hb)
            s == StoresRaw(o.queue, o.canary)
            e == EmbRaw(o.emb)
            raw == [c \in Checks |-> CASE c = "drive" -> d.raw [] c = "ingest" -> i.raw
                                      [] c = "stores" -> s.raw [] OTHER -> e.raw]
            ev == [c \in Checks |-> CASE c = "drive" -> d.ev [] c = "ingest" -> i.ev
                                     [] c = "stores" -> s.ev [] OTHER -> e.ev]
            gr == [c \in Checks |-> CASE c = "ingest" -> i.graced [] c = "stores" -> s.graced
                                     [] c = "embedding" -> e.graced [] OTHER -> FALSE]
            clk == [c \in Stall |-> CASE c = "ingest" -> i.clk [] c = "stores" -> s.clk
                                     [] OTHER -> e.clk]
            h == [c \in Checks |-> Hold(c, raw[c], ev[c], gr[c])]
        IN /\ st' = [c \in Checks |-> h[c].state]
           /\ good' = [c \in Checks |-> IF h[c].good > GoodTicks THEN GoodTicks ELSE h[c].good]
           /\ uk' = [c \in Checks |-> IF h[c].unknown > UnknownAmber THEN UnknownAmber
                                      ELSE h[c].unknown]
           /\ fails' = IF s.fails > CanaryRed THEN CanaryRed ELSE s.fails
           /\ owed' = (s.owed /\ h["stores"].state # "green")
           /\ pendClk' = s.clk
           /\ ep' = [c \in Stall |-> CASE c = "ingest" -> i.rec [] c = "stores" -> s.rec
                                     [] OTHER -> e.rec]
           /\ ovr' = [c \in Stall |->
                 LET k == clk[c]
                     base == IF ovr[c].clk = k THEN ovr[c] ELSE [clk |-> k, first |-> NoTime]
                     overNonRed == k # NoTime /\ now - k > Red(c) /\ h[c].state # "red"
                 IN IF overNonRed /\ base.first = NoTime
                    THEN [base EXCEPT !.first = IF ResumedNow = NoTime THEN -2 ELSE ResumedNow]
                    ELSE base]
           /\ lost' = d.lost
           /\ ingClk' = i.clk
           /\ embClk' = e.clk
           /\ streak' = [c \in Checks |-> IF Demo(c, o)
                           THEN (IF streak[c] < GoodTicks THEN streak[c] + 1 ELSE GoodTicks)
                           ELSE 0]
           /\ loose' = [c \in Checks |-> IF Demo(c, o)
                           THEN (IF loose[c] < GoodTicks THEN loose[c] + 1 ELSE GoodTicks)
                           ELSE IF Observed(c, o) THEN 0 ELSE loose[c]]
           /\ lastObs' = o
           /\ lastCanary' = IF o.canary = "unobs" THEN lastCanary ELSE attempt
    /\ prevSt' = st
    /\ lastTick' = now
    /\ resumed' = ResumedNow
    /\ UNCHANGED <<now, alive, hb, storeOK, attempt, success, pendStart, access,
                   backlog, stallSince>>

Init ==
    /\ now = 0 /\ alive = TRUE /\ hb = 0 /\ storeOK = TRUE
    /\ attempt = "none" /\ success = "none" /\ pendStart = NoTime /\ access = "ok"
    /\ backlog = FALSE /\ stallSince = NoTime
    /\ lastTick = NoTime /\ resumed = NoTime
    /\ st = [c \in Checks |-> "none"] /\ uk = [c \in Checks |-> 0]
    /\ good = [c \in Checks |-> 0] /\ fails = 0 /\ owed = FALSE /\ lost = FALSE
    /\ ingClk = NoTime /\ embClk = NoTime /\ pendClk = NoTime
    /\ ep = [c \in Stall |-> NoRec]
    /\ ovr = [c \in Stall |-> [clk |-> NoTime, first |-> NoTime]]
    /\ prevSt = [c \in Checks |-> "none"] /\ streak = [c \in Checks |-> 0]
    /\ loose = [c \in Checks |-> 0]
    /\ lastCanary = "none"
    /\ lastObs = [drive |-> "unobs", hb |-> "unobs", canary |-> "unobs", queue |-> "unobs",
                  emb |-> "unobs"]

Next == World \/ Advance \/ Tick
Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
TypeOK == \A c \in Checks : st[c] \in {"none", "green", "amber", "red", "unknown"}

(* A red check leaves red only after the evidence §3d names: an observed    *)
(* Drive ok; for ingest and stores, GoodTicks consecutive ticks that each   *)
(* observed the pipe healthy (a fresh heartbeat; a canary success that is   *)
(* the latest attempt, or an empty queue with no failure outstanding); for  *)
(* embedding, an observed clear backlog that vectors actually cleared.      *)
RedHoldsUntilEvidence ==
    \A c \in Checks : prevSt[c] = "red" /\ st[c] # "red" =>
        IF c \in Hyst THEN streak[c] >= GoodTicks ELSE streak[c] >= 1

(* A stall whose true (un-graced) age is past red is non-red only inside    *)
(* one wake-grace window: the window of the first resume seen while it was *)
(* over red, never one granted by a later gap (plan 3d wake-grace ruling). *)
(* first = -2 records "over red and non-red with no resume at all".         *)
PersistedClk(c) == CASE c = "ingest" -> ingClk [] c = "embedding" -> embClk [] OTHER -> pendClk
NoFalseGreenStall ==
    \A c \in Stall :
        LET k == PersistedClk(c) IN
        (k # NoTime /\ lastTick - k > Red(c) /\ st[c] \in {"green", "amber"}) =>
            /\ ovr[c].clk = k /\ ovr[c].first >= 0
            /\ lastTick <= ovr[c].first + Grace

(* Section 3d's "2 consecutive good ticks", for the step to green alone.    *)
RedToGreenNeedsConsecutive ==
    \A c \in Hyst : prevSt[c] = "red" /\ st[c] = "green" => streak[c] >= GoodTicks

(* The code's own reading (health_verdict docstring: "good observations   *)
(* with no bad one between"): a red hysteresis check reaches green only    *)
(* after GoodTicks good observations since the last bad one; a tick that    *)
(* observed nothing neither counts nor resets.                              *)
RedClearsToGreenOnlyOnEvidence ==
    \A c \in Hyst : prevSt[c] = "red" /\ st[c] = "green" => loose[c] >= GoodTicks
=============================================================================
