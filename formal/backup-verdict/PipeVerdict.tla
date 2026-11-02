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
(* ResetRecovery, ReceiptReplaced, AnchoredBacklog all TRUE.                *)
(***************************************************************************)
EXTENDS Integers

CONSTANTS
    Horizon,
    IngestAmber,       \* ingest_amber_s (10 min)
    IngestRed,         \* ingest_red_s (30 min)
    EmbedRed,          \* embedding_red_s (6 h)
    Gap,               \* sleep_gap_s (15 min)
    Grace,             \* wake_grace_s (2 h)
    CanaryRed,         \* canary_red_failures (3)
    GoodTicks,         \* recovery_good_ticks (2)
    UnknownAmber,      \* unknown_amber_ticks (3)
    CanaryLatch,       \* #1583 B1: an unrun canary keeps its failure; an empty queue never repays it
    GraceNotEvidence,  \* #1583 B2: an age that looks fresh only because of a resume is not a good tick
    DriveLatch,        \* #1583 B4: confirmed Drive loss holds until an observed ok
    ResetRecovery,     \* #1595: any non-green raw tick resets good_ticks
    ReceiptReplaced,   \* #1587 B5: every round-trip attempt replaces the receipt
    AnchoredBacklog,   \* #1587 B4: rows leave the backlog only by getting a vector
    QueueStalls,       \* may the pending-store queue stall (FALSE isolates the canary)
    Focus              \* the check whose world moves and is observed; the checks share
                       \* only tick timing in evaluate(), so each is explored on its own

Checks == {"drive", "ingest", "stores", "embedding"}
Hyst == {"ingest", "stores"}
NoTime == -1
Rank(s) == CASE s = "green" -> 0 [] s = "unknown" -> 1 [] s = "amber" -> 2
             [] s = "red" -> 3 [] OTHER -> 0
Lower(a, b) == Rank(a) < Rank(b)

VARIABLES
    now, alive, hb, storeOK, attempt, success, queueStalled, access,
    backlog, stallSince,                                 \* the world
    lastTick, resumed, st, uk, good, fails, owed, lost, ingClk, embClk, pend, \* the monitor
    prevSt, streak, loose, lastCanary                     \* ghosts

world == <<now, alive, hb, storeOK, attempt, success, queueStalled, access,
           backlog, stallSince>>
mon == <<lastTick, resumed, st, uk, good, fails, owed, lost, ingClk, embClk, pend>>
ghost == <<prevSt, streak, loose, lastCanary>>
vars == <<world, mon, ghost>>

-----------------------------------------------------------------------------
(* The world.                                                               *)
Advance ==
    /\ now < Horizon
    /\ now' = now + 1
    /\ hb' = IF alive /\ Focus = "ingest" THEN now + 1 ELSE hb
    /\ UNCHANGED <<alive, storeOK, attempt, success, queueStalled, access, backlog,
                   stallSince, mon, ghost>>

WatcherStops == alive /\ alive' = FALSE /\ UNCHANGED <<now, hb, storeOK, attempt, success,
                    queueStalled, access, backlog, stallSince, mon, ghost>>
WatcherStarts == ~alive /\ alive' = TRUE /\ hb' = now
                 /\ UNCHANGED <<now, storeOK, attempt, success, queueStalled, access,
                                backlog, stallSince, mon, ghost>>
StorePath == storeOK' = ~storeOK
             /\ UNCHANGED <<now, alive, hb, attempt, success, queueStalled, access, backlog,
                            stallSince, mon, ghost>>
QueueFlips == (QueueStalls \/ queueStalled) /\ queueStalled' = ~queueStalled
              /\ UNCHANGED <<now, alive, hb, storeOK, attempt, success, access, backlog,
                             stallSince, mon, ghost>>
\* The nightly round trip (maintenance light/full). With ReceiptReplaced, every
\* attempt replaces store-roundtrip.json; without it a failed write leaves the
\* previous success readable.
RoundTrip ==
    /\ attempt' = IF storeOK THEN "ok" ELSE "fail"
    /\ success' = IF storeOK THEN "ok"
                  ELSE IF ReceiptReplaced THEN "fail" ELSE success
    /\ UNCHANGED <<now, alive, hb, storeOK, queueStalled, access, backlog, stallSince,
                   mon, ghost>>
DriveLost == access = "ok" /\ access' = "lost"
             /\ UNCHANGED <<now, alive, hb, storeOK, attempt, success, queueStalled, backlog,
                            stallSince, mon, ghost>>
Reconnect == access = "lost" /\ access' = "ok"
             /\ UNCHANGED <<now, alive, hb, storeOK, attempt, success, queueStalled, backlog,
                            stallSince, mon, ghost>>
BacklogStalls == ~backlog /\ backlog' = TRUE /\ stallSince' = now
                 /\ UNCHANGED <<now, alive, hb, storeOK, attempt, success, queueStalled, access,
                                mon, ghost>>
VectorsWritten == backlog
                  /\ \E drained \in BOOLEAN :
                        /\ backlog' = ~drained
                        /\ stallSince' = IF drained THEN NoTime ELSE now
                  /\ UNCHANGED <<now, alive, hb, storeOK, attempt, success, queueStalled, access,
                                 mon, ghost>>

World == \/ Focus = "ingest" /\ (WatcherStops \/ WatcherStarts)
         \/ Focus = "stores" /\ (StorePath \/ QueueFlips \/ RoundTrip)
         \/ Focus = "drive" /\ (DriveLost \/ Reconnect)
         \/ Focus = "embedding" /\ (BacklogStalls \/ VectorsWritten)

-----------------------------------------------------------------------------
(* What one tick can observe.                                               *)
Only(c, S) == IF Focus = c THEN S ELSE {"unobs"}
DriveObs == Only("drive", IF access = "ok" THEN {"unobs", "ok", "refresh_failed"}
                          ELSE {"unobs", "refresh_failed", "consent_needed"})
CanaryObs == Only("stores", {"unobs", success} \ {"none"})
QueueObs == Only("stores", {"unobs", IF queueStalled THEN "stalled" ELSE "empty"})
HbObs == Only("ingest", {"unobs", "seen"})
\* Without the anchored window, a stalled backlog's rows can age out of the
\* 24 h window and read as an empty backlog with no vector written.
EmbObs == Only("embedding", {"unobs", IF backlog THEN "stalled" ELSE "clear"}
                            \cup (IF ~AnchoredBacklog /\ backlog THEN {"clear"} ELSE {}))

-----------------------------------------------------------------------------
(* evaluate()'s shared state machine for one check.                        *)
Resumed1 == IF lastTick # NoTime /\ now - lastTick > Gap THEN now ELSE resumed
ResumedNow == IF Resumed1 # NoTime /\ now - Resumed1 > Grace THEN NoTime ELSE Resumed1

Hold(c, raw, ev) ==
    LET prior == st[c]
        g0 == IF ResetRecovery /\ raw # "green" THEN 0 ELSE good[c]
        uk1 == IF raw = "unknown" THEN uk[c] + 1 ELSE 0
    IN IF raw = "unknown"
       THEN [state |-> IF prior = "red" THEN "red"
                       ELSE IF uk1 >= UnknownAmber THEN "amber"
                       ELSE IF prior \in {"none", "unknown"} THEN "unknown" ELSE prior,
             good |-> g0, unknown |-> uk1]
       ELSE IF ~ev /\ prior \in {"amber", "red"} /\ Lower(raw, prior)
       THEN [state |-> prior, good |-> g0, unknown |-> 0]
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

(* ingest: watcher heartbeat with persisted clock and wake grace           *)
AgeState(a) == IF a > IngestRed THEN "red" ELSE IF a > IngestAmber THEN "amber" ELSE "green"
IngestRaw(o) ==
    LET clock == IF o = "seen" THEN (IF hb > ingClk THEN hb ELSE ingClk) ELSE ingClk
        eff == IF ResumedNow # NoTime /\ ResumedNow > clock THEN ResumedNow ELSE clock
        raw == AgeState(now - eff)
        graced == Lower(raw, AgeState(now - clock))
    IN IF clock = NoTime THEN [raw |-> "unknown", ev |-> FALSE, clk |-> NoTime]
       ELSE [raw |-> raw, clk |-> clock,
             ev |-> IF GraceNotEvidence THEN o = "seen" /\ ~graced ELSE TRUE]

(* stores: oldest pending store, plus the latched round-trip canary         *)
StoresRaw(q, k) ==
    LET f1 == CASE k = "ok" -> 0 [] k = "fail" -> fails + 1 [] OTHER -> fails
        canary == CASE k = "ok" -> "green"
                    [] k = "fail" -> (IF f1 >= CanaryRed THEN "red" ELSE "amber")
                    [] CanaryLatch /\ fails > 0 -> (IF fails >= CanaryRed THEN "red" ELSE "amber")
                    [] OTHER -> "unknown"
        \* _stall_age: an unobserved queue keeps aging a known stall (persisted start).
        pending == CASE q = "empty" -> "green" [] q = "stalled" -> "red"
                     [] pend -> "red" [] OTHER -> "unknown"
        worst == IF canary = "unknown" THEN pending
                 ELSE IF pending = "unknown" THEN canary
                 ELSE IF Lower(pending, canary) THEN canary ELSE pending
        ow == CanaryLatch /\ (owed \/ f1 > 0)
        ev == IF ~CanaryLatch THEN TRUE ELSE IF ow THEN k = "ok" ELSE q # "unobs" \/ k = "ok"
    IN [raw |-> worst, ev |-> ev, fails |-> f1, owed |-> ow,
        pend |-> CASE q = "empty" -> FALSE [] q = "stalled" -> TRUE [] OTHER -> pend]

(* embedding: stall age from the anchored backlog                           *)
EmbRaw(o) ==
    CASE o = "unobs" -> IF embClk = NoTime THEN [raw |-> "unknown", ev |-> FALSE, clk |-> NoTime]
                        ELSE [raw |-> IF now - embClk > EmbedRed THEN "red" ELSE "green",
                              ev |-> FALSE, clk |-> embClk]
      [] o = "clear" -> [raw |-> "green", ev |-> TRUE, clk |-> NoTime]
      [] OTHER -> [raw |-> IF now - stallSince > EmbedRed THEN "red" ELSE "green",
                   ev |-> TRUE, clk |-> stallSince]

-----------------------------------------------------------------------------
(* Positive evidence, per §3d "What clears red", judged from the world.     *)
Demo(c, o) ==
    CASE c = "drive" -> o.drive = "ok"
      [] c = "ingest" -> o.hb = "seen" /\ now - hb <= IngestAmber
      [] c = "stores" -> /\ o.queue # "stalled"
                         /\ \/ o.canary = "ok" /\ attempt = "ok"
                            \/ o.canary = "unobs" /\ o.queue = "empty" /\ lastCanary # "fail"
      [] c = "embedding" -> o.emb # "unobs" /\ (~backlog \/ stallSince > embClk)

Observed(c, o) == CASE c = "drive" -> o.drive # "unobs"
                    [] c = "ingest" -> o.hb # "unobs"
                    [] c = "stores" -> o.canary # "unobs" \/ o.queue # "unobs"
                    [] OTHER -> o.emb # "unobs"

Tick ==
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
            h == [c \in Checks |-> Hold(c, raw[c], ev[c])]
        IN /\ st' = [c \in Checks |-> h[c].state]
           /\ good' = [c \in Checks |-> IF h[c].good > GoodTicks THEN GoodTicks ELSE h[c].good]
           /\ uk' = [c \in Checks |-> IF h[c].unknown > UnknownAmber THEN UnknownAmber
                                      ELSE h[c].unknown]
           /\ fails' = IF s.fails > CanaryRed THEN CanaryRed ELSE s.fails
           /\ owed' = (s.owed /\ h["stores"].state # "green")
           /\ pend' = s.pend
           /\ lost' = d.lost
           /\ ingClk' = i.clk
           /\ embClk' = e.clk
           /\ streak' = [c \in Checks |-> IF Demo(c, o)
                           THEN (IF streak[c] < GoodTicks THEN streak[c] + 1 ELSE GoodTicks)
                           ELSE 0]
           /\ loose' = [c \in Checks |-> IF Demo(c, o)
                           THEN (IF loose[c] < GoodTicks THEN loose[c] + 1 ELSE GoodTicks)
                           ELSE IF Observed(c, o) THEN 0 ELSE loose[c]]
           /\ lastCanary' = IF o.canary = "unobs" THEN lastCanary ELSE attempt
    /\ prevSt' = st
    /\ lastTick' = now
    /\ resumed' = ResumedNow
    /\ UNCHANGED <<now, alive, hb, storeOK, attempt, success, queueStalled, access,
                   backlog, stallSince>>

Init ==
    /\ now = 0 /\ alive = TRUE /\ hb = 0 /\ storeOK = TRUE
    /\ attempt = "none" /\ success = "none" /\ queueStalled = FALSE /\ access = "ok"
    /\ backlog = FALSE /\ stallSince = NoTime
    /\ lastTick = NoTime /\ resumed = NoTime
    /\ st = [c \in Checks |-> "none"] /\ uk = [c \in Checks |-> 0]
    /\ good = [c \in Checks |-> 0] /\ fails = 0 /\ owed = FALSE /\ lost = FALSE
    /\ ingClk = NoTime /\ embClk = NoTime /\ pend = FALSE
    /\ prevSt = [c \in Checks |-> "none"] /\ streak = [c \in Checks |-> 0]
    /\ loose = [c \in Checks |-> 0]
    /\ lastCanary = "none"

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
