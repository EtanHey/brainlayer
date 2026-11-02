------------------------------- MODULE DotTable -------------------------------
(***************************************************************************)
(* The BrainBar dot as plan section 3d defines it: a pure function of the   *)
(* outcome checks. Job alerts (job-alerts.json) are notes, never red.       *)
(*                                                                          *)
(* The eleven checks fall into five behaviour classes; one representative   *)
(* per class keeps the state space small (rows -> class):                   *)
(*   aging    backup_local, backup_offsite, transcripts, disk               *)
(*   latched  drive_auth                                                    *)
(*   hyst     ingest, stores, search (two good ticks back to green)         *)
(*   twoState integrity, embedding (no amber column)                        *)
(*   monitor  monitor (never red)                                           *)
(* How each class's raw state is computed is checked in BackupVerdict and   *)
(* PipeVerdict; here raw states are free inputs and only evaluate()'s      *)
(* shared state machine, the dot, badge_alerts() and the alert lifecycle    *)
(* are modelled.                                                            *)
(*                                                                          *)
(* All-fixed values: AlertLightsDot = FALSE, CleanClears = TRUE.            *)
(***************************************************************************)
EXTENDS Integers, FiniteSets

CONSTANTS
    UnknownAmber,    \* unknown_amber_ticks (3)
    AlertLightsDot,  \* pre-#1583: a job alert was a critical badge alert
    CleanClears      \* scratch break when FALSE: a clean run leaves its alert

Checks == {"aging", "latched", "hyst", "twoState", "monitor"}
Jobs == {"maintenance-light", "maintenance-full", "backup-daily"}
Outcomes == Checks \ {"monitor"}
Results(j) == IF j = "maintenance-full"
              THEN {"ok", "failed", "deferred_vacuum_skipped", "deferred_other"}
              ELSE {"ok", "failed", "deferred_other"}
Rank(s) == CASE s = "green" -> 0 [] s = "unknown" -> 1 [] s = "amber" -> 2
             [] s = "red" -> 3 [] OTHER -> 0
DotRank(s) == IF s = "unknown" THEN 0 ELSE Rank(s)

VARIABLES chk, uk, alerts
vars == <<chk, uk, alerts>>

(* The monitor check runs evaluate()'s unknown rule (keep the last state,  *)
(* amber after UnknownAmber blind ticks). The other classes take any state  *)
(* their section 3d columns allow; their transitions are checked elsewhere. *)
Columns(c) == CASE c = "twoState" -> {"green", "red", "unknown"}
                [] OTHER -> {"green", "amber", "red", "unknown"}
MonitorHold(raw) ==
    LET prior == chk["monitor"]
        uk1 == IF raw = "unknown" THEN uk + 1 ELSE 0
    IN IF raw = "green" THEN [state |-> "green", unknown |-> 0]
       ELSE [state |-> IF prior = "red" THEN "red"
                       ELSE IF uk1 >= UnknownAmber THEN "amber"
                       ELSE IF prior = "none" THEN "unknown" ELSE prior,
             unknown |-> IF uk1 > UnknownAmber THEN UnknownAmber ELSE uk1]

Tick ==
    /\ \E raw \in {"green", "unknown"}, o \in [Outcomes -> {"green", "amber", "red", "unknown"}] :
         /\ \A c \in Outcomes : o[c] \in Columns(c)
         /\ LET m == MonitorHold(raw)
            IN /\ chk' = [c \in Checks |-> IF c = "monitor" THEN m.state ELSE o[c]]
               /\ uk' = m.unknown
    /\ UNCHANGED alerts

(* maintenance.main / backup_daily -> job_alerts.report(key, reason|None)   *)
Run(j, r) ==
    /\ alerts' = CASE r = "ok" -> (IF CleanClears THEN alerts \ {j} ELSE alerts)
                   [] r = "failed" -> alerts \cup {j}
                   [] r = "deferred_vacuum_skipped" -> alerts \ {j}
                   [] r = "deferred_other" -> alerts
    /\ UNCHANGED <<chk, uk>>

Jobs1 == \E j \in Jobs : \E r \in Results(j) : Run(j, r)

Init == /\ chk = [c \in Checks |-> "none"]
        /\ uk = 0
        /\ alerts = {}

Next == Tick \/ Jobs1
Spec == Init /\ [][Next]_vars
LiveSpec == Spec /\ \A j \in Jobs : WF_vars(Run(j, "ok"))

-----------------------------------------------------------------------------
(* evaluate()'s dot and badge_alerts(): an outcome_<check> entry is         *)
(* critical only for a red check; job_alert_<key> notes are warnings. The  *)
(* 1.5.83 BrainBar badge lights on any non-warning entry.                   *)
Worst == CHOOSE s \in {"green", "amber", "red"} :
            /\ \A c \in Checks : DotRank(chk[c]) <= Rank(s)
            /\ \/ s = "green"
               \/ \E c \in Checks : DotRank(chk[c]) = Rank(s)
Dot == Worst
CriticalOutcomes == {c \in Checks : chk[c] = "red"}
CriticalAlerts == IF AlertLightsDot THEN alerts ELSE {}
Badge == CriticalOutcomes # {} \/ CriticalAlerts # {}

TypeOK == \A c \in Checks : chk[c] \in {"none", "green", "amber", "red", "unknown"}

(* A job alert never makes the dot red, and the dot cannot be red (or the  *)
(* badge lit) while every outcome check is green.                           *)
NoStuckAlert ==
    /\ Badge => \E c \in Checks : chk[c] = "red"
    /\ Badge <=> Dot = "red"
    /\ (\A c \in Checks : chk[c] = "green") => Dot = "green" /\ ~Badge

(* The monitor check is never red (section 3d: "never red").                *)
MonitorNeverRed == chk["monitor"] # "red"

(* Liveness: each job's next clean run clears its alert.                    *)
AlertClears == \A j \in Jobs : (j \in alerts) ~> (j \notin alerts)
=============================================================================
