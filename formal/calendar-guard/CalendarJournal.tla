-------------------------- MODULE CalendarJournal --------------------------
EXTENDS Naturals, Sequences, FiniteSets, TLC
CONSTANTS DetectAbandoned, SkipTorn, BirthChecked, Kind, Crash, Torn, Reuse, Late, TwoPending, LostStart
VARIABLE s
vars == <<s>>
(* One date per run identity; a retry may use the same scheduled_date. The
   abandoned event + synthetic summary are ONE logical terminal outcome. *)
Row(event, run) == [event |-> event, run |-> run, kind |-> Kind,
                    scheduled_date |-> "2026-10-27", birth |-> 1]
TerminalEvents == {"summary", "abandoned"}
Init == s = [pc |-> "runner", journal |-> <<>>, scan |-> 1, latest |-> 0,
             terminal |-> {}, abandoned |-> {}, interrupted |-> FALSE,
             outcomes |-> [r \in 1..2 |-> 0], observed |-> {},
             owned |-> TRUE, birth |-> 1, wrongKill |-> FALSE, nextRan |-> FALSE]
Runner == /\ s.pc = "runner"
          /\ s' = [s EXCEPT !.journal = Append(@, Row(IF LostStart THEN "torn" ELSE
                              IF Kind = "recovery" THEN "transcript-recovery" ELSE "runner", 1)),
                    !.pc = IF LostStart THEN "next" ELSE "spawn", !.owned = ~LostStart]
Spawn == /\ s.pc = "spawn"
         /\ s' = [s EXCEPT !.journal = Append(@, Row("spawn", 1)), !.pc = "finished"]
Finish == /\ s.pc = "finished"
          /\ s' = [s EXCEPT !.journal = Append(@, Row("finished", 1)), !.pc = "summary"]
Summary == /\ s.pc = "summary"
           /\ s' = [s EXCEPT !.journal = Append(@, Row("summary", 1)),
                     !.outcomes[1] = @ + 1, !.owned = FALSE, !.pc = "next"]
CrashRunner ==
    /\ Crash /\ s.pc \in {"spawn", "finished", "summary"}
    /\ s' = [s EXCEPT !.pc = "next", !.owned = FALSE,
              !.birth = IF Reuse THEN 2 ELSE 0,
              !.journal = IF Torn THEN Append(@, Row("torn", 0)) ELSE @]
SecondPending == /\ TwoPending /\ s.pc = "next" /\ Len(s.journal) < 6
                 /\ ~\E i \in 1..Len(s.journal): s.journal[i].run = 2
                 /\ s' = [s EXCEPT !.journal = Append(@, Row("runner", 2))]
AcquireNext == /\ s.pc = "next" /\ ~s.owned
               /\ (~TwoPending \/ \E i \in 1..Len(s.journal): s.journal[i].run = 2)
               /\ s' = [s EXCEPT !.pc = "scan", !.owned = TRUE]
(* Birth identity mismatch excludes the reused PID from Custody.recover's
   kill set; _recover_abandoned counts the original identity as dead. *)
RecoverGroups == /\ s.pc = "scan"
                 /\ s' = [s EXCEPT !.pc = "read", !.wrongKill = s.birth = 2 /\ ~BirthChecked]
ReadLine ==
    /\ s.pc = "read" /\ s.scan <= Len(s.journal)
    /\ LET row == s.journal[s.scan] IN
       s' = [s EXCEPT !.scan = @ + 1,
             !.pc = IF row.event = "torn" /\ ~SkipTorn THEN "error" ELSE @,
             !.latest = IF row.event = "runner" THEN row.run ELSE @,
             !.abandoned = IF row.event = "abandoned" THEN @ \cup {row.run} ELSE @,
             !.terminal = IF row.event = "summary" THEN @ \cup {row.run} ELSE @]
PublishAbandoned ==
    /\ s.pc = "read" /\ s.scan > Len(s.journal)
    /\ DetectAbandoned /\ s.latest # 0 /\ s.latest \notin s.terminal /\ ~Late
    /\ s' = [s EXCEPT !.journal = IF s.latest \in s.abandoned THEN @
                                      ELSE Append(@, Row("abandoned", s.latest)),
              !.outcomes[s.latest] = IF s.latest \in s.abandoned THEN @ ELSE @ + 1,
              !.abandoned = @ \cup {s.latest}, !.pc = "abandon-summary"]
AbandonSummary ==
    /\ s.pc = "abandon-summary"
    /\ s' = [s EXCEPT !.journal = Append(@, Row("summary", s.latest)),
              !.terminal = @ \cup {s.latest}, !.observed = @ \cup {s.latest}, !.pc = "run-next"]
InterruptPublication ==
    /\ s.pc = "abandon-summary" /\ ~s.interrupted
    /\ s' = [s EXCEPT !.pc = "next", !.owned = FALSE, !.interrupted = TRUE,
              !.scan = 1, !.latest = 0, !.terminal = {}, !.abandoned = {}]
ScanDone ==
    /\ s.pc = "read" /\ s.scan > Len(s.journal)
    /\ (~DetectAbandoned \/ s.latest = 0 \/ s.latest \in s.terminal \/ Late)
    /\ s' = [s EXCEPT !.observed = @ \cup s.terminal, !.pc = "run-next"]
RunNext == /\ s.pc = "run-next"
           /\ s' = [s EXCEPT !.nextRan = TRUE, !.owned = FALSE, !.pc = "done"]
Next == Runner \/ Spawn \/ Finish \/ Summary \/ CrashRunner \/ SecondPending
        \/ AcquireNext \/ RecoverGroups \/ ReadLine \/ PublishAbandoned \/ AbandonSummary
        \/ InterruptPublication \/ ScanDone \/ RunNext
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
TypeOK == /\ s.outcomes \in [1..2 -> 0..1] /\ s.observed \subseteq {1, 2}
          /\ s.scan \in 1..12
AtMostOneTerminal == \A r \in 1..2: s.outcomes[r] <= 1
Custody == s.pc = "done" =>
           \A r \in IF TwoPending THEN {1, 2} ELSE {1}:
               s.outcomes[r] = 1 /\ r \in s.observed
PidReuseSafe == ~s.wrongKill
TornJournalLive == <>s.nextRan
=============================================================================
