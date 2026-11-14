--------------------------- MODULE IngestCoverage ---------------------------
(* Does the BrainBar "Transcript ingest" check leave green when a wired agent
   writes and ingest makes no progress?

   One step is one health tick (300 s in production) or one tick spent asleep.
   The status side follows the real code, not an idealised one:
   - watcher.py poll_once writes watcher-health.json (updated_at, poll_count) in
     its finally block on every completed poll, whatever the poll ingested;
   - health_facts.watcher_health_fact reads updated_at; a missing file is None and
     an unreadable one is UNOBSERVED, and both reach the verdict as None;
   - health_verdict._raw_checks/_graced_age/evaluate turn that one clock into the
     ingest state (age > amber / > red, unknown ticks, hysteresis, suspension
     credit on a confirmed clock, one wake-grace window per clock otherwise).
   - the same read also returns the watcher's stall codes (health_facts.
     watcher_stall_codes): file_ingestion_failure at once; coverage_drop and
     root_missing once their episode has lasted alert_after_s (one tick); the
     verdict turns them amber, and red on the AlertRed-th new snapshot in a row.
   Rule = "heartbeat" is the code before that fix (#1780's finding): the alert
   codes never reach the verdict. *)
EXTENDS Integers, Sequences

CONSTANTS AmberT, RedT, UnknownAmber, Recovery, SleepGap, WakeGrace, AlertRed,
          Cap, N, Rule, Mut, Clock, Faults, Script

VARIABLES t, fault, asleep, slept, hbA, hbW, pA, pW, resumed, grace,
          v, unk, good, alertT, fileAlert, backlog, dropped, stall, blindRun, faultEver, wokeInStall, obs,
          covPrev, rootPrev, covLinger, fileNew, sBlind, partialRun

NoScript == <<>>   \* cfg files cannot spell <<>>: they say Script <- NoScript

vars == <<t, fault, asleep, slept, hbA, hbW, pA, pW, resumed, grace,
          v, unk, good, alertT, fileAlert, backlog, dropped, stall, blindRun, faultEver, wokeInStall, obs,
          covPrev, rootPrev, covLinger, fileNew, sBlind, partialRun>>

(* The eight shapes, with shape 8 split in two. "stuck" is a live process whose
   poll loop no longer completes, so poll_count and updated_at freeze together
   (same payload). "rewrite" is a rewrite the #732 resume fingerprint sees (head
   digest or the bytes before the offset changed): the watcher re-reads from 0
   and every rewritten record is re-ingested, which the replay checks record by
   record. "rewrite_lossy" is one it cannot see: a same-length edit of a record
   past the first 4 KiB and before the last 256 bytes. The watcher resumes at its
   offset and the edited record is never ingested.
   "drift" (records in a shape no parser takes) publishes no watcher signal;
   finding-real-drift shows it stays green. *)
Shapes == {"dead", "stuck", "lag", "coverage", "dbfail",
           "blind_stale", "blind_unreadable", "rootgone", "rewrite", "rewrite_lossy", "drift", "partial"}
NoProgress == {"dead", "stuck", "lag", "coverage", "dbfail", "rootgone", "rewrite_lossy", "drift"}
\* Ground truth: records these faults consume and never index (independent of Mut).
Dropping == {"coverage", "rewrite_lossy", "drift"}
\* Mutation rewrite_progress: count every rewrite as progress, lossy or not.
ProgressLabel(f) == IF Mut = "rewrite_progress" THEN f \notin NoProgress \ {"rewrite_lossy"}
                    ELSE f \notin NoProgress
\* partial: the heartbeat reads but the stall fields are absent or malformed (codes UNOBSERVED)
Blind == {"blind_stale", "blind_unreadable", "partial"}

Rank(s) == CASE s = "green" -> 0 [] s = "unknown" -> 1 [] s = "amber" -> 2 [] s = "red" -> 3
NonGreen(s) == s \in {"amber", "red"}   \* the dot ignores "unknown" (evaluate: worst)
AgeState(a) == IF a > RedT THEN "red" ELSE IF a > AmberT THEN "amber" ELSE "green"
Inc(x) == IF x < 0 THEN x ELSE IF x >= Cap THEN Cap ELSE x + 1
Min(a, b) == IF a < b THEN a ELSE b
Wall(x) == IF Clock = "confirmed" THEN 0 ELSE x   \* wall ages matter only unconfirmed

InitSteady == /\ hbA = 0 /\ hbW = 0 /\ pA = 0 /\ pW = 0 /\ v = "green"
InitFresh  == /\ hbA = -1 /\ hbW = -1 /\ pA = -1 /\ pW = -1 /\ v = "unknown"
Init == /\ t = 0 /\ fault = "none" /\ asleep = FALSE /\ slept = 0
        /\ resumed = -1 /\ grace = -1 /\ unk = 0 /\ good = 0 /\ alertT = 0 /\ fileAlert = FALSE
        /\ backlog = FALSE /\ dropped = FALSE /\ stall = 0 /\ blindRun = 0 /\ faultEver = FALSE /\ wokeInStall = FALSE
        /\ obs = "fresh" /\ covPrev = FALSE /\ rootPrev = FALSE /\ covLinger = FALSE
        /\ fileNew = FALSE /\ sBlind = 0 /\ partialRun = 0
        /\ IF Script = <<>> THEN InitSteady \/ InitFresh ELSE InitSteady

(* Lid closed: no watcher poll, no agent writes, no health tick. Wall time runs. *)
Sleep == /\ asleep' = TRUE
         /\ slept' = Min(slept + 1, Cap)
         /\ hbW' = Wall(Inc(hbW)) /\ pW' = Wall(Inc(pW))
         /\ resumed' = IF resumed >= 0 THEN Min(resumed + 1, Cap) ELSE -1
         /\ grace' = IF grace >= 0 THEN Min(grace + 1, WakeGrace + 1) ELSE -1
         /\ obs' = "asleep"
         /\ t' = IF Script = <<>> THEN t ELSE t + 1
         /\ UNCHANGED <<fault, hbA, pA, v, unk, good, alertT, fileAlert, backlog, dropped, stall, blindRun,
                        faultEver, wokeInStall, covPrev, rootPrev, covLinger, fileNew, sBlind, partialRun>>

(* One awake health tick. f: the fault this tick; w: the agent appends;
   o: on the first tick after a wake, whether the monitor reads before the
   watcher's first poll ("monitorFirst") or after it. *)
Tick(f, w, o) ==
  LET woke   == asleep
      polls  == f \notin {"dead", "stuck"}
      writes == polls /\ f # "blind_stale"            \* tmp write + replace fails: stale
      early  == writes /\ ~(woke /\ o = "monitorFirst")
      rA == IF early THEN 0 ELSE Inc(hbA)              \* heartbeat age at the read
      rW == IF early THEN 0 ELSE Wall(Inc(hbW))
      seen == f # "blind_unreadable" /\ rA >= 0        \* watcher_health_fact returned a time
      reset == Mut = "sleep_resets" /\ woke            \* mutation: a wake forgives the clock
      \* clock = max(watcher, persisted): a newer heartbeat if seen, else the persisted one
      cA == IF reset THEN 0 ELSE IF seen THEN rA ELSE Inc(pA)
      cW == IF reset THEN 0 ELSE IF seen THEN rW ELSE Wall(Inc(pW))
      known == cA >= 0
      newClock == seen /\ (pA < 0 \/ rA < Inc(pA))
      \* unconfirmed clock: resumed_at after a gap > sleep_gap_s, expires after wake grace
      res0 == IF Clock = "unconfirmed" /\ woke /\ slept + 1 > SleepGap THEN 0
              ELSE IF resumed >= 0 THEN Min(resumed + 1, Cap) ELSE -1
      res  == IF res0 > WakeGrace THEN -1 ELSE res0
      \* one window per clock value; an expired window is kept so it is never renewed
      g0 == IF Clock = "confirmed" \/ newClock \/ ~known THEN -1
            ELSE IF grace >= 0 THEN Min(grace + 1, WakeGrace + 1) ELSE -1
      g  == IF g0 >= 0 THEN g0
            ELSE IF Clock = "unconfirmed" /\ known /\ res >= 0 /\ cW > res THEN res
            ELSE -1
      trueAge == IF Clock = "confirmed" THEN cA ELSE cW   \* confirmed: suspension credited
      age == IF g >= 0 /\ g <= WakeGrace THEN Min(trueAge, g) ELSE trueAge
      graced == known /\ Rank(AgeState(age)) < Rank(AgeState(trueAge))
      backlogNow == backlog \/ w \/ f = "rewrite_lossy"   \* the edit is itself agent content
      progress == ProgressLabel(f)
      \* What this poll publishes. An unreadable transcript is a file_ingestion_failure at
      \* once. Retained failed flushes (dbfail) or writes that never reach the probed DB
      \* (coverage) are coverage_drop, and a vanished root is root_missing, once the
      \* condition held at the previous poll too (alert_after_s = one tick of awake time).
      \* A coverage window stays open while it writes nothing, so the alert is still in
      \* the first poll after the fault (covLinger).
      covC == Mut # "no_coverage" /\ f \in {"coverage", "dbfail"} /\ backlogNow
      rootC == f = "rootgone"
      alertWritten == \/ f = "lag" /\ backlogNow
                      \/ covC /\ covPrev
                      \/ rootC /\ rootPrev
                      \/ covLinger /\ f # "coverage"
      \* the file keeps the last payload written: a dead watcher leaves its last alert
      \* An alert check counts only on a snapshot newer than the one last counted (a re-read
      \* frozen file is not a new check); with no count running, any read counts.
      \* mutation partial_clear: a payload without stall fields reads as "no stall"
      codesSeen == seen /\ (f # "partial" \/ Mut = "partial_clear")
      inFile == IF f = "partial" THEN FALSE ELSE IF early THEN alertWritten ELSE fileAlert
      advanced == early \/ fileNew \/ alertT = 0
      aT == CASE Rule # "real" -> 0
              [] ~codesSeen -> alertT
              [] advanced -> IF inFile THEN Min(alertT + 1, Cap) ELSE 0
              [] OTHER -> alertT
      \* a missing file is "observed, nothing there" (codes ()), not a blind stall reading
      sB == CASE Rule # "real" -> 0
              [] codesSeen -> 0
              [] seen \/ f = "blind_unreadable" -> Min(sBlind + 1, Cap)
              [] OTHER -> sBlind
      alertState == CASE aT >= AlertRed -> "red"
                      [] aT > 0 -> "amber"
                      [] sB >= UnknownAmber -> "amber"
                      [] OTHER -> "green"
      raw0 == CASE Mut = "stale_green" /\ seen -> "green"           \* age ignored
                [] Mut = "trust_alive" /\ f # "dead" -> "green"     \* frozen poll_count ignored
                [] OTHER -> IF known THEN AgeState(age) ELSE "unknown"
      raw == IF raw0 # "unknown" /\ Rank(alertState) > Rank(raw0) THEN alertState ELSE raw0
      evidence == CASE Mut = "trust_alive" /\ f # "dead" -> TRUE
                    [] Mut = "stale_green" /\ seen -> TRUE
                    [] OTHER -> seen /\ ~graced /\ (Rule # "real" \/ codesSeen)
      \* evaluate(): unknown ticks, holds without evidence, hysteresis
      unk1 == IF raw = "unknown" THEN Min(unk + 1, Cap) ELSE 0   \* every value >= UnknownAmber acts alike
      base == IF v \in {"green", "amber", "red"} THEN v ELSE "unknown"
      g1 == IF raw # "green" \/ ~evidence THEN 0 ELSE good
      vNext == CASE raw = "unknown" ->
                      IF base = "red" THEN "red" ELSE IF unk1 >= UnknownAmber THEN "amber" ELSE base
                 [] ~evidence /\ NonGreen(v) /\ Rank(raw) < Rank(v) -> v
                 [] v = "red" /\ raw = "amber" -> "red"
                 [] raw = "green" /\ NonGreen(v) -> IF g1 + 1 < Recovery THEN v ELSE "green"
                 [] OTHER -> raw
      goodNext == CASE raw = "unknown" -> 0
                    [] ~evidence /\ NonGreen(v) /\ Rank(raw) < Rank(v) -> 0
                    [] v = "red" /\ raw = "amber" -> 0
                    [] raw = "green" /\ NonGreen(v) -> IF g1 + 1 < Recovery THEN g1 + 1 ELSE 0
                    [] OTHER -> 0
      stalledNow == backlogNow /\ ~progress
  IN /\ t' = IF Script = <<>> THEN t ELSE t + 1
     /\ fault' = f /\ asleep' = FALSE /\ slept' = 0
     /\ hbA' = IF writes THEN 0 ELSE rA
     /\ hbW' = IF writes THEN 0 ELSE rW
     /\ pA' = IF known THEN cA ELSE pA
     /\ pW' = IF known THEN cW ELSE pW
     /\ resumed' = res
     /\ grace' = IF known THEN g ELSE -1
     /\ v' = vNext /\ unk' = unk1 /\ good' = goodNext
     /\ alertT' = aT
     /\ fileAlert' = IF writes THEN alertWritten ELSE fileAlert
     /\ backlog' = (backlogNow /\ ~progress)
     /\ dropped' = (backlogNow /\ f \in Dropping)
     /\ stall' = IF stalledNow THEN Min(stall + 1, Cap) ELSE 0
     /\ blindRun' = IF f \in Blind /\ ~newClock THEN Min(blindRun + 1, Cap) ELSE 0
     \* a fresh heartbeat whose stall fields cannot be read (no file at all is not partial)
     /\ partialRun' = IF f = "partial" /\ seen THEN Min(partialRun + 1, Cap) ELSE 0
     /\ faultEver' = (faultEver \/ f # "none")
     /\ wokeInStall' = (stalledNow /\ (wokeInStall \/ (woke /\ stall > 0)))
     /\ covPrev' = IF polls THEN covC ELSE covPrev
     /\ rootPrev' = IF polls THEN rootC ELSE rootPrev
     /\ covLinger' = IF polls THEN (f = "coverage" /\ covC /\ covPrev) ELSE covLinger
     /\ fileNew' = IF codesSeen THEN writes /\ ~early ELSE fileNew \/ writes
     /\ sBlind' = sB
     /\ obs' = IF f = "blind_unreadable" THEN "unreadable"
               ELSE IF rA < 0 THEN "missing" ELSE IF early THEN "fresh" ELSE "old"

Explore == \/ \E f \in Faults, w \in BOOLEAN, o \in {"watcherFirst", "monitorFirst"}: Tick(f, w, o)
           \/ Sleep
Scripted == /\ t < Len(Script)
            /\ LET e == Script[t + 1] IN
                 IF e.s = "sleep" THEN Sleep ELSE Tick(e.f, e.w, e.o)
Next == IF Script = <<>> THEN Explore ELSE Scripted
Spec == Init /\ [][Next]_vars

TypeOK == /\ partialRun \in 0..Cap /\ fault \in Faults \cup {"none"} /\ v \in {"green", "amber", "red", "unknown"}
          /\ stall \in 0..Cap /\ blindRun \in 0..Cap /\ hbA \in -1..Cap /\ pA \in -1..Cap

(* A wired agent's content waits with no ingest progress for more than N awake
   ticks, or the monitor has been blind for more than N ticks => the check is
   amber or red. stall resets only on progress (blindRun only when the monitor
   reads a heartbeat newer than the one it had; partialRun only when the stall
   fields read again), so this also says it stays non-green until that happens. *)
IngestFailureLeavesGreen == (stall > N \/ blindRun > N \/ partialRun > N) => NonGreen(v)
(* The stall clock never ignores agent content that is waiting or was dropped
   this tick: the bookkeeping IngestFailureLeavesGreen rests on. *)
StallClockIsTruthful == (backlog \/ dropped) => stall > 0
(* No fault in the whole behaviour: lid closes alone never leave green. *)
SleepIsNotFailure == ~faultEver => ~NonGreen(v)
(* A stall that was already running when the lid closed still counts after wake. *)
StallAcrossWakeStillCounts == (wokeInStall /\ stall > N) => NonGreen(v)
(* Replay helper: violated exactly when the script has run, so TLC prints it. *)
ScriptRunning == t < Len(Script)
=============================================================================
