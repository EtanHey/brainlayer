---------------------------- MODULE BackupVerdict ----------------------------
(***************************************************************************)
(* backup-daily receipts -> backup_facts() -> evaluate() backup checks,     *)
(* plus the weekly VACUUM gate.                                             *)
(*                                                                          *)
(* Receipts are an append-only log (backup-daily.log). The environment      *)
(* appends them in any order its preconditions allow: snapshot, upload,     *)
(* local removal after a verified upload, abandonment, supersession,        *)
(* revocation and local-only re-verification. The health monitor ticks at  *)
(* any hour, can skip hours (Mac asleep, supervisor down) and can find the  *)
(* log unreadable.                                                          *)
(*                                                                          *)
(* Each pre-fix behaviour is a constant, so TLC can replay tonight's bugs.  *)
(* All-fixed values: ClockMode="authoritative", GraceMode="same_copy",     *)
(* LegacyState=FALSE, Fix1595=TRUE, Fix1596=TRUE, VacuumGate="latest",     *)
(* DeferredAdvances=FALSE.                                                  *)
(***************************************************************************)
EXTENDS Integers, Sequences, FiniteSets

CONSTANTS
    Snaps,            \* snapshot ids 1..n; a higher id is always created later
    Horizon,          \* last logical hour explored
    MaxLog,           \* bound on receipts appended
    RedSnap,          \* backup_local red age (54 h)
    RedOff,           \* backup_offsite red age (78 h)
    Grace,            \* wake_grace_s (2 h)
    Gap,              \* sleep_gap_s: a tick more than Gap after the last one is a resume
    VacAge,           \* backup_reuse_max_age_hours (6 h)
    UnknownAmber,     \* unknown_amber_ticks (3)
    Kinds,            \* receipt kinds the environment may append
    Reverify,         \* may a local-only success follow a revocation
    FilenameOnly,     \* may a revoking receipt lack snapshot_created_at
    Unreadable,       \* may a tick fail to read backup-daily.log
    ClockMode,        \* "authoritative" #1589 | "max_unless_revoked" #1583 final | "max" #1583 R1
    GraceMode,        \* "same_copy" #1589 | "episode" #1583 final | "renewable" #1583 R1
    LegacyState,      \* persisted state written before clock_id existed (#1589 B5c)
    Fix1595,          \* local_removed drops the local copy; Drive copy still counts
    Fix1596,          \* abandoned / superseded drop the local copy
    VacuumGate,       \* "latest" receipt per snapshot | "any" (scratch break)
    DeferredAdvances, \* scratch break: a deferred weekly run advances last_weekly
    Weekly,           \* explore weekly maintenance --full runs
    Monitor,          \* explore health ticks (the VACUUM gate does not read them)
    Strict            \* track the ghosts NoFalseGreenStrict needs

Checks == {"snap", "off"}           \* backup_local ("Database snapshot (verified)"), backup_offsite
Red(c) == IF c = "snap" THEN RedSnap ELSE RedOff
None == 0
NoTime == -1
States == {"none", "green", "amber", "red", "unknown"}
Rank(s) == CASE s = "green" -> 0 [] s = "unknown" -> 1 [] s = "amber" -> 2
             [] s = "red" -> 3 [] OTHER -> 0
AllKinds == {"local_ok", "upload_ok", "revoke", "removed", "abandon", "supersede"}
LocalKinds == AllKinds
OffKinds == {"upload_ok", "revoke"}

(* The log is kept as what its readers depend on. Truth: the kind of the    *)
(* last receipt touching each snapshot's local copy (lastLoc) and its Drive *)
(* copy (lastOff). Implementation: backup_facts()' running fold (acc).      *)
(* Two logs with the same summaries are one state.                          *)
VARIABLES
    now, created, file, events, lastLoc, lastOff, acc, \* the world and the log
    lastTick, resumed, lastReadable,       \* the monitor
    st, clk, dl, uk, removedFlag,          \* persisted verdict state per check
    everRead, kLoc, kOff, sLoc, sOff, prevSt, grace, graceS, \* ghosts
    lastWeekly, weeklyAt, vac               \* weekly VACUUM

world == <<now, created, file, events, lastLoc, lastOff, acc>>
monitor == <<lastTick, resumed, lastReadable, st, clk, dl, uk, removedFlag>>
ghosts == <<everRead, kLoc, kOff, sLoc, sOff, prevSt, grace, graceS>>
weekly == <<lastWeekly, weeklyAt, vac>>
vars == <<world, monitor, ghosts, weekly>>

MaxOf(S) == IF S = {} THEN None ELSE CHOOSE x \in S : \A y \in S : y <= x
Age(s) == now - created[s]

-----------------------------------------------------------------------------
(* Truth: from the last receipt that touches each copy. L = last local     *)
(* kind per snapshot, O = last Drive kind per snapshot.                     *)
LocalOK(L, s) == L[s] \in {"local_ok", "upload_ok"}
OffOK(O, s) == O[s] = "upload_ok"
Covered(L, O, c, s) == IF c = "snap" THEN LocalOK(L, s) \/ OffOK(O, s) ELSE OffOK(O, s)
TrueClock(L, O, c) == MaxOf({s \in Snaps : created[s] # NoTime /\ Covered(L, O, c, s)})

-----------------------------------------------------------------------------
(* Implementation: backup_facts() folds the receipts in order.              *)
(* rev[s] is #1583-final's shared revocation record: "none", or the         *)
(* revoking receipt's identity ("exact" creation time or filename "approx").*)
Init0 == [local |-> {}, off |-> {}, rev |-> [s \in Snaps |-> "none"]]
Fold1(a, e) ==
    LET s == e.snap IN
    CASE e.kind = "revoke" ->
            [a EXCEPT !.local = @ \ {s}, !.off = @ \ {s},
                      !.rev[s] = IF ClockMode # "max_unless_revoked" THEN "none"
                                 ELSE IF e.exact THEN "exact" ELSE "approx"]
      [] e.kind = "upload_ok" ->
            [a EXCEPT !.local = @ \cup {s}, !.off = @ \cup {s}, !.rev[s] = "none"]
      [] e.kind = "local_ok" ->
            [a EXCEPT !.local = @ \cup {s}, !.rev[s] = "none"]
      [] e.kind = "removed" ->
            IF Fix1595 THEN [a EXCEPT !.local = @ \ {s}] ELSE a
      [] e.kind \in {"abandon", "supersede"} ->
            IF Fix1596 THEN [a EXCEPT !.local = @ \ {s}] ELSE a
Reader == acc
ReaderClock(r, c) ==
    IF c = "off" THEN MaxOf(r.off)
    ELSE IF Fix1595 THEN MaxOf(r.local \cup r.off) ELSE MaxOf(r.local)

(* _is_revoked: by clock_id, or by exact creation time for legacy state.    *)
Revoked(r, p) == IF LegacyState THEN r.rev[p] = "exact" ELSE r.rev[p] # "none"

(* _verified_age's choice of clock for one check.                           *)
Clock(readable, r, c) ==
    LET pc == clk[c]
        obs == ReaderClock(r, c)
    IN IF ~readable THEN pc
       ELSE CASE ClockMode = "authoritative" -> obs
              [] ClockMode = "max" -> IF obs >= pc THEN obs ELSE pc
              [] ClockMode = "max_unless_revoked" ->
                    LET p2 == IF pc # None /\ Revoked(r, pc) THEN None ELSE pc
                    IN IF obs >= p2 THEN obs ELSE p2

RawState(readable, k, c) ==
    IF k = None THEN (IF readable THEN "red" ELSE "unknown")
    ELSE IF Age(k) > Red(c) THEN "red" ELSE "green"

-----------------------------------------------------------------------------
(* evaluate() for one backup check.                                         *)
Resumed1 == IF lastTick # NoTime /\ now - lastTick > Gap THEN now ELSE resumed
ResumedNow == IF Resumed1 # NoTime /\ now - Resumed1 > Grace THEN NoTime ELSE Resumed1

Eval(readable, r, c) ==
    LET k == Clock(readable, r, c)
        raw == RawState(readable, k, c)
        prior == st[c]
        ev == readable
        uk1 == IF raw = "unknown" THEN uk[c] + 1 ELSE 0
        s1 == IF raw = "unknown"
                THEN (IF prior = "red" THEN "red"
                      ELSE IF uk1 >= UnknownAmber THEN "amber"
                      ELSE IF prior \in {"none", "unknown"} THEN "unknown" ELSE prior)
              ELSE IF ~ev /\ prior \in {"amber", "red"} /\ Rank(raw) < Rank(prior)
                THEN prior
              ELSE raw
        same == k # None /\ k = clk[c]
        d0 == dl[c]
        \* same_copy (#1589): only this verified clock crossing red during a gap.
        d1 == IF ~same \/ raw = "green" THEN NoTime ELSE d0
        crossed == lastTick # NoTime /\ ResumedNow # NoTime
                   /\ k # None /\ lastTick <= created[k] + Red(c)
                   /\ created[k] + Red(c) < ResumedNow
        d2 == IF s1 = "red" /\ prior # "red" /\ same /\ d1 = NoTime /\ crossed
                THEN ResumedNow + Grace ELSE d1
        sameCopy == IF s1 = "red" /\ prior # "red" /\ same /\ d2 # NoTime /\ now <= d2
                      THEN <<"amber", d2>> ELSE <<s1, d2>>
        \* episode (#1583 final): one deadline per episode, any newly red check.
        e1 == IF raw = "green" THEN NoTime ELSE d0
        e2 == IF s1 = "red" /\ prior # "red" /\ e1 = NoTime /\ ResumedNow # NoTime
                THEN ResumedNow + Grace ELSE e1
        episode == IF s1 = "red" /\ prior # "red" /\ e2 # NoTime /\ now <= e2
                     THEN <<"amber", e2>> ELSE <<s1, IF raw = "green" THEN NoTime ELSE e2>>
        \* renewable (#1583 R1): every resume grants a fresh window.
        renew == IF s1 = "red" /\ prior # "red" /\ ResumedNow # NoTime
                   THEN <<"amber", NoTime>> ELSE <<s1, NoTime>>
        out == CASE GraceMode = "same_copy" -> sameCopy
                 [] GraceMode = "episode" -> episode
                 [] GraceMode = "renewable" -> renew
    IN [state |-> out[1], deadline |-> out[2], unknown |-> uk1,
        clock |-> IF k # None THEN k ELSE IF raw = "unknown" THEN clk[c] ELSE None]

-----------------------------------------------------------------------------
(* The world.                                                               *)
AddReceipt(e) ==
    /\ events < MaxLog
    /\ events' = events + 1
    /\ lastLoc' = [lastLoc EXCEPT ![e.snap] = e.kind]
    /\ lastOff' = IF e.kind \in OffKinds THEN [lastOff EXCEPT ![e.snap] = e.kind] ELSE lastOff
    /\ acc' = Fold1(acc, e)
Taken == {s \in Snaps : created[s] # NoTime}

TakeSnapshot(s) ==
    /\ "local_ok" \in Kinds
    /\ created[s] = NoTime
    /\ \A t \in Snaps : t < s => created[t] # NoTime /\ created[t] < now
    /\ AddReceipt([kind |-> "local_ok", snap |-> s, exact |-> TRUE])
    /\ created' = [created EXCEPT ![s] = now]
    /\ file' = [file EXCEPT ![s] = TRUE]
    /\ UNCHANGED now

Upload(s) ==
    /\ "upload_ok" \in Kinds
    /\ s \in Taken /\ file[s] /\ LocalOK(lastLoc, s) /\ ~OffOK(lastOff, s)
    /\ AddReceipt([kind |-> "upload_ok", snap |-> s, exact |-> TRUE])
    /\ UNCHANGED <<now, created, file>>

RemoveAfterUpload(s) ==         \* backup_daily deletes the archive once Drive verifies it
    /\ "removed" \in Kinds
    /\ s \in Taken /\ file[s] /\ OffOK(lastOff, s)
    /\ AddReceipt([kind |-> "removed", snap |-> s, exact |-> TRUE])
    /\ file' = [file EXCEPT ![s] = FALSE]
    /\ UNCHANGED <<now, created>>

Abandon(s) ==                   \* _local_upload_receipt: local file missing or changed
    /\ "abandon" \in Kinds
    /\ s \in Taken /\ file[s] /\ ~OffOK(lastOff, s)
    /\ AddReceipt([kind |-> "abandon", snap |-> s, exact |-> TRUE])
    /\ file' = [file EXCEPT ![s] = FALSE]
    /\ UNCHANGED <<now, created>>

Supersede(s) ==                 \* _reclaim_superseded_local: unlink a never-uploaded stall,
    /\ "supersede" \in Kinds     \* run only after a NEWER snapshot's verified upload
    /\ s \in Taken /\ file[s] /\ ~OffOK(lastOff, s)
    /\ \E t \in Taken : t > s /\ OffOK(lastOff, t)
    /\ AddReceipt([kind |-> "supersede", snap |-> s, exact |-> TRUE])
    /\ file' = [file EXCEPT ![s] = FALSE]
    /\ UNCHANGED <<now, created>>

Revoke(s) ==                    \* verification_error or drive_md5_match = false
    /\ "revoke" \in Kinds
    /\ s \in Taken /\ (LocalOK(lastLoc, s) \/ OffOK(lastOff, s))
    /\ \E exact \in (IF FilenameOnly THEN BOOLEAN ELSE {TRUE}) :
          AddReceipt([kind |-> "revoke", snap |-> s, exact |-> exact])
    /\ UNCHANGED <<now, created, file>>

LocalReverify(s) ==             \* a local-only success after a revocation
    /\ Reverify
    /\ s \in Taken /\ file[s] /\ lastLoc[s] = "revoke"
    /\ AddReceipt([kind |-> "local_ok", snap |-> s, exact |-> TRUE])
    /\ UNCHANGED <<now, created, file>>

World == /\ \E s \in Snaps : TakeSnapshot(s) \/ Upload(s) \/ RemoveAfterUpload(s)
                             \/ Abandon(s) \/ Supersede(s) \/ Revoke(s) \/ LocalReverify(s)
         /\ UNCHANGED <<monitor, ghosts, weekly>>

Advance == /\ now < Horizon
           /\ now' = now + 1
           /\ UNCHANGED <<created, file, events, lastLoc, lastOff, acc, monitor, ghosts, weekly>>

-----------------------------------------------------------------------------
(* One health tick.                                                         *)
GraceGhost(L, O, c, g) ==
    LET k == TrueClock(L, O, c) IN
    IF k # None /\ Age(k) > Red(c)
    THEN IF g.clk = k THEN g
         ELSE [clk |-> k, at |-> now,
               ok |-> lastTick # NoTime /\ created[k] + Red(c) >= lastTick]
    ELSE [clk |-> None, at |-> NoTime, ok |-> FALSE]

Tick(readable) ==
    /\ Monitor
    /\ readable \/ Unreadable
    /\ lastTick < now
    /\ LET r == Reader
           e == [c \in Checks |-> Eval(readable, r, c)]
           kl == IF readable THEN lastLoc ELSE kLoc
           ko == IF readable THEN lastOff ELSE kOff
       IN /\ st' = [c \in Checks |-> e[c].state]
          /\ clk' = [c \in Checks |-> e[c].clock]
          /\ dl' = [c \in Checks |-> e[c].deadline]
          /\ uk' = [c \in Checks |-> IF e[c].unknown > UnknownAmber THEN UnknownAmber
                                     ELSE e[c].unknown]
          /\ removedFlag' = (readable /\ Fix1595 /\ ReaderClock(r, "snap") # None
                             /\ ReaderClock(r, "snap") \notin r.local)
          /\ kLoc' = kl
          /\ kOff' = ko
          /\ grace' = [c \in Checks |->
                IF everRead \/ readable THEN GraceGhost(kl, ko, c, grace[c]) ELSE grace[c]]
          /\ graceS' = IF Strict THEN [c \in Checks |-> GraceGhost(lastLoc, lastOff, c, graceS[c])]
                       ELSE graceS
    /\ everRead' = (everRead \/ readable)
    /\ sLoc' = IF Strict THEN lastLoc ELSE sLoc
    /\ sOff' = IF Strict THEN lastOff ELSE sOff
    /\ prevSt' = st
    /\ lastTick' = now
    /\ resumed' = ResumedNow
    /\ lastReadable' = readable
    /\ UNCHANGED <<world, weekly>>

-----------------------------------------------------------------------------
(* Weekly maintenance --full: freelist gate, then the fresh-snapshot gate. *)
GateFresh(s) == s \in Taken /\ file[s] /\ Age(s) <= VacAge /\ Age(s) >= 0
                /\ (VacuumGate = "latest" => lastLoc[s] \in {"local_ok", "upload_ok"})
TrulyFresh == \E s \in Taken : file[s] /\ LocalOK(lastLoc, s) /\ Age(s) <= VacAge

WeeklyRun ==
    /\ Weekly
    /\ weeklyAt < now
    /\ weeklyAt' = now
    /\ \E high \in BOOLEAN :
         IF ~high
         THEN /\ vac' = [kind |-> "not_needed", fresh |-> TrulyFresh, advanced |-> TRUE]
              /\ lastWeekly' = now
         ELSE IF \E s \in Snaps : GateFresh(s)
         THEN /\ vac' = [kind |-> "vacuumed", fresh |-> TrulyFresh, advanced |-> TRUE]
              /\ lastWeekly' = now
         ELSE /\ vac' = [kind |-> "deferred", fresh |-> TrulyFresh, advanced |-> DeferredAdvances]
              /\ lastWeekly' = IF DeferredAdvances THEN now ELSE lastWeekly
    /\ UNCHANGED <<world, monitor, ghosts>>

-----------------------------------------------------------------------------
Init ==
    /\ now = 0
    /\ created = [s \in Snaps |-> NoTime]
    /\ file = [s \in Snaps |-> FALSE]
    /\ events = 0
    /\ lastLoc = [s \in Snaps |-> "none"]
    /\ lastOff = [s \in Snaps |-> "none"]
    /\ acc = Init0
    /\ lastTick = NoTime
    /\ resumed = NoTime
    /\ lastReadable = FALSE
    /\ st = [c \in Checks |-> "none"]
    /\ clk = [c \in Checks |-> None]
    /\ dl = [c \in Checks |-> NoTime]
    /\ uk = [c \in Checks |-> 0]
    /\ removedFlag = FALSE
    /\ everRead = FALSE
    /\ kLoc = [s \in Snaps |-> "none"]
    /\ kOff = [s \in Snaps |-> "none"]
    /\ sLoc = [s \in Snaps |-> "none"]
    /\ sOff = [s \in Snaps |-> "none"]
    /\ prevSt = [c \in Checks |-> "none"]
    /\ grace = [c \in Checks |-> [clk |-> None, at |-> NoTime, ok |-> FALSE]]
    /\ graceS = [c \in Checks |-> [clk |-> None, at |-> NoTime, ok |-> FALSE]]
    /\ lastWeekly = NoTime
    /\ weeklyAt = NoTime
    /\ vac = [kind |-> "none", fresh |-> FALSE, advanced |-> FALSE]

Next == World \/ Advance \/ Tick(TRUE) \/ Tick(FALSE) \/ WeeklyRun
Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
(* Properties.                                                              *)
TypeOK ==
    /\ now \in 0..Horizon
    /\ \A c \in Checks : st[c] \in States /\ clk[c] \in Snaps \cup {None}
    /\ events <= MaxLog

Fresh(L, O, c, g) ==
    LET k == TrueClock(L, O, c) IN
    \/ k # None /\ lastTick - created[k] <= Red(c)
    \/ k # None /\ g.ok /\ g.clk = k /\ lastTick <= g.at + Grace

(* A non-red backup check means a verified copy of its kind still exists    *)
(* within the red age, as far as the last readable receipts can tell, or   *)
(* the Mac just woke and that same real copy aged out during the gap        *)
(* (one bounded window that a later gap cannot renew).                      *)
NoFalseGreen ==
    everRead => \A c \in Checks : st[c] # "red" => Fresh(kLoc, kOff, c, grace[c])

(* The same claim against the whole log, including receipts appended while  *)
(* the log was unreadable. Expected to fail: a design limit, not a bug.     *)
NoFalseGreenStrict ==
    everRead => \A c \in Checks : st[c] # "red" => Fresh(sLoc, sOff, c, graceS[c])

(* A red backup check leaves red only on a readable tick that shows a new   *)
(* verified copy within the red age.                                        *)
RedHoldsUntilEvidence ==
    \A c \in Checks : prevSt[c] = "red" /\ st[c] # "red" =>
        lastReadable /\ TrueClock(kLoc, kOff, c) # None
        /\ lastTick - created[TrueClock(kLoc, kOff, c)] <= Red(c)

(* The label never claims a local copy that is gone, and the "local copy    *)
(* removed after Drive verification" note is true when shown.              *)
ClaimsLocal == ~Fix1595             \* pre-#1595 label: "Local database backup"
TruthfulSnapshotLabel ==
    st["snap"] # "red" /\ clk["snap"] # None /\ lastReadable =>
        /\ ClaimsLocal => LocalOK(kLoc, clk["snap"])
        /\ removedFlag => ~LocalOK(kLoc, clk["snap"]) /\ OffOK(kOff, clk["snap"])

(* VACUUM only with a verified local snapshot at most VacAge old, and a    *)
(* deferred weekly run keeps the weekly pass due.                           *)
VacuumNeedsFreshSnapshot ==
    /\ vac.kind = "vacuumed" => vac.fresh
    /\ vac.kind = "deferred" => ~vac.advanced
=============================================================================
