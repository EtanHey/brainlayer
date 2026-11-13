--------------------------- MODULE Retention ---------------------------
EXTENDS Naturals, FiniteSets, TLC
CONSTANT Bug
VARIABLES made, local, drive, receipts, saved, phase, now, born, verifiedAt, checkedAt,
          badLive, badContent, badAge, badNewest, badRestore
vars == <<made, local, drive, receipts, saved, phase, now, born, verifiedAt, checkedAt,
          badLive, badContent, badAge, badNewest, badRestore>>
A == {1,2}
Init == /\ made = {} /\ local = {} /\ drive = [a \in A |-> "absent"]
        /\ receipts = {} /\ saved = {} /\ phase = "idle" /\ now = 0
        /\ born = [a \in A |-> 0] /\ verifiedAt = [a \in A |-> 0] /\ checkedAt = [a \in A |-> 0]
        /\ badLive = FALSE /\ badContent = FALSE /\ badAge = FALSE /\ badNewest = FALSE /\ badRestore = FALSE
Newest == IF 2 \in made THEN 2 ELSE 1
New(a) == /\ a \notin made /\ (a = 1 \/ 1 \in made)
          /\ made' = made \cup {a} /\ local' = local \cup {a}
          /\ born' = [born EXCEPT ![a] = now]
          /\ UNCHANGED <<checkedAt, drive, receipts, saved, phase, now, verifiedAt,
                          badLive, badContent, badAge, badNewest, badRestore>>
Upload(a, bytes) == /\ a \in local /\ drive[a] # "good"
                    /\ drive' = [drive EXCEPT ![a] = bytes]
                    /\ UNCHANGED <<checkedAt, made, local, receipts, saved, phase, now, born, verifiedAt,
                                    badLive, badContent, badAge, badNewest, badRestore>>
Verify(a) == /\ phase = "idle" /\ a \in local /\ a \notin receipts
             /\ (drive[a] = "good" \/ (Bug = "md5" /\ drive[a] = "different"))
             /\ receipts' = receipts \cup {a}
             /\ verifiedAt' = [verifiedAt EXCEPT ![a] = now]
             /\ checkedAt' = [checkedAt EXCEPT ![a] = now]
             /\ UNCHANGED <<made, local, drive, saved, phase, now, born,
                             badLive, badContent, badAge, badNewest, badRestore>>
\* A later run backfills a missing receipt from the live listing (name, size, md5).
\* checkedAt is when the copy was really checked; verifiedAt is what the receipt records.
Reconcile(a) == /\ phase = "idle" /\ a \in local /\ a \notin receipts /\ drive[a] = "good"
                /\ receipts' = receipts \cup {a}
                /\ verifiedAt' = [verifiedAt EXCEPT ![a] = IF Bug = "backdate" THEN 0 ELSE now]
                /\ checkedAt' = [checkedAt EXCEPT ![a] = now]
                /\ UNCHANGED <<made, local, drive, saved, phase, now, born,
                                badLive, badContent, badAge, badNewest, badRestore>>
Delete(a) == /\ a \in local /\ a \in receipts /\ phase = "idle"
             /\ (Bug = "newest" \/ a # Newest)
             /\ (Bug = "live" \/ drive[a] = "good" \/ (Bug = "md5" /\ drive[a] = "different"))
             /\ (Bug = "age" \/ (now-born[a] >= 14 /\ now-verifiedAt[a] >= 14))
             /\ local' = local \ {a}
             /\ badLive' = (badLive \/ drive[a] = "absent")
             /\ badContent' = (badContent \/ drive[a] = "different")
             /\ badAge' = (badAge \/ now-born[a] < 14 \/ now-checkedAt[a] < 14)
             /\ badNewest' = (badNewest \/ a = Newest)
             /\ UNCHANGED <<checkedAt, made, drive, receipts, saved, phase, now, born, verifiedAt, badRestore>>
Revoke == /\ phase = "idle" /\ receipts # {}
          /\ saved' = receipts /\ receipts' = {} /\ phase' = "revoked"
          /\ UNCHANGED <<checkedAt, made, local, drive, now, born, verifiedAt,
                          badLive, badContent, badAge, badNewest, badRestore>>
Prune == /\ phase = "revoked" /\ phase' = "pruned"
         /\ drive' = [a \in A |-> IF now-born[a] >= 30 THEN "absent" ELSE drive[a]]
         /\ UNCHANGED <<checkedAt, made, local, receipts, saved, now, born, verifiedAt,
                         badLive, badContent, badAge, badNewest, badRestore>>
Restore == /\ phase = "pruned" /\ phase' = "idle"
           /\ receipts' = {a \in saved: Bug = "restore" \/ drive[a] # "absent"}
           /\ badRestore' = (badRestore \/ (\E a \in receipts': drive[a] = "absent"))
           /\ saved' = {}
           /\ UNCHANGED <<checkedAt, made, local, drive, now, born, verifiedAt,
                           badLive, badContent, badAge, badNewest>>
Crash == /\ phase # "idle" /\ phase' = "idle" /\ saved' = {}
         /\ UNCHANGED <<checkedAt, made, local, drive, receipts, now, born, verifiedAt,
                         badLive, badContent, badAge, badNewest, badRestore>>
RemoteDelete(a) == /\ a \in made /\ drive[a] # "absent"
                   /\ (a \in local \/ now-born[a] >= 30 \/ Bug = "external")
                   /\ drive' = [drive EXCEPT ![a] = "absent"]
                   /\ UNCHANGED <<checkedAt, made, local, receipts, saved, phase, now, born, verifiedAt,
                                   badLive, badContent, badAge, badNewest, badRestore>>
Clock(to) == /\ to > now /\ now' = to
             /\ UNCHANGED <<checkedAt, made, local, drive, receipts, saved, phase, born, verifiedAt,
                             badLive, badContent, badAge, badNewest, badRestore>>
RemoteReplace(a) == /\ a \in local /\ drive[a] = "good"
                    /\ drive' = [drive EXCEPT ![a] = "different"]
                    /\ UNCHANGED <<checkedAt, made, local, receipts, saved, phase, now, born, verifiedAt,
                                    badLive, badContent, badAge, badNewest, badRestore>>
Tick == Clock(IF now < 14 THEN 14 ELSE IF now < 28 THEN 28 ELSE 31)
ClockJump == Clock(31)
Next == (\E a \in A: New(a) \/ Verify(a) \/ Reconcile(a) \/ Delete(a) \/ RemoteDelete(a) \/ RemoteReplace(a)
                      \/ (\E bytes \in {"good", "different"}: Upload(a,bytes)))
        \/ Revoke \/ Prune \/ Restore \/ Crash \/ (now < 31 /\ (Tick \/ ClockJump))
Spec == Init /\ [][Next]_vars
NeverDeleteWithoutVerifiedCopy == ~badLive /\ ~badContent
NeverDeleteYoungerThan14Days == ~badAge
NeverDeleteNewest == ~badNewest
RestoreOnlySurvivors == ~badRestore
AlwaysAtLeastOneCopy == \A a \in made: now-born[a] >= 30 \/ a \in local \/ drive[a] = "good"
=============================================================================
