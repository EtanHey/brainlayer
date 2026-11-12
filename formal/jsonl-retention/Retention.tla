--------------------------- MODULE Retention ---------------------------
EXTENDS Naturals, FiniteSets
CONSTANT Bug
VARIABLES local, uploaded, verified, age, deleted, deleteAge
vars == <<local, uploaded, verified, age, deleted, deleteAge>>
Archives == {1, 2}
Newest == 2
Init == /\ local = Archives /\ uploaded = {} /\ verified = {}
        /\ age = [a \in Archives |-> 0] /\ deleted = {}
        /\ deleteAge = [a \in Archives |-> 0]
Upload(a) == /\ a \in local /\ uploaded' = uploaded \cup {a}
             /\ UNCHANGED <<local, verified, age, deleted, deleteAge>>
Verify(a) == /\ a \in uploaded /\ verified' = verified \cup {a}
             /\ UNCHANGED <<local, uploaded, age, deleted, deleteAge>>
Delete(a) == /\ a \in local /\ a # Newest
             /\ (Bug = "verified" \/ a \in verified)
             /\ (Bug = "age" \/ age[a] >= 14)
             /\ local' = local \ {a} /\ deleted' = deleted \cup {a}
             /\ deleteAge' = [deleteAge EXCEPT ![a] = age[a]]
             /\ UNCHANGED <<uploaded, verified, age>>
Tick == /\ age' = [a \in Archives |-> IF age[a] < 14 THEN age[a]+1 ELSE 14]
        /\ UNCHANGED <<local, uploaded, verified, deleted, deleteAge>>
Crash == UNCHANGED vars
Next == (\E a \in Archives: Upload(a) \/ Verify(a) \/ Delete(a)) \/ Tick \/ Crash
Spec == Init /\ [][Next]_vars
NeverDeleteWithoutVerifiedCopy == deleted \subseteq verified
NeverDeleteYoungerThan14Days == \A a \in deleted: deleteAge[a] >= 14
NeverDeleteNewest == Newest \in local
AlwaysAtLeastOneCopy == \A a \in Archives: a \in local \/ a \in verified
=============================================================================
