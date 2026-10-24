---------------- MODULE InterruptedTailRegression ----------------
EXTENDS BackupProgress
VARIABLE pc
ReplayInit == Init /\ pc = 0
ReplayNext ==
    \/ /\ pc = 0 /\ Spawn(1, TRUE) /\ pc' = 1
    \/ /\ pc = 1 /\ Admit(1) /\ pc' = 2
    \/ /\ pc = 2 /\ Finish(1, "ok") /\ pc' = 3
    \/ /\ pc = 3 /\ Spawn(2, TRUE) /\ pc' = 4
    \/ /\ pc = 4 /\ Admit(2) /\ pc' = 5
    \/ /\ pc = 5 /\ Snapshot(2) /\ pc' = 6
    \/ /\ pc = 6 /\ Advance(2, "pending", "in_progress") /\ pc' = 7
    \/ /\ pc = 7 /\ BadTail /\ pc' = 8
    \/ /\ pc = 8 /\ Finish(2, "stopped") /\ pc' = 9
ReplaySpec == ReplayInit /\ [][ReplayNext]_<<vars, pc>>
InterruptedPairDeclined == pc = 9 => Bound = 0
EarlierReceipt == INSTANCE BackupProgress WITH receiptNonce <- 1
EarlierPairPreserved == pc = 9 => EarlierReceipt!Bound = 1
==============================================================
