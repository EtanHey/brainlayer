-------------------------- MODULE BackupProgress --------------------------
EXTENDS Naturals, FiniteSets
CONSTANT Corrected
C == {1, 2}
Actors == {1, 2, 3}  \* Calendar invocations 1/2; independent manual contender 3.
VARIABLES phase, lock, nonce, bodySpan, receiptOwner, receiptNonce, receiptSpan,
          receipt, terminal, gap, ambiguous, tail, verifiedHistory, paused
vars == <<phase, lock, nonce, bodySpan, receiptOwner, receiptNonce, receiptSpan,
          receipt, terminal, gap, ambiguous, tail, verifiedHistory, paused>>
Pending == receipt \in {"pending", "in_progress"}
OpenCalendars == {c \in C: phase[c] \in {"contending", "active", "lost", "dead"}}
Completed == {c \in C: phase[c] = "finished"}
Init == /\ phase = [a \in Actors |-> "new"] /\ lock = 0
        /\ nonce = [a \in Actors |-> 0] /\ bodySpan = [a \in Actors |-> {}]
        /\ receiptOwner = 0 /\ receiptNonce = 0 /\ receiptSpan = {}
        /\ receipt = "none" /\ terminal = [c \in C |-> "none"]
        /\ gap = {} /\ ambiguous = {} /\ tail = FALSE
        /\ verifiedHistory = {0} /\ paused = FALSE
Spawn(a, tagged) ==
    /\ phase[a] = "new" /\ ~paused /\ (a # 2 \/ phase[1] = "finished")
    /\ phase' = [phase EXCEPT ![a] = "contending"]
    /\ nonce' = [nonce EXCEPT ![a] = IF a \in C /\ tagged THEN a ELSE 0]
    /\ UNCHANGED <<lock, bodySpan, receiptOwner, receiptNonce, receiptSpan,
                   receipt, terminal, gap, ambiguous, tail, verifiedHistory, paused>>
Admit(a) ==
    /\ phase[a] = "contending" /\ lock = 0 /\ lock' = a
    /\ phase' = [phase EXCEPT ![a] = "active"]
    \* Admission time lies inside these spawn/finish intervals, independent of ownership.
    /\ bodySpan' = [bodySpan EXCEPT ![a] = OpenCalendars]
    /\ UNCHANGED <<nonce, receiptOwner, receiptNonce, receiptSpan, receipt,
                   terminal, gap, ambiguous, tail, verifiedHistory, paused>>
Lose(a) ==
    /\ phase[a] = "contending" /\ lock # 0
    /\ phase' = [phase EXCEPT ![a] = "lost"]
    /\ UNCHANGED <<lock, nonce, bodySpan, receiptOwner, receiptNonce, receiptSpan,
                   receipt, terminal, gap, ambiguous, tail, verifiedHistory, paused>>
Snapshot(a) ==
    /\ phase[a] = "active" /\ lock = a /\ receiptOwner # a
    /\ receiptOwner' = a /\ receiptNonce' = nonce[a]
    /\ receiptSpan' = bodySpan[a] /\ receipt' = "pending"
    /\ UNCHANGED <<phase, lock, nonce, bodySpan, terminal, gap, ambiguous,
                   tail, verifiedHistory, paused>>
Advance(a, before, after) ==
    /\ phase[a] = "active" /\ receiptOwner = a /\ receipt = before
    /\ receipt' = after
    /\ verifiedHistory' = IF after = "verified" THEN verifiedHistory \cup {a} ELSE verifiedHistory
    /\ UNCHANGED <<phase, lock, nonce, bodySpan, receiptOwner, receiptNonce,
                   receiptSpan, terminal, gap, ambiguous, tail, paused>>
Death(a) ==
    /\ phase[a] = "active" /\ phase' = [phase EXCEPT ![a] = "dead"]
    /\ lock' = IF lock = a THEN 0 ELSE lock  \* OS releases the body lock before calendar finish.
    /\ UNCHANGED <<nonce, bodySpan, receiptOwner, receiptNonce, receiptSpan,
                   receipt, terminal, gap, ambiguous, tail, verifiedHistory, paused>>
Finish(a, status) ==
    /\ phase[a] \in {"active", "lost", "dead"}
    /\ (phase[a] # "lost" \/ status = "failed")
    /\ phase' = [phase EXCEPT ![a] = "finished"]
    /\ lock' = IF lock = a THEN 0 ELSE lock
    /\ terminal' = IF a \in C THEN [terminal EXCEPT ![a] = status] ELSE terminal
    \* The child can die without a final producer receipt; calendar still records its result.
    /\ UNCHANGED <<nonce, bodySpan, receiptOwner, receiptNonce, receiptSpan,
                   receipt, gap, ambiguous, tail, verifiedHistory, paused>>
Gap == /\ gap' = gap \cup OpenCalendars
       /\ UNCHANGED <<phase, lock, nonce, bodySpan, receiptOwner, receiptNonce,
                      receiptSpan, receipt, terminal, ambiguous, tail, verifiedHistory, paused>>
Duplicate(c) == /\ c \in Completed /\ ambiguous' = ambiguous \cup {c}
                /\ UNCHANGED <<phase, lock, nonce, bodySpan, receiptOwner, receiptNonce,
                               receiptSpan, receipt, terminal, gap, tail, verifiedHistory, paused>>
BadTail == /\ Completed # {} /\ tail' = TRUE
           /\ gap' = gap \cup OpenCalendars
           /\ UNCHANGED <<phase, lock, nonce, bodySpan, receiptOwner, receiptNonce,
                          receiptSpan, receipt, terminal, ambiguous, verifiedHistory, paused>>
PauseResume == /\ paused' = ~paused
               /\ UNCHANGED <<phase, lock, nonce, bodySpan, receiptOwner, receiptNonce,
                              receiptSpan, receipt, terminal, gap, ambiguous, tail, verifiedHistory>>
Next == (\E a \in Actors: (\E tagged \in BOOLEAN: Spawn(a, tagged)) \/ Admit(a) \/ Lose(a)
          \/ Snapshot(a) \/ Death(a) \/ (\E s \in {"ok", "failed", "stopped"}: Finish(a, s))
          \/ Advance(a, "pending", "in_progress") \/ Advance(a, "in_progress", "uploaded")
          \/ Advance(a, "uploaded", "verified"))
        \/ Gap \/ BadTail \/ (\E c \in C: Duplicate(c)) \/ PauseResume
Spec == Init /\ [][Next]_vars

OldCandidates == receiptSpan \cap Completed
ProofCandidates == {c \in Completed: receiptNonce # 0 /\ nonce[c] = receiptNonce
                     /\ c \notin gap \cup ambiguous}
Candidates == IF Corrected THEN ProofCandidates ELSE OldCandidates
Bound == IF Candidates = {} THEN 0 ELSE CHOOSE c \in Candidates: TRUE
Projected == [state |-> IF Pending /\ Bound # 0 THEN "failed" ELSE receipt,
              history |-> verifiedHistory,
              activity |-> IF receipt = "in_progress" /\ Bound = 0
                          THEN IF Corrected THEN "unknown" ELSE "uploading" ELSE "none"]
NoWrongOwner == Pending /\ Bound # 0 => Bound = receiptOwner
NoWrongActiveOwner == receipt = "in_progress" /\ Bound # 0 => Bound = receiptOwner
OrdinaryTerminalHandled == Pending /\ receiptOwner \in ProofCandidates => Projected.state = "failed"
CompletedReceiptNotRevoked == receipt \in {"uploaded", "verified"} => Projected.state = receipt
LegacyUnconfirmed == Corrected /\ receiptNonce = 0 => Bound = 0
HistoryForwarded == Projected.history = verifiedHistory /\ 0 \in Projected.history
UnprovenIsNotLive == Corrected /\ receipt = "in_progress" /\ Bound = 0 => Projected.activity = "unknown"
TypeOK == /\ phase \in [Actors -> {"new", "contending", "active", "lost", "dead", "finished"}]
          /\ lock \in 0..3 /\ receiptOwner \in 0..3 /\ receiptNonce \in 0..2
          /\ nonce \in [Actors -> 0..2] /\ bodySpan \in [Actors -> SUBSET C]
          /\ receiptSpan \subseteq C /\ gap \subseteq C /\ ambiguous \subseteq C
          /\ terminal \in [C -> {"none", "ok", "failed", "stopped"}]
          /\ receipt \in {"none", "pending", "in_progress", "uploaded", "verified"}
          /\ verifiedHistory \subseteq 0..3 /\ tail \in BOOLEAN /\ paused \in BOOLEAN
=============================================================================
