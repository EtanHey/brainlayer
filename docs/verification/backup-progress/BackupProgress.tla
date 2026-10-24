-------------------------- MODULE BackupProgress --------------------------
EXTENDS Naturals, FiniteSets
CONSTANT Corrected
VARIABLES attempt, phase, receipt, receiptAttempt, terminal, paused, other
vars == <<attempt, phase, receipt, receiptAttempt, terminal, paused, other>>

Init == /\ attempt = 0 /\ phase = "idle"
        /\ receipt = "none" /\ receiptAttempt = 0
        /\ terminal = [a \in 0..2 |-> "none"]
        /\ paused = FALSE /\ other = "none"

Start == /\ attempt < 2 /\ phase \in {"idle", "ended"} /\ ~paused
         /\ attempt' = attempt + 1 /\ phase' = "snapshot"
         /\ UNCHANGED <<receipt, receiptAttempt, terminal, paused, other>>
Snapshot == /\ phase = "snapshot" /\ phase' = "upload"
            /\ receipt' = "pending" /\ receiptAttempt' = attempt
            /\ UNCHANGED <<attempt, terminal, paused, other>>
Checkpoint == /\ phase = "upload" /\ receipt = "pending"
              /\ receipt' = "in_progress"
              /\ UNCHANGED <<attempt, phase, receiptAttempt, terminal, paused, other>>
Verify == /\ phase = "upload" /\ receipt \in {"pending", "in_progress"}
          /\ receipt' = "verified"
          /\ UNCHANGED <<attempt, phase, receiptAttempt, terminal, paused, other>>
Finish(status) == /\ phase \in {"snapshot", "upload"}
                  /\ terminal' = [terminal EXCEPT ![attempt] = status]
                  /\ phase' = "ended"
                  /\ UNCHANGED <<attempt, receipt, receiptAttempt, paused, other>>
Pause == /\ ~paused /\ paused' = TRUE
         /\ UNCHANGED <<attempt, phase, receipt, receiptAttempt, terminal, other>>
Resume == /\ paused /\ paused' = FALSE
          /\ UNCHANGED <<attempt, phase, receipt, receiptAttempt, terminal, other>>
Death == /\ phase \in {"snapshot", "upload"} /\ phase' = "dead"
         /\ UNCHANGED <<attempt, receipt, receiptAttempt, terminal, paused, other>>
RecordDeath == /\ phase = "dead" /\ phase' = "ended"
               /\ terminal' = [terminal EXCEPT ![attempt] = "stopped"]
               /\ UNCHANGED <<attempt, receipt, receiptAttempt, paused, other>>
OtherJob(status) == /\ other' = status
                    /\ UNCHANGED <<attempt, phase, receipt, receiptAttempt, terminal, paused>>
Next == Start \/ Snapshot \/ Checkpoint \/ Verify \/ Pause \/ Resume \/ Death
        \/ RecordDeath \/ (\E s \in {"stopped", "failed", "ok"}: Finish(s))
        \/ (\E s \in {"failed", "ok"}: OtherJob(s))
Spec == Init /\ [][Next]_vars

Pending == receipt \in {"pending", "in_progress"}
Progress == Pending /\ (~Corrected \/ terminal[receiptAttempt] = "none")
NoTerminalProgress == terminal[receiptAttempt] # "none" => ~Progress
NoStoppedUploadProgress == receipt = "in_progress" /\ terminal[receiptAttempt] = "stopped" => ~Progress
VerifiedReceiptPreserved == receipt = "verified" => ~Progress
TypeOK == /\ attempt \in 0..2 /\ receiptAttempt \in 0..attempt
          /\ phase \in {"idle", "snapshot", "upload", "dead", "ended"}
          /\ receipt \in {"none", "pending", "in_progress", "verified"}
          /\ terminal \in [0..2 -> {"none", "stopped", "failed", "ok"}]
          /\ paused \in BOOLEAN /\ other \in {"none", "failed", "ok"}
=============================================================================
