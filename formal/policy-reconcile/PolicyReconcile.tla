------------------------ MODULE PolicyReconcile ------------------------
EXTENDS Naturals
CONSTANT RenameBeforeJournal
VARIABLE s
vars == <<s>>
(* Atomic renames have visible and directory-fsynced versions. A power cut
   may retain either; an installer SIGKILL retains the visible version.
   old/new abstract exact bytes + SHA; all other bytes are foreign. *)
Init == s = [pc |-> "intent", journal |-> "old", dj |-> "old",
             manifest |-> "old", dm |-> "old", staged |-> FALSE,
             service |-> "old", crashes |-> 0]
Intent == /\ s.pc = "intent"
          /\ s' = [s EXCEPT !.journal = "pending", !.pc = "intent-sync"]
IntentSync == /\ s.pc = "intent-sync"
              /\ s' = [s EXCEPT !.dj = "pending", !.pc = "stop"]
Stop == /\ s.pc = "stop"
        /\ s' = [s EXCEPT !.service = "off", !.pc = "stage"]
Stage == /\ s.pc = "stage"
         /\ s' = [s EXCEPT !.staged = TRUE, !.pc = "rename"]
Rename == /\ s.pc = "rename" /\ s.staged
          /\ s' = [s EXCEPT !.manifest = "new", !.staged = FALSE, !.pc = "manifest-sync"]
ManifestSync == /\ s.pc = "manifest-sync"
                /\ s' = [s EXCEPT !.dm = "new", !.pc = "start"]
Start == /\ s.pc = "start"
         /\ s' = [s EXCEPT !.service = "new", !.pc = "verify"]
Verify == /\ s.pc = "verify" /\ s' = [s EXCEPT !.pc = "complete"]
Complete == /\ s.pc = "complete"
            /\ s' = [s EXCEPT !.journal = "new", !.pc = "complete-sync"]
CompleteSync == /\ s.pc = "complete-sync"
                /\ s' = [s EXCEPT !.dj = "new", !.pc = "done"]
(* Existing restart fallback also restores and fsyncs the old manifest before
   bootstrapping the old supervisor and committing the old receipt. *)
Fail == /\ s.pc \in {"verify", "complete-sync"}
        /\ s' = [s EXCEPT !.pc = IF s.journal = "new" THEN "rollback-intent" ELSE "rollback-stop"]
RollbackIntent == /\ s.pc = "rollback-intent"
                  /\ s' = [s EXCEPT !.journal = "pending", !.pc = "rollback-intent-sync"]
RollbackIntentSync == /\ s.pc = "rollback-intent-sync"
                      /\ s' = [s EXCEPT !.dj = "pending", !.pc = "rollback-stop"]
RollbackStop == /\ s.pc = "rollback-stop"
                /\ s' = [s EXCEPT !.service = "off", !.pc = "rollback-publish"]
RollbackPublish == /\ s.pc = "rollback-publish"
                   /\ s' = [s EXCEPT !.manifest = "old", !.pc = "rollback-sync"]
RollbackSync == /\ s.pc = "rollback-sync"
                /\ s' = [s EXCEPT !.dm = "old", !.pc = "rollback-start"]
RollbackStart == /\ s.pc = "rollback-start"
                 /\ s' = [s EXCEPT !.service = "old", !.pc = "rollback-complete"]
RollbackComplete == /\ s.pc = "rollback-complete"
                    /\ s' = [s EXCEPT !.journal = "old", !.pc = "rollback-done"]
RollbackDone == /\ s.pc = "rollback-done"
                /\ s' = [s EXCEPT !.dj = "old", !.pc = "done"]
Crash == /\ s.pc \notin {"done", "refused", "recover"} /\ s.crashes = 0
         /\ \E j \in {s.journal, s.dj}, m \in {s.manifest, s.dm}:
            s' = [s EXCEPT !.pc = "recover", !.journal = j, !.dj = j,
                  !.manifest = m, !.dm = m, !.staged = FALSE,
                  !.service = "off", !.crashes = 1]
Recover == /\ s.pc = "recover"
           /\ s' = [s EXCEPT !.pc = IF s.journal = "pending" /\ s.manifest \in {"old", "new"}
                                    THEN "stop" ELSE
                                    IF s.journal = s.manifest /\ s.manifest \in {"old", "new"}
                                    THEN "done" ELSE "refused"]
(* Foreign writes are refused regardless of the two authorized byte versions. *)
Foreign == /\ s.pc = "recover" /\ s.manifest # "foreign"
           /\ s' = [s EXCEPT !.manifest = "foreign", !.dm = "foreign"]
Bug == /\ RenameBeforeJournal /\ s.pc = "intent"
       /\ s' = [s EXCEPT !.pc = "stop"]
Next == Intent \/ IntentSync \/ Stop \/ Stage \/ Rename \/ ManifestSync
        \/ Start \/ Verify \/ Complete \/ CompleteSync \/ Fail \/ RollbackStop
        \/ RollbackPublish \/ RollbackSync \/ RollbackStart \/ RollbackComplete
        \/ RollbackDone \/ RollbackIntent \/ RollbackIntentSync \/ Crash \/ Recover \/ Foreign \/ Bug
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
TypeOK == /\ s.journal \in {"old", "pending", "new"}
          /\ s.manifest \in {"old", "new", "foreign"} /\ s.crashes \in 0..1
JournalOwnsManifest == s.manifest # "foreign" =>
    (s.journal = "old" => s.manifest = "old") /\ (s.journal = "new" => s.manifest = "new")
VerifiedCompletion == s.pc = "done" => s.journal = s.manifest /\ s.manifest # "foreign"
DurableIntentBeforeRename == s.manifest = "new" => s.dj \in {"pending", "new"}
Terminates == <>(s.pc \in {"done", "refused"})
=============================================================================
