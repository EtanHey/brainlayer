--------------------------- MODULE CalendarGuard ---------------------------
EXTENDS Naturals, TLC
CONSTANTS OffLoop, Budgeted, Fault, Mode, Kill, OperationBudget, HeartbeatBudget, Horizon
VARIABLE s
vars == <<s>>
Operations == {"pre", "runner", "custody", "finish", "summary"}
Init == s = [pc |-> "pre", work |-> FALSE, left |-> 0, age |-> 0,
             tick |-> 0, heartbeat |-> 0, childAge |-> 0, guard |-> "off",
             parent |-> TRUE, lock |-> TRUE, child |-> "none", durable |-> FALSE,
             sent |-> FALSE, registered |-> FALSE, ran |-> 0, killed |-> FALSE,
             summary |-> FALSE, envKilled |-> FALSE]

BeginOperation == /\ s.parent /\ s.pc \in Operations /\ ~s.work
                  /\ \E delay \in {0, 1, 3}:
                     s' = [s EXCEPT !.work = TRUE, !.left = delay, !.age = 0]
Wedged == Mode \in {"wedge", "loop"} /\ s.pc = Fault
CompleteOperation ==
    /\ s.parent /\ s.work /\ s.left = 0 /\ ~Wedged
    /\ s' = [s EXCEPT !.work = FALSE,
              !.pc = CASE s.pc = "pre" -> "launch"
                       [] s.pc = "runner" -> "spawn"
                       [] s.pc = "custody" -> "register"
                       [] s.pc = "finish" -> "shutdown"
                       [] OTHER -> "close",
              !.durable = IF s.pc = "custody" THEN TRUE ELSE @,
              !.summary = IF s.pc = "summary" THEN TRUE ELSE @]
LaunchGuard == /\ s.parent /\ s.pc = "launch"
               /\ s' = [s EXCEPT !.guard = "H", !.pc = "runner", !.heartbeat = 0]
Spawn == /\ s.parent /\ s.pc = "spawn"
         /\ s' = [s EXCEPT !.child = "gated", !.pc = "custody"]
WriteRegistration == /\ s.parent /\ s.pc = "register" /\ s.durable
                     /\ s' = [s EXCEPT !.sent = TRUE, !.pc = "release"]
ReadRegistration == /\ s.guard \in {"H", "reap"} /\ s.sent /\ ~s.registered
                    /\ s' = [s EXCEPT !.registered = TRUE]
ReleaseGate == /\ s.parent /\ s.pc = "release" /\ s.durable /\ s.sent
               /\ s' = [s EXCEPT !.child = "running", !.ran = @ + 1, !.pc = "wait"]
FinishChild == /\ s.parent /\ s.pc = "wait" /\ s.childAge = 2
               /\ s' = [s EXCEPT !.child = "gone", !.pc = "finish"]
(* The lease-channel EOF is closed immediately before Q+stdin EOF. The guard
   can observe either EOF first; both enter the same registered-group reap. *)
Quiesce == /\ s.parent /\ s.pc = "shutdown"
           /\ s' = [s EXCEPT !.guard = "Q", !.pc = "guard-exit", !.heartbeat = 0]
GuardEOF == /\ s.guard = "Q"
            /\ s' = [s EXCEPT !.guard = "reap"]
(* Buffered +pid is consumed before stdin EOF cleanup. *)
GuardReap == /\ s.guard = "reap" /\ (~s.sent \/ s.registered)
             /\ s' = [s EXCEPT !.guard = "gone",
                       !.child = IF s.registered \/ s.child = "gated" THEN "gone" ELSE @,
                       !.lock = IF ~s.parent THEN FALSE ELSE @,
                       !.pc = IF s.parent THEN "summary" ELSE @]
Close == /\ s.parent /\ s.pc = "close"
         /\ s' = [s EXCEPT !.pc = "done", !.parent = FALSE, !.lock = FALSE]
EnvironmentKill ==
    /\ ~s.envKilled /\ Kill \in {"parent", "guard", "both"}
    /\ IF Kill = "guard" THEN s.pc = "custody" /\ s.work ELSE s.pc = "wait"
    /\ s' = [s EXCEPT !.envKilled = TRUE,
              !.parent = IF Kill \in {"parent", "both"} THEN FALSE ELSE @,
              !.guard = IF Kill \in {"guard", "both"} THEN "gone" ELSE "reap",
              !.lock = IF Kill = "both" THEN FALSE ELSE @]

(* finish already used an executor before #1519; the old synchronous runner
   record and prepare/custody paths are the historical starvation target. *)
LoopAvailable == ~(s.work /\ ((~OffLoop /\ s.pc # "finish")
                             \/ (Mode = "loop" /\ s.pc = Fault)))
SendH == s.guard = "H" /\ s.parent /\ LoopAvailable
         /\ (~Budgeted \/ ~s.work \/ s.age < OperationBudget)
NeedsTime == (s.parent /\ s.work /\ (s.left > 0 \/ Wedged))
             \/ (s.parent /\ s.pc = "wait" /\ s.childAge < 2)
             \/ (s.guard = "Q" /\ (s.heartbeat < 1 \/ Mode = "q-stall"))
Tick == /\ NeedsTime /\ s.tick < Horizon
        /\ ~(s.guard = "H" /\ s.heartbeat >= HeartbeatBudget)
        /\ ~(s.guard = "Q" /\ s.heartbeat >= HeartbeatBudget + 1)
        /\ s' = [s EXCEPT !.tick = @ + 1,
                  !.age = IF s.work THEN @ + 1 ELSE @,
                  !.left = IF s.work /\ s.left > 0 /\ ~Wedged THEN @ - 1 ELSE @,
                  !.heartbeat = IF SendH THEN 0 ELSE @ + 1,
                  !.childAge = IF s.child = "running" THEN @ + 1 ELSE @]
GuardTimeout == /\ ((s.guard = "H" /\ s.heartbeat >= HeartbeatBudget)
                   \/ (s.guard = "Q" /\ s.heartbeat >= HeartbeatBudget + 1))
                /\ s' = [s EXCEPT !.parent = FALSE, !.killed = TRUE, !.guard = "reap",
                          !.child = IF s.child = "gated" THEN "gone" ELSE @]
Progress == BeginOperation \/ CompleteOperation \/ LaunchGuard \/ Spawn
            \/ WriteRegistration \/ ReleaseGate \/ FinishChild \/ Quiesce \/ Close
Next == Progress \/ ReadRegistration \/ GuardEOF \/ GuardReap \/ Tick
        \/ GuardTimeout \/ EnvironmentKill
Spec == Init /\ [][Next]_vars /\ WF_vars(Progress) /\ WF_vars(Tick)
        /\ WF_vars(GuardTimeout) /\ WF_vars(GuardEOF) /\ WF_vars(GuardReap)
        /\ WF_vars(ReadRegistration)
TypeOK == /\ s.pc \in Operations \cup {"launch", "spawn", "register", "release",
                                       "wait", "shutdown", "guard-exit", "close", "done"}
          /\ s.guard \in {"off", "H", "Q", "reap", "gone"}
          /\ s.child \in {"none", "gated", "running", "gone"}
          /\ s.ran \in 0..1 /\ s.tick \in 0..Horizon
ExactlyOneChild == s.ran <= 1 /\ (s.ran = 1 => s.durable /\ s.sent)
(* Deliberately false reachability assertion: the parent can die after sending
   registration but before receipt. This is not a product safety promise. *)
ParentDeathAfterReceipt == ~(s.envKilled /\ ~s.parent /\ s.sent /\ ~s.registered)
RegistrationAcknowledged == s.ran = 1 => s.registered
NoFalseKill == (Mode = "healthy" /\ Kill = "none") => ~s.killed
Settled == ~s.lock /\ s.child \in {"none", "gone"}
BoundedTermination == s.tick = Horizon => Settled
EventuallyReleased == <>Settled
=============================================================================
