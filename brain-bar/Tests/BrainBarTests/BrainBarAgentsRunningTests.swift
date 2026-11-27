import XCTest
@testable import BrainBar

/// #1899 R1: the column's number is agents RUNNING now, per harness, from live session state: a live session process, or a
/// hosted Codex thread whose own rollout says a turn is under way. A finished session written in the last 10 minutes is
/// not running (B1). Antigravity (`agy`, any model) and the Gemini CLI are separate harnesses, and the column's Antigravity
/// row shows Antigravity only (B2).
@MainActor final class BrainBarAgentsRunningTests: XCTestCase {
    // MARK: B2: Gemini CLI and Antigravity

    private func running(_ snapshot: String) -> AgentRunning {
        AgentActivityMonitor.parse(snapshot).running
    }

    func testGeminiCLIOnlyIsNotAntigravity() {
        let r = running("""
        19001     1 gemini           gemini --model gemini-2.5-pro
        19009     1 node             node /opt/agents/.npm/bin/gemini --yolo
        """)
        XCTAssertEqual(r.count(.geminiCLI), 2)
        XCTAssertEqual(r.count(.antigravity), 0)
    }

    func testAntigravityCountsWhateverModelItRuns() {
        let r = running("""
        19002     1 agy              agy --dangerously-skip-permissions --model Gemini 3.1 Pro (High)
        19003     1 agy              /opt/agents/.local/bin/agy --model claude-opus-5-5
        19004     1 agy              /opt/agents/.local/bin/agy
        """)
        XCTAssertEqual(r.count(.antigravity), 3, "a non-Gemini agy session is still an Antigravity agent")
        XCTAssertEqual(r.count(.geminiCLI), 0)
        // The session buckets are unchanged: only the Gemini-model agy is in `.gemini` there.
        XCTAssertEqual(AgentActivityMonitor.parse("""
        19002     1 agy              agy --dangerously-skip-permissions --model Gemini 3.1 Pro (High)
        19003     1 agy              /opt/agents/.local/bin/agy --model claude-opus-5-5
        """).count(for: .gemini), 1)
    }

    func testMixedGeminiCLIAndAntigravityMapEachToItsOwnName() throws {
        let snapshot = AgentActivityMonitor.parse("""
        19001     1 gemini           gemini --model gemini-2.5-pro
        19002     1 agy              agy --model Gemini 3.1 Pro (High)
        19003     1 agy              agy --model gpt-6.1-sol
         2918     1 2.1.114          claude --resume 3679128a-f371-445f-82ba-b3946e2f20b6
        """)
        XCTAssertEqual(snapshot.running.count(.geminiCLI), 1)
        XCTAssertEqual(snapshot.running.count(.antigravity), 2)
        XCTAssertEqual(snapshot.running.count(.claude), 1)
        // The column's `.gemini` row is Antigravity's count, never the Gemini CLI's.
        let agents = NDStore.agents(snapshot, nil)
        XCTAssertEqual(agents.agents.first { $0.family == .gemini }?.running, 2)
        XCTAssertEqual(NDProvider.name(.gemini), "Antigravity")
        XCTAssertEqual(NDStore.runningHarness(.gemini), .antigravity)
    }

    func testAnAntigravityServerMakesItsCountUnknown() {
        let r = running("""
        19020     1 agy_acp_server.p /opt/agents/.t3/providers/agy/agy_acp_server.par --stdio
        19002     1 agy              agy --model Gemini 3.1 Pro (High)
        """)
        XCTAssertTrue(r.antigravityHostAlive)
        XCTAssertNil(r.count(.antigravity), "its conversations' liveness cannot be read")
        XCTAssertEqual(r.count(.geminiCLI), 0)
    }

    // MARK: B1, R1-R3: running, proven by a live host

    private var home: URL!

    override func setUpWithError() throws {
        home = FileManager.default.temporaryDirectory.appendingPathComponent("w307-running-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: home, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        // An unreadable fixture must be readable again before it can be removed.
        if let files = FileManager.default.enumerator(atPath: home.path) {
            for case let name as String in files {
                try? FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: home.appendingPathComponent(name).path)
            }
        }
        try? FileManager.default.removeItem(at: home)
    }

    private func event(_ type: String, message: String? = nil) -> String {
        let extra = message.map { #","last_agent_message":"\#($0)""# } ?? ""
        return #"{"timestamp":"2026-11-28T00:00:00Z","type":"event_msg","payload":{"type":"\#(type)"\#(extra)}}"#
    }

    private func meta(_ uuid: String, originator: String, subagent: Bool) -> String {
        let source = subagent ? #""source":{"subagent":"review"},"# : ""
        return #"{"type":"session_meta","payload":{\#(source)"id":"\#(uuid)","originator":"\#(originator)"}}"#
    }

    /// A rollout: its session record, then `lines`, each ending in a newline.
    @discardableResult
    private func rollout(_ n: Int, originator: String = "T3 Code", subagent: Bool = false, lines: [String]) throws -> String {
        let uuid = String(format: "00000000-0000-7000-8000-%012d", n)
        let dir = home.appendingPathComponent(".codex/sessions/2026/11/28", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("rollout-2026-11-28T00-00-00-\(uuid).jsonl")
        try ([meta(uuid, originator: originator, subagent: subagent)] + lines).map { $0 + "\n" }.joined()
            .write(to: url, atomically: true, encoding: .utf8)
        return url.path
    }

    private func append(_ path: String, _ text: String) throws {
        let handle = try XCTUnwrap(FileHandle(forWritingAtPath: path))
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: Data(text.utf8))
    }

    /// A tracker whose hosts hold exactly these files (nil: that host's open files cannot be listed).
    private func tracker(_ held: [Int32: [String]?]) -> AgentTranscriptActivity.CodexTurnTracker {
        AgentTranscriptActivity.CodexTurnTracker(openFiles: { pid in held[pid] ?? [] })
    }

    func testOnlyHeldThreadsMidTurnRun() throws {
        let finished = try rollout(1, lines: [event("task_started"), event("task_complete")])
        let aborted = try rollout(2, lines: [event("task_started"), event("turn_aborted")])
        let restarted = try rollout(3, lines: [event("task_started"), event("task_complete"), event("task_started")])
        let opened = try rollout(4, lines: [])
        let subagent = try rollout(5, subagent: true, lines: [event("task_started")])
        let cli = try rollout(6, originator: "codex_cli_rs", lines: [event("task_started")])
        let held = [finished, aborted, restarted, opened, subagent, cli, "/opt/other/notes.txt"]
        XCTAssertEqual(tracker([10: held]).runningThreads(hosts: [10]), 1, "only the restarted turn runs")
    }

    /// R2: ownership is a live host holding the rollout.
    func testAThreadOrphanedWhenItsServerDiedIsNotRunningWhileAnotherServerLives() throws {
        let orphan = try rollout(11, lines: [event("task_started")])  // server A died mid-turn: nobody holds it
        let live = try rollout(12, lines: [event("task_started")])
        XCTAssertFalse([orphan].isEmpty)
        XCTAssertEqual(tracker([20: [live]]).runningThreads(hosts: [20]), 1, "server B's thread only")
    }

    func testARestartedServerOwnsOnlyWhatItReopened() throws {
        let before = try rollout(21, lines: [event("task_started")])  // left mid-turn by the old process
        let resumed = try rollout(22, lines: [event("task_started"), event("task_complete")])
        let t = tracker([31: [resumed]])  // the restarted server (a new PID) reopened one thread, now idle
        XCTAssertEqual(t.runningThreads(hosts: [31]), 0)
        try append(resumed, event("task_started") + "\n")
        XCTAssertEqual(t.runningThreads(hosts: [31]), 1, "its new turn")
        XCTAssertFalse(before.isEmpty)
    }

    func testTwoLiveServersEachCountTheirOwnAndASharedFileOnce() throws {
        let a = try rollout(41, lines: [event("task_started")])
        let b = try rollout(42, lines: [event("task_started")])
        let shared = try rollout(43, lines: [event("task_started")])
        XCTAssertEqual(tracker([1: [a, shared], 2: [b, shared]]).runningThreads(hosts: [1, 2]), 3)
    }

    /// R1: anything not measured is unknown, never a number.
    func testAHostWhoseOpenFilesCannotBeListedIsUnknown() throws {
        let live = try rollout(51, lines: [event("task_started")])
        let t = AgentTranscriptActivity.CodexTurnTracker(openFiles: { pid in pid == 1 ? [live] : nil })
        XCTAssertEqual(t.runningThreads(hosts: [1]), 1)
        XCTAssertNil(t.runningThreads(hosts: [1, 2]))
    }

    func testAnUnreadableHeldRolloutIsUnknown() throws {
        let path = try rollout(52, lines: [event("task_started")])
        try FileManager.default.setAttributes([.posixPermissions: 0o000], ofItemAtPath: path)
        XCTAssertNil(tracker([1: [path]]).runningThreads(hosts: [1]))
        XCTAssertNil(tracker([1: ["/nonexistent/w307/rollout-2026-11-28T00-00-00-x.jsonl"]]).runningThreads(hosts: [1]), "vanished")
    }

    func testAMarkerLineThatDoesNotParseIsUnknownNotRunning() throws {
        let path = try rollout(53, lines: [event("task_started"), #"{"type":"event_msg","payload":{"type":"task_complete""#])
        XCTAssertNil(tracker([1: [path]]).runningThreads(hosts: [1]))
    }

    /// R3: a completion record longer than any read chunk is still a completion.
    func testACompletionRecordOverOneMiBIsFinished() throws {
        let huge = String(repeating: "y", count: (1 << 20) + 300_000)
        let path = try rollout(54, lines: [event("task_started"), event("task_complete", message: huge)])
        XCTAssertGreaterThan(try XCTUnwrap(FileManager.default.attributesOfItem(atPath: path)[.size] as? Int), 1 << 20)
        let t = tracker([1: [path]])
        XCTAssertEqual(t.runningThreads(hosts: [1]), 0)
        // Read on: a new turn after it, then that turn's own long completion.
        try append(path, event("task_started") + "\n")
        XCTAssertEqual(t.runningThreads(hosts: [1]), 1)
        try append(path, event("task_complete", message: huge) + "\n")
        XCTAssertEqual(t.runningThreads(hosts: [1]), 0)
    }

    func testAHalfWrittenRecordWaitsForItsNewline() throws {
        let path = try rollout(55, lines: [event("task_started")])
        let t = tracker([1: [path]])
        let done = event("task_complete")
        try append(path, String(done.prefix(30)))
        XCTAssertEqual(t.runningThreads(hosts: [1]), 1, "the completion is not written yet")
        try append(path, String(done.dropFirst(30)) + "\n")
        XCTAssertEqual(t.runningThreads(hosts: [1]), 0)
    }

    /// The live lister on the real kernel: this process's own open file is listed.
    func testTheKernelListsAProcessesOpenFiles() throws {
        let path = try rollout(56, lines: [])
        let handle = try XCTUnwrap(FileHandle(forReadingAtPath: path))
        defer { try? handle.close() }
        let files = try XCTUnwrap(AgentTranscriptActivity.openFilePaths(getpid()))
        XCTAssertTrue(files.contains { $0.hasSuffix((path as NSString).lastPathComponent) })
        XCTAssertNil(AgentTranscriptActivity.openFilePaths(-7), "no such process")
    }

    /// The whole path: `ps` and the held rollouts → the monitor's snapshot → the chart's state.
    private func chart(_ ps: String, transcripts: [AgentFamily: Int], held: [Int32: [String]?]) -> NDPipelineState {
        let t = tracker(held)
        let monitor = AgentActivityMonitor(snapshotProvider: { ps }, executablePathResolver: { _ in nil },
                                           transcriptSessions: { transcripts },
                                           hostedCodexThreads: { t.runningThreads(hosts: $0) })
        var i = BrainBarDesignFixture.inputs(.calm)
        i.agents = NDStore.agents(monitor.sample(), nil)
        return NDPipelineState.make(i)
    }

    private let hostAndClaude = """
    59611 93185 codex            /opt/agents/bin/codex app-server -c thread_unload_delay_secs=0
     2918     1 2.1.114          claude --resume 3679128a-f371-445f-82ba-b3946e2f20b6
    """

    func testACompletedRecentTranscriptReachesTheChartAsNotRunning() throws {
        let finished = try rollout(61, lines: [event("task_started"), event("task_complete")])
        let midTurn = try rollout(62, lines: [event("task_started")])
        // Five Claude and four Codex sessions wrote in the last 10 minutes; one Claude process and one held turn run.
        let state = chart(hostAndClaude, transcripts: [.claude: 5, .codex: 4], held: [59611: [finished, midTurn]])
        XCTAssertEqual(state.agentsRunning[.claude], 1, "a finished Claude session is not running")
        XCTAssertEqual(state.agentsRunning[.codex], 1, "the completed thread is not running; the mid-turn one is")
        XCTAssertEqual(state.agentsRunning[.cursor], 0)
        XCTAssertEqual(NDPipelineChart.agentCount(.codex, state: state), 1)
    }

    func testTheChartShowsUnknownWhenAHostCannotBeRead() throws {
        let state = chart(hostAndClaude, transcripts: [:], held: [59611: nil])
        XCTAssertNil(state.agentsRunning[.codex])
        XCTAssertEqual(NDPipelineChart.agentTone(.codex, state: state), .neutral)
        XCTAssertNil(NDPipelineChart.agentCount(.codex, state: state))
        XCTAssertEqual(NDPipelineChart.providerLabel(.codex, state: state), "Codex, running agents not known")
        XCTAssertEqual(state.agentsRunning[.claude], 1, "other harnesses keep their proof")
    }

    func testWithNoServerNoHostedThreadIsAsked() throws {
        let state = chart(" 2918     1 2.1.114          claude --resume x\n", transcripts: [:], held: [:])
        XCTAssertEqual(state.agentsRunning[.codex], 0, "no app server hosts anything")
    }

    func testAnUnmeasuredSnapshotIsUnknownEverywhere() {
        let snapshot = AgentActivityMonitor(snapshotProvider: { nil }, executablePathResolver: { _ in nil }).sample()
        XCTAssertTrue(AgentHarness.allCases.allSatisfy { snapshot.running.count($0) == nil })
    }
    // MARK: Cursor pass on 96c61fd8 (K1-K5)

    /// K1: a list that fills its buffer may be cut short: read again bigger, and give up as unknown, never a short list.
    func testAFullFDListIsReadAgainAndNeverReturnedCutShort() {
        var calls = 0
        let alwaysFull: AgentTranscriptActivity.ListFDs = { _, buffer, bytes in
            if buffer == nil { return 64 * Int32(MemoryLayout<proc_fdinfo>.stride) }
            calls += 1
            buffer?.initializeMemory(as: UInt8.self, repeating: 0, count: Int(bytes))
            return bytes
        }
        XCTAssertNil(AgentTranscriptActivity.openFilePaths(getpid(), listFDs: alwaysFull))
        XCTAssertEqual(calls, AgentTranscriptActivity.listRetries)

        var sizes: [Int32] = []
        let fullOnce: AgentTranscriptActivity.ListFDs = { _, buffer, bytes in
            if buffer == nil { return 64 * Int32(MemoryLayout<proc_fdinfo>.stride) }
            sizes.append(bytes)
            buffer?.initializeMemory(as: UInt8.self, repeating: 0, count: Int(bytes))  // fd type 0: not a file
            return sizes.count == 1 ? bytes : bytes / 2
        }
        XCTAssertEqual(AgentTranscriptActivity.openFilePaths(getpid(), listFDs: fullOnce), [])
        XCTAssertEqual(sizes.count, 2)
        XCTAssertEqual(sizes[1], sizes[0] * 2, "read again into a buffer twice the size")
    }

    /// K2: a new file is read within the sample's budget and resumed on the next; until it is caught up, unknown.
    func testALargeNewRolloutIsUnknownUntilCaughtUp() throws {
        let filler = #"{"type":"response_item","payload":{"type":"message","text":"\#(String(repeating: "z", count: 4000))"}}"#
        let lines = [event("task_started")] + Array(repeating: filler, count: 5000) + [event("task_complete")]
        let path = try rollout(71, lines: lines)
        let size = try XCTUnwrap(FileManager.default.attributesOfItem(atPath: path)[.size] as? Int)
        XCTAssertGreaterThan(size, 2 * AgentTranscriptActivity.CodexTurnTracker.defaultSampleBudget)
        let t = tracker([1: [path]])
        var samples: [Int?] = []
        for _ in 0..<(size / AgentTranscriptActivity.CodexTurnTracker.defaultSampleBudget + 1) {
            samples.append(t.runningThreads(hosts: [1]))
        }
        XCTAssertEqual(samples.dropLast(), Array(repeating: nil, count: samples.count - 1), "unknown while catching up")
        XCTAssertEqual(samples.last, 0, "caught up: the turn completed")
        // Another held thread is still proven while the big one catches up? No: the harness stays unknown until then.
        let small = try rollout(72, lines: [event("task_started")])
        let big = try rollout(73, lines: lines)
        let u = tracker([1: [small, big]])
        XCTAssertNil(u.runningThreads(hosts: [1]))
    }

    /// K3: failed samples never grow the kept state beyond what live hosts hold.
    func testFailedSamplesDoNotGrowTheKeptState() throws {
        let unreadable = try rollout(80, lines: [event("task_started")])
        try FileManager.default.setAttributes([.posixPermissions: 0o000], ofItemAtPath: unreadable)
        let paths = try (81...90).map { try rollout($0, lines: [event("task_started")]) }
        final class Tick: @unchecked Sendable { var n = 0 }
        let tick = Tick()
        let t = AgentTranscriptActivity.CodexTurnTracker(openFiles: { pid in
            pid == 1 ? [paths[tick.n % paths.count], unreadable] : nil
        })
        for n in 0..<paths.count {
            tick.n = n
            XCTAssertNil(t.runningThreads(hosts: [1]), "one held file is unreadable")
            XCTAssertLessThanOrEqual(t.trackedFiles, 2, "only what is held now")
        }
        tick.n = 0
        XCTAssertNil(t.runningThreads(hosts: [1, 2]), "the second host cannot be listed")
        XCTAssertLessThanOrEqual(t.trackedFiles, 2)
    }

    /// K4: the same path holding a different file starts over, whether it was replaced, rewritten in place or shrank.
    func testAReplacedFileAtTheSamePathStartsOver() throws {
        let path = try rollout(91, lines: [event("task_started")])
        let t = tracker([1: [path]])
        XCTAssertEqual(t.runningThreads(hosts: [1]), 1)
        // Replaced: a new file renamed over the old one (a new inode), longer than what was read.
        let uuid = String(format: "00000000-0000-7000-8000-%012d", 91)
        let replaced = [meta(uuid, originator: "T3 Code", subagent: false), event("task_started"), event("task_complete"),
                        event("task_started"), event("task_complete")].map { $0 + "\n" }.joined()
        try replaced.write(toFile: path, atomically: true, encoding: .utf8)
        XCTAssertEqual(t.runningThreads(hosts: [1]), 0)
        // Rewritten in place (same inode), at least as long, its first bytes changed.
        let handle = try XCTUnwrap(FileHandle(forWritingAtPath: path))
        let rewritten = [meta(uuid, originator: "Codex Desktop", subagent: false), event("task_started"),
                         event("task_started"), event("task_started"), event("task_started")].map { $0 + "\n" }.joined()
        try handle.truncate(atOffset: 0)
        try handle.write(contentsOf: Data(rewritten.utf8))
        try handle.close()
        XCTAssertEqual(t.runningThreads(hosts: [1]), 1)
        // Shrunk below what was read: read again from the start.
        try ([meta(uuid, originator: "T3 Code", subagent: false), event("task_started"), event("task_complete")]
            .map { $0 + "\n" }.joined()).write(toFile: path, atomically: false, encoding: .utf8)
        XCTAssertEqual(t.runningThreads(hosts: [1]), 0)
    }

    /// K5: a complete session record whose originator cannot be read is unknown, never hosted.
    func testAnUnreadableSessionRecordIsUnknown() throws {
        let dir = home.appendingPathComponent(".codex/sessions/2026/11/28", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let path = dir.appendingPathComponent("rollout-2026-11-28T00-00-00-00000000-0000-7000-8000-000000000095.jsonl").path
        try (#"{"type":"session_meta","payload":{"id":"x","originator":"# + "\n" + event("task_started") + "\n")
            .write(toFile: path, atomically: true, encoding: .utf8)
        XCTAssertNil(tracker([1: [path]]).runningThreads(hosts: [1]))
        // A sub-agent record is known (not hosted) even when the rest of it does not parse.
        let sub = dir.appendingPathComponent("rollout-2026-11-28T00-00-00-00000000-0000-7000-8000-000000000096.jsonl").path
        try (#"{"type":"session_meta","payload":{"source":{"subagent":"review"},"id":"#  + "\n" + event("task_started") + "\n")
            .write(toFile: sub, atomically: true, encoding: .utf8)
        XCTAssertEqual(tracker([1: [sub]]).runningThreads(hosts: [1]), 0)
    }
}
