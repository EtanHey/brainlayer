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
        return #"{"timestamp":"2026-11-28T00:00:00Z","ordinal":1,"type":"event_msg","payload":{"type":"\#(type)"\#(extra)}}"#
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
        AgentTranscriptActivity.CodexTurnTracker(openFiles: { pid in (held[pid] ?? []).map { $0.map(Self.heldFile) } })
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
        let t = AgentTranscriptActivity.CodexTurnTracker(openFiles: { pid in pid == 1 ? [Self.heldFile(live)] : nil })
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
        XCTAssertTrue(files.contains { $0.path.hasSuffix((path as NSString).lastPathComponent) })
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
            pid == 1 ? [paths[tick.n % paths.count], unreadable].map(Self.heldFile) : nil
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
    // MARK: #1902 sampler hardening (w327)

    func testNoHostSampleEvictsTheLastHostsPartialRecord() throws {
        let path = try rollout(101, lines: [event("task_started")])
        try append(path, "{unfinished")
        let t = tracker([1: [path]])
        XCTAssertEqual(t.runningThreads(hosts: [1]), 1)
        XCTAssertEqual(t.trackedFiles, 1)
        let monitor = AgentActivityMonitor(snapshotProvider: { "" }, executablePathResolver: { _ in nil },
                                           hostedCodexThreads: { t.runningThreads(hosts: $0) })
        XCTAssertEqual(monitor.sample().running.count(.codex), 0)
        XCTAssertEqual(t.trackedFiles, 0, "no host must release cached state, including its partial record")
        // An empty host list never needs an fd listing.
        let noListing = AgentTranscriptActivity.CodexTurnTracker(openFiles: { _ in
            XCTFail("no live host to list")
            return nil
        })
        XCTAssertEqual(noListing.runningThreads(hosts: []), 0)
    }

    func testAFIFOAtARolloutPathIsUnknownWithoutWaitingForAWriter() throws {
        let path = home.appendingPathComponent("rollout-fifo.jsonl").path
        XCTAssertEqual(mkfifo(path, 0o600), 0)
        // Rescue the old blocking open so the red test terminates. An RDWR fd also keeps the
        // FIFO alive until the sample returns. The fixed sampler returns before this fires.
        let rescued = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        DispatchQueue.global().asyncAfter(deadline: .now() + 0.3) {
            let fd = open(path, O_RDWR | O_NONBLOCK)
            if fd >= 0 { _ = write(fd, "\n", 1) }
            rescued.signal()
            _ = release.wait(timeout: .now() + 2)
            if fd >= 0 { close(fd) }
        }
        defer { release.signal(); _ = rescued.wait(timeout: .now() + 3) }
        let start = Date()
        XCTAssertNil(tracker([1: [path]]).runningThreads(hosts: [1]))
        XCTAssertLessThan(Date().timeIntervalSince(start), 0.25, "must not wait for the rescue writer")
    }

    /// CodeRabbit on #1904 (outside the diff): the recent-session count reads each rollout's first line through a
    /// second open path, which blocked on a FIFO until a writer appeared.
    func testAFIFOAtARolloutPathDoesNotBlockTheFirstLineRead() throws {
        let path = home.appendingPathComponent("rollout-fifo-first-line.jsonl").path
        XCTAssertEqual(mkfifo(path, 0o600), 0)
        let rescued = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        DispatchQueue.global().asyncAfter(deadline: .now() + 0.3) {
            let fd = open(path, O_RDWR | O_NONBLOCK)
            if fd >= 0 { _ = write(fd, "\n", 1) }
            rescued.signal()
            _ = release.wait(timeout: .now() + 2)
            if fd >= 0 { close(fd) }
        }
        defer { release.signal(); _ = rescued.wait(timeout: .now() + 3) }
        let start = Date()
        XCTAssertEqual(AgentTranscriptActivity.codexFirstLine(URL(fileURLWithPath: path)), Data())
        XCTAssertLessThan(Date().timeIntervalSince(start), 0.25, "must not wait for the rescue writer")
    }

    func testSameSizeRewriteBeyondTheHeadInvalidatesTheTurn() throws {
        let padding = String(repeating: " ", count: 512)
        let path = try rollout(102, lines: [padding, event("task_started")])
        let t = tracker([1: [path]])
        XCTAssertEqual(t.runningThreads(hosts: [1]), 1)
        let before = try Data(contentsOf: URL(fileURLWithPath: path))
        let text = try String(contentsOfFile: path, encoding: .utf8)
        // One extra byte in complete is balanced by one removed padding byte: same size/head/inode.
        let after = text.replacingOccurrences(of: padding, with: String(padding.dropLast()))
            .replacingOccurrences(of: "task_started", with: "task_complete")
        XCTAssertEqual(after.utf8.count, before.count)
        XCTAssertEqual(Data(after.utf8).prefix(256), before.prefix(256))
        let handle = try XCTUnwrap(FileHandle(forWritingAtPath: path))
        try handle.write(contentsOf: Data(after.utf8))
        try handle.close()
        XCTAssertEqual(t.runningThreads(hosts: [1]), 0, "consumed bytes changed beyond the cached head")
    }

    func testAnOversizedUnterminatedRecordIsUnknownAndRecoversAtANewline() throws {
        let path = try rollout(103, lines: [event("task_started")])
        let t = tracker([1: [path]])
        XCTAssertEqual(t.runningThreads(hosts: [1]), 1)
        // 3 MiB exceeds the supported 2 MiB record cap; spans several reads and samples.
        for _ in 0..<2 {
            try append(path, String(repeating: "x", count: 1 << 20))
            XCTAssertEqual(t.runningThreads(hosts: [1]), 1, "supported partial records carry across samples")
        }
        try append(path, String(repeating: "x", count: 1 << 20))
        XCTAssertNil(t.runningThreads(hosts: [1]), "never retain an unbounded record or guess the turn")
        try append(path, "still oversized")
        XCTAssertNil(t.runningThreads(hosts: [1]))
        try append(path, "\n")
        XCTAssertNil(t.runningThreads(hosts: [1]), "a boundary alone cannot prove a turn")
        try append(path, event("task_complete") + "\n")
        XCTAssertEqual(t.runningThreads(hosts: [1]), 0)
        try append(path, event("task_started") + "\n")
        XCTAssertEqual(t.runningThreads(hosts: [1]), 1)
    }

    nonisolated private static func heldFile(_ path: String) -> AgentTranscriptActivity.HeldFile {
        var st = stat()
        _ = stat(path, &st)
        return .init(path: path, device: UInt64(UInt32(bitPattern: st.st_dev)), inode: UInt64(st.st_ino))
    }

    func testKernelArgumentBoundariesDecideTheCodexRole() {
        let row = "59611 1 codex            codex --cd /fixture/home/AI Tools app-server"
        let server = AgentActivityMonitor.parse(row, argv: { _ in
            ["codex", "--cd", "/fixture/home/AI Tools", "app-server"]
        })
        XCTAssertEqual(server.running.codexHosts, [59611])
        XCTAssertEqual(server.count(for: .codex), 0, "an app server is not a CLI session")
        let prompt = AgentActivityMonitor.parse("59611 1 codex            codex --cd /fixture/home/AI Tools explain app-server", argv: { _ in
            ["codex", "--cd", "/fixture/home/AI Tools", "explain app-server"]
        })
        XCTAssertEqual(prompt.running.codexHosts, [])
        XCTAssertEqual(prompt.running.count(.codex), 1, "mentioning app-server in a prompt stays a CLI")
    }

    func testAReopenedPathMustMatchTheHostsDescriptor() throws {
        let path = try rollout(104, lines: [event("task_complete")])
        let held = try XCTUnwrap(FileHandle(forReadingAtPath: path))
        defer { try? held.close() }
        let files = try XCTUnwrap(AgentTranscriptActivity.openFilePaths(getpid()))
        let original = try XCTUnwrap(files.first { $0.path == path || $0.path == "/private" + path })
        // Preserve the kernel's original path spelling while replacing its pathname's target.
        final class Listing: @unchecked Sendable {
            var file: AgentTranscriptActivity.HeldFile
            init(_ file: AgentTranscriptActivity.HeldFile) { self.file = file }
        }
        let listing = Listing(original)
        let t = AgentTranscriptActivity.CodexTurnTracker(openFiles: { _ in [listing.file] })
        XCTAssertEqual(t.runningThreads(hosts: [1]), 0)
        let replacement = try String(contentsOfFile: path, encoding: .utf8)
            .replacingOccurrences(of: "task_complete", with: "task_started")
        try replacement.write(toFile: path, atomically: true, encoding: .utf8)
        XCTAssertNil(t.runningThreads(hosts: [1]), "the host holds the idle original, not this running replacement")
        // A new listing after a host really reopens the replacement proves the new file normally.
        listing.file = Self.heldFile(original.path)
        XCTAssertEqual(t.runningThreads(hosts: [1]), 1, "the same tracker retries successfully on a fresh host listing")
    }

    func testKernelProcargsRetainsSpacesEmptyValuesAndArgc() throws {
        let argv = ["/fixture/home/AI Tools/codex", "--cd", "/fixture/home/AI Tools", "--config", "", "app-server"]
        var argc = Int32(argv.count)
        var bytes = withUnsafeBytes(of: &argc) { Array($0) }
        bytes += Array("/opt/bin/codex".utf8) + [0]
        while (bytes.count - MemoryLayout<Int32>.size) % 8 != 0 { bytes.append(0) }
        for argument in argv { bytes += Array(argument.utf8) + [0] }
        let truncated = bytes
        bytes += Array("PROMPT=app-server".utf8) + [0]
        XCTAssertEqual(AgentActivityMonitor.arguments(fromProcessArgs: bytes), argv)
        XCTAssertNil(AgentActivityMonitor.arguments(fromProcessArgs: Array(truncated.dropLast())), "unterminated argv")
        XCTAssertEqual(try XCTUnwrap(AgentActivityMonitor.kernelArguments(getpid())), CommandLine.arguments,
                       "exercise KERN_PROCARGS2 against this real test process")
    }

    func testAnUnreadableKernelArgvReachesTheChartAsUnknown() {
        let monitor = AgentActivityMonitor(
            snapshotProvider: { "59611 1 codex            codex --cd /fixture/home/AI Tools app-server" },
            executablePathResolver: { _ in nil }, argumentsResolver: { _ in nil })
        let snapshot = monitor.sample()
        XCTAssertNil(snapshot.running.count(.codex))
        var inputs = BrainBarDesignFixture.inputs(.calm)
        inputs.agents = NDStore.agents(snapshot, nil)
        let state = NDPipelineState.make(inputs)
        XCTAssertNil(NDPipelineChart.agentCount(.codex, state: state))
        XCTAssertEqual(NDPipelineChart.providerLabel(.codex, state: state), "Codex, running agents not known")
    }

    func testASymlinkOrDirectoryAtARolloutPathIsUnknown() throws {
        let target = try rollout(105, lines: [event("task_started")])
        let link = home.appendingPathComponent("rollout-link.jsonl").path
        try FileManager.default.createSymbolicLink(atPath: link, withDestinationPath: target)
        XCTAssertNil(tracker([1: [link]]).runningThreads(hosts: [1]))
        let directory = home.appendingPathComponent("rollout-directory.jsonl")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        XCTAssertNil(tracker([1: [directory.path]]).runningThreads(hosts: [1]))
    }

    // MARK: Claude review round 1

    func testLargeToolOutputDoesNotEraseARunningTurn() throws {
        for (n, type) in [(111, "response_item"), (112, "custom_tool_call_output")] {
            let large = #"{"timestamp":"2026-11-28T00:00:00Z","ordinal":2,"type":"\#(type)","payload":{"type":"custom_tool_call_output","output":"\#(String(repeating: "x", count: 56 * (1 << 20) / 10))"}}"#
            let path = try rollout(n, lines: [event("task_started")])
            let t = tracker([1: [path]])
            XCTAssertEqual(t.runningThreads(hosts: [1]), 1)
            try append(path, large + "\n")
            XCTAssertEqual(t.runningThreads(hosts: [1]), 1, "a bounded discarded non-marker leaves the turn running")
            try append(path, event("task_complete") + "\n")
            XCTAssertEqual(t.runningThreads(hosts: [1]), 0)
        }
    }

    func testGoneAndOtherUserCodexRowsDoNotMakeOurCountUnknown() {
        for status in [AgentActivityMonitor.ProcessStatus.gone, .otherUser] {
            let snapshot = AgentActivityMonitor.parse("59611 1 codex            codex app-server",
                executablePath: { _ in "/fixture/bin/codex" }, argv: { _ in nil }, processStatus: { _ in status })
            XCTAssertEqual(snapshot.running.count(.codex), 0)
        }
        let gonePath = AgentActivityMonitor.parse("59611 1 codex            codex app-server", argv: { _ in nil },
                                                    processStatus: { _ in .gone })
        XCTAssertEqual(gonePath.running.count(.codex), 0, "a proven-gone process skips the stale row even without a path")
    }

    func testAFailedArgvProbePreservesTheTrackersCachedPartialRecord() throws {
        let path = try rollout(113, lines: [event("task_started")])
        try append(path, "{partial")
        let t = tracker([1: [path]])
        XCTAssertEqual(t.runningThreads(hosts: [1]), 1)
        let monitor = AgentActivityMonitor(
            snapshotProvider: { "59611 1 codex            codex app-server" },
            executablePathResolver: { _ in "/fixture/bin/codex" }, argumentsResolver: { _ in nil },
            hostedCodexThreads: { t.runningThreads(hosts: $0) })
        XCTAssertNil(monitor.sample().running.count(.codex))
        XCTAssertEqual(t.trackedFiles, 1, "an incomplete host census must not evict unreadable hosts")
    }

    func testALargeReadTailDoesNotConsultOrCopyThePreviousChunk() {
        var previousReads = 0
        func previous() -> Data { previousReads += 1; return Data(repeating: 1, count: 256) }
        let data = Data(repeating: 2, count: 1 << 20)
        XCTAssertEqual(AgentTranscriptActivity.CodexTurnTracker.consumedTail(data: data, previous: previous()), data.suffix(256))
        XCTAssertEqual(previousReads, 0, "a full new tail is independent of previous data")
        let small = Data([3])
        XCTAssertEqual(AgentTranscriptActivity.CodexTurnTracker.consumedTail(data: small, previous: previous()),
                       Data(repeating: 1, count: 255) + small)
        XCTAssertEqual(previousReads, 1)
    }

    func testAnEmptyArgvZeroIsNotPaddingOrAnEnvironmentArgument() {
        let argv = ["", "app-server", ""]
        var argc = Int32(argv.count)
        var bytes = withUnsafeBytes(of: &argc) { Array($0) }
        bytes += Array("/fixture/bin/codex".utf8) + [0]
        // XNU aligns the string area (which starts after argc) to the 64-bit process's pointer width.
        while (bytes.count - MemoryLayout<Int32>.size) % 8 != 0 { bytes.append(0) }
        for argument in argv { bytes += Array(argument.utf8) + [0] }
        bytes += Array("ENVIRONMENT=not-an-argument".utf8) + [0]
        XCTAssertEqual(AgentActivityMonitor.arguments(fromProcessArgs: bytes), argv)
        let emptyNameHost = AgentActivityMonitor.parse("59611 1 codex            app-server",
            executablePath: { _ in "/fixture/bin/codex" }, argv: { _ in ["", "app-server"] })
        XCTAssertEqual(emptyNameHost.running.codexHosts, [59611])
    }

    func testDiscardedPrefixesSeparateEventMarkersFromToolOutput() {
        typealias Tracker = AgentTranscriptActivity.CodexTurnTracker
        for marker in ["task_started", "task_complete", "turn_aborted"] {
            XCTAssertTrue(Tracker.discardedRecordAffectsTurn(Data(event(marker, message: "incomplete").utf8)))
        }
        XCTAssertFalse(Tracker.discardedRecordAffectsTurn(Data(event("token_count").utf8)))
        let output = #"{"timestamp":"2026-11-28T00:00:00Z", "ordinal":3, "type" : "response_item", "payload":{"type":"task_complete","text":""#
        XCTAssertFalse(Tracker.discardedRecordAffectsTurn(Data(output.utf8) + Data([0xf0])),
                       "a UTF-8 character split by the prefix cap is irrelevant to the known header")
        XCTAssertTrue(Tracker.discardedRecordAffectsTurn(Data("{unclassifiable".utf8)))
        XCTAssertTrue(Tracker.discardedRecordAffectsTurn(Data(#"{"type":"event_msg","payload":{"text":"type comes later""#.utf8)))
    }

    func testOversizedMarkersAndSessionRecordsStayUnknown() throws {
        let big = String(repeating: "z", count: 3 << 20)
        let path = try rollout(114, lines: [event("task_started"), event("task_complete", message: big)])
        let t = tracker([1: [path]])
        XCTAssertNil(t.runningThreads(hosts: [1]), "an oversized turn marker still needs a later valid marker")
        try append(path, event("task_complete") + "\n")
        XCTAssertEqual(t.runningThreads(hosts: [1]), 0)
        let metaPath = try rollout(115, lines: [])
        let oversizedMeta = #"{"timestamp":"2026-11-28T00:00:00Z","ordinal":0,"type":"session_meta","payload":{"originator":"T3 Code","padding":"\#(big)"}}"#
        try (oversizedMeta + "\n" + event("task_started") + "\n").write(toFile: metaPath, atomically: false, encoding: .utf8)
        XCTAssertNil(tracker([1: [metaPath]]).runningThreads(hosts: [1]), "unreadable session kind must never be guessed")
    }

    func testKernelProcessStatusDistinguishesThisUserAndAnAbsentPid() {
        XCTAssertEqual(AgentActivityMonitor.kernelProcessStatus(getpid()), .sameUser)
        XCTAssertEqual(AgentActivityMonitor.kernelProcessStatus(Int32.max), .gone)
        let unknownOwner = AgentActivityMonitor.parse("59611 1 codex            codex app-server",
            executablePath: { _ in "/fixture/bin/codex" }, argv: { _ in nil }, processStatus: { _ in .unknown })
        XCTAssertNil(unknownOwner.running.count(.codex), "an unmeasurable owner is not silently discarded")
    }

    // MARK: Claude review round 2

    func testOversizedOutputWithOrdinalPreservesTheRunningTurn() throws {
        let output = #"{"timestamp":"2026-11-28T00:00:00Z","ordinal":42,"type":"response_item","payload":{"type":"custom_tool_call_output","output":"\#(String(repeating: "x", count: 56 * (1 << 20) / 10))"}}"#
        let path = try rollout(121, lines: [event("task_started")])
        let t = tracker([1: [path]])
        XCTAssertEqual(t.runningThreads(hosts: [1]), 1)
        try append(path, output + "\n")
        XCTAssertEqual(t.runningThreads(hosts: [1]), 1, "ordinal before type is the production record shape")
    }

    func testKernelProcessStatusReadsTheRootOwnedLaunchd() {
        XCTAssertEqual(AgentActivityMonitor.kernelProcessStatus(1), getuid() == 0 ? .sameUser : .otherUser,
                       "the short BSD probe can read launchd's UID across users")
    }

    func testScalarMetadataBeforeTypeIsNotConfusedWithPayloadTypes() {
        typealias Tracker = AgentTranscriptActivity.CodexTurnTracker
        let output = #"{"ordinal":-7,"note":"escaped \"quote\" and { brace","timestamp":"2026-11-28T00:00:00Z","type":"response_item","payload":{"type":"task_started""#
        XCTAssertFalse(Tracker.discardedRecordAffectsTurn(Data(output.utf8)))
        XCTAssertTrue(Tracker.discardedRecordAffectsTurn(Data(event("task_complete", message: "partial").utf8)))
        XCTAssertFalse(Tracker.discardedRecordAffectsTurn(Data(event("token_count").utf8)))
        let unreadableName = AgentActivityMonitor.parse("59611 1 codex            /fixture/AI Tools/codex app-server",
            argv: { _ in nil }, processStatus: { _ in .sameUser })
        XCTAssertNil(unreadableName.running.count(.codex), "a live Codex row stays unknown even when ps cannot delimit its path")
    }

}
