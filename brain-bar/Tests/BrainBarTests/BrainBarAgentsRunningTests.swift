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

    // MARK: B1: running, not recently written

    private var home: URL!

    override func setUpWithError() throws {
        home = FileManager.default.temporaryDirectory.appendingPathComponent("w307-running-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: home, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: home)
    }

    private func event(_ type: String) -> String {
        #"{"timestamp":"2026-11-28T00:00:00Z","type":"event_msg","payload":{"type":"\#(type)"}}"#
    }

    /// A rollout under `.codex/sessions/2026/11/28`, its first line the session_meta, then `events`.
    @discardableResult
    private func rollout(_ uuid: String, originator: String = "T3 Code", subagent: Bool = false, events: [String],
                         filler: Int = 0) throws -> URL {
        let dir = home.appendingPathComponent(".codex/sessions/2026/11/28", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let source = subagent ? #""source":{"subagent":"review"},"# : ""
        var lines = [#"{"type":"session_meta","payload":{\#(source)"id":"\#(uuid)","originator":"\#(originator)"}}"#]
        lines += events.map(event)
        // A long turn: tool output after the start, past the tail the reader looks at.
        for i in 0..<filler {
            lines.append(#"{"type":"response_item","payload":{"type":"message","n":\#(i),"text":"\#(String(repeating: "x", count: 1000))"}}"#)
        }
        let url = dir.appendingPathComponent("rollout-2026-11-28T00-00-00-\(uuid).jsonl")
        try (lines.joined(separator: "\n") + "\n").write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    func testAFinishedThreadWrittenRecentlyIsNotRunning() throws {
        try rollout("00000000-0000-7000-8000-000000000071", events: ["task_started", "task_complete"])
        try rollout("00000000-0000-7000-8000-000000000072", events: ["task_started", "turn_aborted"])
        try rollout("00000000-0000-7000-8000-000000000073", events: ["task_started", "task_complete", "task_started"])
        try rollout("00000000-0000-7000-8000-000000000074", events: [])  // opened, no turn yet
        try rollout("00000000-0000-7000-8000-000000000075", subagent: true, events: ["task_started"])
        try rollout("00000000-0000-7000-8000-000000000076", originator: "codex_cli_rs", events: ["task_started"])  // has its own process
        try rollout("00000000-0000-7000-8000-000000000077", events: ["task_started"], filler: 1200)  // > 1 MiB since the start
        XCTAssertEqual(AgentTranscriptActivity.runningHostedCodexThreads(home: home, now: Date()), 2,
                       "only the mid-turn hosted threads: the restarted one and the long one")
    }

    /// The whole path: `ps` and the rollouts → the monitor's snapshot → the chart's state.
    private func chart(_ ps: String, transcripts: [AgentFamily: Int]) -> NDPipelineState {
        let home = self.home!
        let monitor = AgentActivityMonitor(snapshotProvider: { ps }, executablePathResolver: { _ in nil },
                                           transcriptSessions: { transcripts },
                                           hostedCodexThreads: { AgentTranscriptActivity.runningHostedCodexThreads(home: home, now: Date()) })
        var i = BrainBarDesignFixture.inputs(.calm)
        i.agents = NDStore.agents(monitor.sample(), nil)
        return NDPipelineState.make(i)
    }

    func testACompletedRecentTranscriptReachesTheChartAsNotRunning() throws {
        try rollout("00000000-0000-7000-8000-000000000081", events: ["task_started", "task_complete"])
        try rollout("00000000-0000-7000-8000-000000000082", events: ["task_started"])
        let ps = """
        59611 93185 codex            /opt/agents/.local/bin/codex app-server -c thread_unload_delay_secs=0
         2918     1 2.1.114          claude --resume 3679128a-f371-445f-82ba-b3946e2f20b6
        """
        // Five Claude and four Codex sessions wrote in the last 10 minutes; one Claude process and one hosted turn run.
        let state = chart(ps, transcripts: [.claude: 5, .codex: 4])
        XCTAssertEqual(state.agentsRunning[.claude], 1, "a finished Claude session is not running")
        XCTAssertEqual(state.agentsRunning[.codex], 1, "the completed thread is not running; the mid-turn one is")
        XCTAssertEqual(state.agentsRunning[.cursor], 0)
        XCTAssertEqual(NDPipelineChart.agentCount(.codex, state: state), 1)
        // The session counts (the Runtime card's) still read 10 minutes of transcripts, as before.
        XCTAssertEqual(NDStore.agents(AgentActivityMonitor(snapshotProvider: { ps }, executablePathResolver: { _ in nil },
                                                           transcriptSessions: { [.claude: 5, .codex: 4] }).sample(), nil)
                        .agents.first { $0.family == .claude }?.sessions, 5)
    }

    func testAThreadLeftMidTurnWithNoServerIsNotRunning() throws {
        try rollout("00000000-0000-7000-8000-000000000091", events: ["task_started"])
        let state = chart(" 2918     1 2.1.114          claude --resume x\n", transcripts: [:])
        XCTAssertEqual(state.agentsRunning[.codex], 0, "no app server hosts it")
    }

    func testUnreadHostedThreadsAreUnknownNotZero() {
        let ps = "59611 93185 codex            /opt/agents/.local/bin/codex app-server\n"
        let snapshot = AgentActivityMonitor(snapshotProvider: { ps }, executablePathResolver: { _ in nil }).sample()
        XCTAssertTrue(snapshot.running.codexHostAlive)
        XCTAssertNil(snapshot.running.count(.codex))
        var i = BrainBarDesignFixture.inputs(.calm)
        i.agents = NDStore.agents(snapshot, nil)
        let state = NDPipelineState.make(i)
        XCTAssertNil(state.agentsRunning[.codex])
        XCTAssertEqual(NDPipelineChart.agentTone(.codex, state: state), .neutral)
        XCTAssertEqual(NDPipelineChart.providerLabel(.codex, state: state), "Codex, running agents not known")
    }

    func testAnUnmeasuredSnapshotIsUnknownEverywhere() {
        let snapshot = AgentActivityMonitor(snapshotProvider: { nil }, executablePathResolver: { _ in nil }).sample()
        XCTAssertTrue(AgentHarness.allCases.allSatisfy { snapshot.running.count($0) == nil })
    }
}
