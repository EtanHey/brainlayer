import XCTest
@testable import BrainBar

/// Etan, 2026-11-08: Details read "8 sessions · 8 Claude · 0 Codex · 0 Gemini · 0 Cursor" while
/// 11 Codex threads were writing. The row counted `ps` sessions only, and every T3 Code / Codex
/// app thread runs inside one shared `codex app-server`, which is rightly not a session. A
/// session's transcript is its per-session trace, so the row now also counts sessions with a
/// transcript written in the last 10 minutes, per CLI, and shows the larger of the two.
final class AgentTranscriptSessionTests: XCTestCase {
    private var home: URL!
    private let now = Date(timeIntervalSince1970: 1_794_200_000)

    override func setUpWithError() throws {
        home = FileManager.default.temporaryDirectory.appendingPathComponent("agent-transcripts-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: home, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: home)
    }

    private func transcript(_ path: String, ageSeconds: TimeInterval = 60, firstLine: String = #"{"type":"session_meta","payload":{"source":"vscode"}}"#) throws {
        let url = home.appendingPathComponent(path)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data((firstLine + "\n").utf8).write(to: url)
        try FileManager.default.setAttributes([.modificationDate: now.addingTimeInterval(-ageSeconds)], ofItemAtPath: url.path)
    }

    /// All four agent classes, each with sessions that count and transcripts that must not.
    private func writeFourAgentHome() throws {
        // Claude: one fresh top-level session; a stale one; a fresh sub-agent of the fresh one.
        try transcript(".claude/projects/-Users-dev-repo/11111111-aaaa.jsonl")
        try transcript(".claude/projects/-Users-dev-repo/22222222-bbbb.jsonl", ageSeconds: 3600)
        try transcript(".claude/projects/-Users-dev-repo/11111111-aaaa/subagents/agent-a1.jsonl")
        // Codex: a thread started on an older day that is still writing, one started today,
        // a spawned sub-agent thread, and a stale thread.
        try transcript(".codex/sessions/2026/10/22/rollout-2026-10-22T09-00-00-old.jsonl")
        try transcript(".codex/sessions/2026/11/08/rollout-2026-11-08T08-00-00-new.jsonl", ageSeconds: 5)
        try transcript(
            ".codex/sessions/2026/11/08/rollout-2026-11-08T08-01-00-sub.jsonl",
            firstLine: #"{"type":"session_meta","payload":{"id":"x","source":{"subagent":{"thread_spawn":{"depth":1}}}}}"#
        )
        try transcript(".codex/sessions/2026/11/07/rollout-2026-11-07T08-00-00-stale.jsonl", ageSeconds: 7200)
        // Gemini CLI session.
        try transcript(".gemini/sessions/2026-11-08/session-g1.jsonl")
        // Cursor: an agent-transcripts session, a sessions-root session, and an unrelated project file.
        try transcript(".cursor/projects/Users-dev-repo/agent-transcripts/c1/c1.jsonl")
        try transcript(".cursor/sessions/c2.jsonl")
        try transcript(".cursor/projects/Users-dev-repo/terminals/notes.jsonl")
    }

    /// The live shape: one Claude CLI, and T3 Code's Codex app-server hosting every Codex thread.
    private static let psRows = [
        AgentSessionCountTests.psRow(200, 1, "2.1.319", "/Users/dev/.local/bin/claude --output-format stream-json --verbose --input-format stream-json --model=claude-opus-5-5"),
        AgentSessionCountTests.psRow(300, 1, "codex", "/Users/dev/.local/bin/codex app-server"),
        AgentSessionCountTests.psRow(301, 300, "node", "/Users/dev/.cache/codex-runtimes/node/bin/node /Users/dev/.codex/plugins/cache/server.mjs --stdio"),
    ].joined(separator: "\n")

    func test_transcript_counter_counts_each_agent_class_and_skips_subagents_and_stale_files() throws {
        try writeFourAgentHome()

        let counts = AgentTranscriptActivity.count(home: home, now: now)

        XCTAssertEqual(counts, [.claude: 1, .codex: 2, .gemini: 1, .cursor: 2])
    }

    func test_monitor_shows_app_server_codex_threads_and_every_agent_class() throws {
        try writeFourAgentHome()
        let home = self.home!
        let now = self.now
        let monitor = AgentActivityMonitor(
            snapshotProvider: { Self.psRows },
            executablePathResolver: { _ in nil },
            transcriptSessions: { AgentTranscriptActivity.count(home: home, now: now) }
        )

        let activity = monitor.sample()

        XCTAssertEqual(activity.runtimeRowText, "6 sessions · 1 Claude · 2 Codex · 1 Gemini · 2 Cursor")
    }

    /// Codex review round 1: transcript FILES were counted, not sessions. A Claude session
    /// resumed in another worktree leaves the same `<session>.jsonl` under both project dirs.
    func test_one_session_in_two_places_counts_once_and_two_sessions_count_twice() throws {
        let claude = "e3b0c442-98fc-4c14-9afb-f4c8996fb924"
        try transcript(".claude/projects/-Users-dev-repo/\(claude).jsonl")
        try transcript(".claude/projects/-Users-dev-repo--worktrees-wt/\(claude).jsonl", ageSeconds: 30)
        // A copied Codex rollout: same thread UUID, in two date folders.
        let codex = "01a1ba31-942c-7e73-b81e-6c60bc949a77"
        try transcript(".codex/sessions/2026/11/08/rollout-2026-11-08T10-06-13-\(codex).jsonl")
        try transcript(".codex/sessions/2026/11/07/rollout-2026-11-08T10-06-13-\(codex).jsonl", ageSeconds: 30)

        XCTAssertEqual(AgentTranscriptActivity.count(home: home, now: now)[.claude], 1)
        XCTAssertEqual(AgentTranscriptActivity.count(home: home, now: now)[.codex], 1)

        try transcript(".claude/projects/-Users-dev-repo/9f86d081-884c-4d63-9a0b-1a2c3d4e5f60.jsonl")
        try transcript(".codex/sessions/2026/11/08/rollout-2026-11-08T10-06-20-01a1ba31-ae14-7953-ab0d-71cc421279a3.jsonl")

        XCTAssertEqual(AgentTranscriptActivity.count(home: home, now: now)[.claude], 2)
        XCTAssertEqual(AgentTranscriptActivity.count(home: home, now: now)[.codex], 2)
    }

    func test_session_id_is_the_codex_thread_uuid_or_the_file_stem() {
        XCTAssertEqual(
            AgentTranscriptActivity.sessionID(URL(fileURLWithPath: "/s/2026/11/08/rollout-2026-11-08T10-06-13-01a1ba31-942c-7e73-b81e-6c60bc949a77.jsonl")),
            "01A1BA31-942C-7E73-B81E-6C60BC949A77"
        )
        XCTAssertEqual(AgentTranscriptActivity.sessionID(URL(fileURLWithPath: "/p/repo/abc-123.jsonl")), "abc-123")
        XCTAssertEqual(AgentTranscriptActivity.sessionID(URL(fileURLWithPath: "/s/rollout-short.jsonl")), "rollout-short")
    }

    /// The real on-disk shape (`~/.codex/sessions`, names and mtimes only, 2026-11-08): a
    /// sub-agent rollout is a flat `rollout-<start>-<uuidv7>.jsonl` in the same day folder as
    /// its parent, so no path rule can drop it. Only the `session_meta` first line can, and the
    /// marker may sit past any fixed prefix when earlier fields are long.
    func test_codex_subagent_with_the_real_on_disk_shape_is_excluded_by_its_header_only() throws {
        let day = ".codex/sessions/2026/11/08/"
        let parent = day + "rollout-2026-11-08T10-06-13-01a1ba31-942c-7e73-b81e-6c60bc949a77.jsonl"
        let subagent = day + "rollout-2026-11-08T10-09-07-01a1ba34-39be-7f60-9da9-ac4a475206f6.jsonl"
        let longSubagent = day + "rollout-2026-11-08T10-10-19-01a1ba35-52b1-71b0-bbc0-f31f8f18af79.jsonl"
        try transcript(parent, firstLine: #"{"type":"session_meta","payload":{"id":"01a1ba31-942c-7e73-b81e-6c60bc949a77","originator":"T3 Code","source":"vscode"}}"#)
        try transcript(subagent, firstLine: #"{"type":"session_meta","payload":{"id":"01a1ba34-39be-7f60-9da9-ac4a475206f6","originator":"T3 Code","source":{"subagent":{"thread_spawn":{"parent_thread_id":"01a1ba31-942c-7e73-b81e-6c60bc949a77","depth":1}}}}}"#)
        let padding = String(repeating: "x", count: 20_000)
        try transcript(longSubagent, firstLine: #"{"type":"session_meta","payload":{"id":"01a1ba35-52b1-71b0-bbc0-f31f8f18af79","instructions":""# + padding + #"","source":{"subagent":{"thread_spawn":{"depth":1}}}}}"#)

        XCTAssertEqual(AgentTranscriptActivity.count(home: home, now: now)[.codex], 1)

        // Same name, same folder, no marker: it counts. The path played no part.
        try transcript(subagent, firstLine: #"{"type":"session_meta","payload":{"source":"cli"}}"#)
        XCTAssertEqual(AgentTranscriptActivity.count(home: home, now: now)[.codex], 2)
    }

    func test_merge_keeps_the_larger_lower_bound_per_cli() {
        let processes = AgentActivitySnapshot(presences: [
            AgentPresence(family: .claude, count: 8), AgentPresence(family: .codex, count: 0),
            AgentPresence(family: .cursor, count: 1), AgentPresence(family: .gemini, count: 0),
        ])

        let merged = processes.merging(transcriptSessions: [.claude: 6, .codex: 11])

        XCTAssertEqual(merged.runtimeRowText, "20 sessions · 8 Claude · 11 Codex · 0 Gemini · 1 Cursor")
    }

    func test_cache_rescans_only_after_its_ttl() {
        final class Calls: @unchecked Sendable { var count = 0 }
        let calls = Calls()
        let cache = AgentTranscriptActivity.Cache(ttl: 60) { _ in calls.count += 1; return [.codex: calls.count] }

        XCTAssertEqual(cache.counts(now: now), [.codex: 1])
        XCTAssertEqual(cache.counts(now: now.addingTimeInterval(59)), [.codex: 1])
        XCTAssertEqual(cache.counts(now: now.addingTimeInterval(60)), [.codex: 2])
        XCTAssertEqual(cache.counts(now: now.addingTimeInterval(-1)), [.codex: 3], "a clock step back rescans")
    }

    func test_missing_roots_are_no_sessions_and_a_failed_ps_stays_unavailable() {
        XCTAssertEqual(AgentTranscriptActivity.count(home: home, now: now), [.claude: 0, .codex: 0, .gemini: 0, .cursor: 0])

        let monitor = AgentActivityMonitor(
            snapshotProvider: { nil },
            executablePathResolver: { _ in nil },
            transcriptSessions: { [.codex: 3] }
        )
        XCTAssertFalse(monitor.sample().isMeasured)
    }
}
