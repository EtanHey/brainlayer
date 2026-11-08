import Foundation

/// Sessions per agent CLI whose transcript was written in the last `window`.
///
/// `ps` sees one process per Claude session, but a Codex thread run by `codex app-server`
/// (T3 Code, the Codex app) has no process of its own: every thread shares the one server,
/// which is rightly not a session. Its rollout file is the only per-session trace, so the
/// Runtime row read "0 Codex" with a dozen Codex threads writing (Etan, 2026-11-08).
///
/// The roots are the watcher's own (`watcher.py` `default_watch_roots`); a source BrainLayer
/// does not ingest is not counted here either. Sub-agent transcripts are not sessions.
enum AgentTranscriptActivity {
    static let window: TimeInterval = 10 * 60

    /// Codex names a spawned sub-agent thread's origin in its first line's `session_meta`,
    /// a few hundred bytes in.
    static let codexHeaderBytes = 4096
    private static let codexSubagentMarker = Data(#""source":{"subagent""#.utf8)

    static func count(
        home: URL,
        now: Date,
        window: TimeInterval = window,
        fileManager: FileManager = .default
    ) -> [AgentFamily: Int] {
        let cutoff = now.addingTimeInterval(-window)
        func recent(_ root: URL, depth: Int) -> [URL] {
            recentTranscripts(under: root, maxDepth: depth, cutoff: cutoff, fileManager: fileManager)
        }

        // `<project>/<session>.jsonl`; a session's sub-agents live below it, in `<session>/subagents/`.
        let claude = recent(home.appendingPathComponent(".claude/projects"), depth: 2)
            .filter { $0.deletingLastPathComponent().deletingLastPathComponent().lastPathComponent == "projects" }
        // `YYYY/MM/DD/rollout-*.jsonl`, dated by the thread's START: a live thread may sit in an old day.
        let codex = recent(home.appendingPathComponent(".codex/sessions"), depth: 4)
            .filter { !isCodexSubagent($0) }
        let cursorProjects = home.appendingPathComponent(".cursor/projects")
        let cursorTranscriptRoots = ((try? fileManager.contentsOfDirectory(
            at: cursorProjects, includingPropertiesForKeys: nil
        )) ?? []).map { $0.appendingPathComponent("agent-transcripts") }
        let cursor = recent(home.appendingPathComponent(".cursor/sessions"), depth: 4)
            + cursorTranscriptRoots.flatMap { recent($0, depth: 3) }
        let gemini = recent(home.appendingPathComponent(".gemini/sessions"), depth: 4)

        return [.claude: claude.count, .codex: codex.count, .cursor: cursor.count, .gemini: gemini.count]
    }

    /// `*.jsonl` files at most `maxDepth` levels below `root`, modified at or after `cutoff`,
    /// with no `subagents` directory between them and `root`. A missing root is no sessions.
    static func recentTranscripts(
        under root: URL,
        maxDepth: Int,
        cutoff: Date,
        fileManager: FileManager = .default
    ) -> [URL] {
        let keys: [URLResourceKey] = [.isDirectoryKey, .contentModificationDateKey]
        guard let enumerator = fileManager.enumerator(at: root, includingPropertiesForKeys: keys) else { return [] }
        var found: [URL] = []
        while let url = enumerator.nextObject() as? URL {
            guard let values = try? url.resourceValues(forKeys: Set(keys)) else { continue }
            if values.isDirectory == true {
                if url.lastPathComponent == "subagents" || enumerator.level >= maxDepth { enumerator.skipDescendants() }
                continue
            }
            guard url.pathExtension == "jsonl",
                  let modified = values.contentModificationDate, modified >= cutoff else { continue }
            found.append(url)
        }
        return found
    }

    /// A live scan stats every transcript file (~0.3 s for ~4,600 files measured 2026-11-08), and
    /// the monitor samples every 5 s. A 10-minute window needs no 5-second freshness.
    final class Cache: @unchecked Sendable {
        private let lock = NSLock()
        private let ttl: TimeInterval
        private let compute: @Sendable (Date) -> [AgentFamily: Int]
        private var last: (at: Date, counts: [AgentFamily: Int])?

        init(ttl: TimeInterval = 60, compute: @escaping @Sendable (Date) -> [AgentFamily: Int]) {
            self.ttl = ttl
            self.compute = compute
        }

        func counts(now: Date = Date()) -> [AgentFamily: Int] {
            lock.lock()
            defer { lock.unlock() }
            if let last, (0..<ttl).contains(now.timeIntervalSince(last.at)) { return last.counts }
            let counts = compute(now)
            last = (now, counts)
            return counts
        }
    }

    private static func isCodexSubagent(_ url: URL) -> Bool {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return false }
        defer { try? handle.close() }
        let header = (try? handle.read(upToCount: codexHeaderBytes)) ?? Data()
        return header.range(of: codexSubagentMarker) != nil
    }
}
