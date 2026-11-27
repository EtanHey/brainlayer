import Foundation

/// Sessions per agent CLI whose transcript was written in the last `window`.
///
/// `ps` sees one process per Claude session, but a Codex thread run by `codex app-server`
/// (T3 Code, the Codex app) has no process of its own: every thread shares the one server,
/// which is rightly not a session. Its rollout file is the only per-session trace, so the
/// Runtime row read "0 Codex" with a dozen Codex threads writing (Etan, 2026-11-08).
///
/// The roots are the watcher's own (`watcher.py` `default_watch_roots`); a source BrainLayer
/// does not ingest is not counted here either. Sub-agent transcripts are not sessions, and a
/// session is counted once however many transcript files carry it (a resumed or copied Claude
/// transcript sits under every project directory it ran in).
///
/// Antigravity (`agy`) is counted from `ps` only: its conversations
/// (`~/.gemini/antigravity-cli/conversations/*.db`) are not a wired BrainLayer source. The
/// Gemini CLI (`~/.gemini/sessions`) and both Cursor roots are scanned here.
enum AgentTranscriptActivity {
    static let window: TimeInterval = 10 * 60

    /// Codex sub-agent rollouts can NOT be told apart by path: on disk (measured 2026-11-08,
    /// 1,850 rollouts) every thread, sub-agent or not, is a flat
    /// `YYYY/MM/DD/rollout-<start>-<uuid>.jsonl` beside its parent. Only the first line's
    /// `session_meta` names the origin (`"source":{"subagent":…}`), so the whole first line is
    /// read, up to this cap. It runs only on rollouts already inside the window.
    static let codexFirstLineCap = 256 * 1024
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

        func sessions(_ urls: [URL]) -> Int { Set(urls.map(sessionID)).count }
        return [
            .claude: sessions(claude), .codex: sessions(codex),
            .cursor: sessions(cursor), .gemini: sessions(gemini),
        ]
    }

    /// The session a transcript belongs to. Claude, Cursor and Gemini name the file after the
    /// session id; a Codex rollout ends in its thread's UUID after the start timestamp.
    static func sessionID(_ url: URL) -> String {
        let stem = url.deletingPathExtension().lastPathComponent
        if stem.hasPrefix("rollout-"), stem.count > 36 {
            let tail = String(stem.suffix(36))
            if let uuid = UUID(uuidString: tail) { return uuid.uuidString }
        }
        return stem
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
        codexFirstLine(url).range(of: codexSubagentMarker) != nil
    }

    /// A rollout's first line (its `session_meta`), up to `codexFirstLineCap`.
    private static func codexFirstLine(_ url: URL) -> Data {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return Data() }
        defer { try? handle.close() }
        var line = Data()
        while line.count < codexFirstLineCap,
              let chunk = try? handle.read(upToCount: min(16 * 1024, codexFirstLineCap - line.count)),
              !chunk.isEmpty {
            if let newline = chunk.firstIndex(of: UInt8(ascii: "\n")) {
                line.append(chunk[chunk.startIndex..<newline])
                break
            }
            line.append(chunk)
        }
        return line
    }

    // MARK: running hosted Codex threads (#1899 R1)

    /// Originators whose sessions have a process of their own, counted from `ps`: the interactive CLI and `codex exec`.
    /// Every other originator (T3 Code, the Codex app, the SDK) is a thread hosted by an app server.
    static let codexOwnProcessOriginators: Set<String> = ["codex_cli_rs", "codex_exec"]
    /// How much of a rollout's end is read for its last turn marker.
    static let codexTailCap = 1 << 20
    private static let codexTurnMarkers: Set<String> = ["task_started", "task_complete", "turn_aborted"]

    /// Hosted Codex threads that are mid-turn now: a top-level rollout written in the last `window`, from a hosted
    /// originator, whose last turn marker is `task_started`. A finished turn (`task_complete`, `turn_aborted`) is not
    /// running however recently it was written. A thread silent for longer than `window` mid-turn is missed (a lower bound).
    static func runningHostedCodexThreads(
        home: URL,
        now: Date,
        window: TimeInterval = window,
        fileManager: FileManager = .default
    ) -> Int {
        let rollouts = recentTranscripts(under: home.appendingPathComponent(".codex/sessions"), maxDepth: 4,
                                         cutoff: now.addingTimeInterval(-window), fileManager: fileManager)
        var threads: Set<String> = []
        for url in rollouts {
            let first = codexFirstLine(url)
            guard first.range(of: codexSubagentMarker) == nil else { continue }
            if let originator = codexOriginator(first), codexOwnProcessOriginators.contains(originator) { continue }
            if codexTurnInProgress(url) { threads.insert(sessionID(url)) }
        }
        return threads.count
    }

    static func codexOriginator(_ firstLine: Data) -> String? {
        guard let object = try? JSONSerialization.jsonObject(with: firstLine) as? [String: Any],
              let payload = object["payload"] as? [String: Any] else { return nil }
        return payload["originator"] as? String
    }

    /// The rollout's last turn marker is `task_started`. Read from the end, at most `codexTailCap`: a turn's completion is
    /// its last record, so a tail longer than the cap with no marker in it is the middle of a started turn.
    static func codexTurnInProgress(_ url: URL) -> Bool {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return false }
        defer { try? handle.close() }
        guard let size = try? handle.seekToEnd() else { return false }
        let start = size > UInt64(codexTailCap) ? size - UInt64(codexTailCap) : 0
        guard (try? handle.seek(toOffset: start)) != nil, let tail = try? handle.readToEnd() else { return false }
        for line in tail.split(separator: UInt8(ascii: "\n")).reversed() {
            guard codexTurnMarkers.contains(where: { line.range(of: Data($0.utf8)) != nil }),
                  let object = try? JSONSerialization.jsonObject(with: Data(line)) as? [String: Any],
                  object["type"] as? String == "event_msg",
                  let payload = object["payload"] as? [String: Any],
                  let type = payload["type"] as? String, codexTurnMarkers.contains(type) else { continue }
            return type == "task_started"
        }
        return start > 0
    }
}
