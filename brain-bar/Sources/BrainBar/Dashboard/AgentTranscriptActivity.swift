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
    static let codexSubagentMarker = Data(#""source":{"subagent""#.utf8)

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
    static func codexFirstLine(_ url: URL) -> Data {
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

    // MARK: running hosted Codex threads (#1899 R1, R2)

    /// Originators whose sessions have a process of their own, counted from `ps`: the interactive CLI and `codex exec`.
    /// Every other originator (T3 Code, the Codex app, the SDK) is a thread hosted by an app server.
    static let codexOwnProcessOriginators: Set<String> = ["codex_cli_rs", "codex_exec"]

    static func codexOriginator(_ firstLine: Data) -> String? {
        guard let object = try? JSONSerialization.jsonObject(with: firstLine) as? [String: Any],
              let payload = object["payload"] as? [String: Any] else { return nil }
        return payload["originator"] as? String
    }

    /// #1899 R2: hosted Codex threads that are running now, each PROVEN by its own host. The rule:
    /// a thread runs while (1) a live `codex app-server` process holds its rollout open, and (2) the rollout's last valid
    /// turn marker is `task_started` (no later `task_complete` or `turn_aborted`).
    /// (1) is ownership by a live host, read from that process's own open files. A server that died mid-turn holds
    /// nothing, so its orphaned thread is not counted while another (or a restarted) server lives. Two live servers each
    /// own the files they hold; a file held by both counts once. (2) is read incrementally per file, from the file's start,
    /// so no record is ever cut by a tail window (R3).
    /// Anything not measured makes the whole count nil (unknown): a host whose open files cannot be listed, a held rollout
    /// that cannot be opened or read, a session record that cannot be read, or a marker line that does not parse (R1).
    final class CodexTurnTracker: @unchecked Sendable {
        enum Turn: Equatable { case none, started, finished, unknown }

        private struct Entry {
            var file: UInt64   // the file's id: a replaced file starts over
            var offset: UInt64 // the end of the last complete line read
            var turn: Turn
            var hosted: Bool   // a top-level thread from a hosted originator
        }

        private let lock = NSLock()
        private var entries: [String: Entry] = [:]
        private let openFiles: @Sendable (Int32) -> [String]?
        /// Bytes read per call: the read is bounded by the size measured first, so appends during it wait for the next.
        static let chunk = 1 << 20

        init(openFiles: @escaping @Sendable (Int32) -> [String]?) {
            self.openFiles = openFiles
        }

        func runningThreads(hosts: [Int32]) -> Int? {
            lock.lock()
            defer { lock.unlock() }
            var held: Set<String> = []
            for pid in hosts {
                guard let files = openFiles(pid) else { return nil }
                held.formUnion(files.filter(Self.isRollout))
            }
            var running: Set<String> = []
            for path in held {
                guard let entry = advance(path) else { return nil }
                guard entry.hosted else { continue }
                switch entry.turn {
                case .started: running.insert(AgentTranscriptActivity.sessionID(URL(fileURLWithPath: path)))
                case .none, .finished: continue
                case .unknown: return nil
                }
            }
            entries = entries.filter { held.contains($0.key) }  // forget files no live host holds
            return running.count
        }

        static func isRollout(_ path: String) -> Bool {
            let name = (path as NSString).lastPathComponent
            return name.hasPrefix("rollout-") && name.hasSuffix(".jsonl")
        }

        /// Reads what was appended since the last call (from the start for a new or replaced file); nil when unreadable.
        private func advance(_ path: String) -> Entry? {
            guard let attributes = try? FileManager.default.attributesOfItem(atPath: path),
                  let size = (attributes[.size] as? NSNumber)?.uint64Value,
                  let file = (attributes[.systemFileNumber] as? NSNumber)?.uint64Value else { return nil }
            var entry = entries[path].flatMap { $0.file == file && $0.offset <= size ? $0 : nil }
            if entry == nil {
                guard let probe = FileHandle(forReadingAtPath: path) else { return nil }
                try? probe.close()
                let first = AgentTranscriptActivity.codexFirstLine(URL(fileURLWithPath: path))
                // Its session record is not written yet: nothing has started (asked again next time).
                if UInt64(first.count) >= size { return Entry(file: file, offset: 0, turn: .none, hosted: false) }
                let subagent = first.range(of: AgentTranscriptActivity.codexSubagentMarker) != nil
                let own = AgentTranscriptActivity.codexOriginator(first).map(codexOwnProcessOriginators.contains) ?? false
                entry = Entry(file: file, offset: 0, turn: .none, hosted: !subagent && !own)
            }
            guard var next = entry else { return nil }
            if next.offset < size {
                guard let handle = FileHandle(forReadingAtPath: path) else { return nil }
                defer { try? handle.close() }
                guard (try? handle.seek(toOffset: next.offset)) != nil else { return nil }
                var pending = Data()
                var position = next.offset
                while position < size {
                    guard let data = try? handle.read(upToCount: Int(min(UInt64(Self.chunk), size - position))),
                          !data.isEmpty else { return nil }
                    position += UInt64(data.count)
                    pending.append(data)
                    // Complete lines only; a partial last line waits for its newline.
                    guard let lastNewline = pending.lastIndex(of: UInt8(ascii: "\n")) else { continue }
                    let complete = pending[pending.startIndex..<lastNewline]
                    for line in complete.split(separator: UInt8(ascii: "\n")) { next.turn = Self.turn(after: next.turn, line: line) }
                    let consumed = UInt64(lastNewline - pending.startIndex + 1)
                    next.offset += consumed
                    pending = Data(pending[(lastNewline + 1)...])
                }
            }
            entries[path] = next
            return next
        }

        private static let markers = ["task_started", "task_complete", "turn_aborted"].map { Data("\"type\":\"\($0)\"".utf8) }

        /// A line that carries a turn marker moves the state; one that names a marker but does not parse is unknown.
        static func turn(after current: Turn, line: Data.SubSequence) -> Turn {
            guard markers.contains(where: { line.range(of: $0) != nil }) else { return current }
            guard let object = try? JSONSerialization.jsonObject(with: Data(line)) as? [String: Any] else { return .unknown }
            guard object["type"] as? String == "event_msg", let payload = object["payload"] as? [String: Any],
                  let type = payload["type"] as? String else { return current }
            switch type {
            case "task_started": return .started
            case "task_complete", "turn_aborted": return .finished
            default: return current
            }
        }
    }

    /// The paths of the files a live process holds open, from the kernel (`proc_pidinfo` / `proc_pidfdinfo`); nil when the
    /// list cannot be read. An fd closed between the list and its read is skipped; any other failure is nil.
    static func openFilePaths(_ pid: Int32) -> [String]? {
        let needed = proc_pidinfo(pid, PROC_PIDLISTFDS, 0, nil, 0)
        guard needed > 0 else { return nil }
        let stride = MemoryLayout<proc_fdinfo>.stride
        var fds = [proc_fdinfo](repeating: proc_fdinfo(), count: Int(needed) / stride + 32)
        let used = fds.withUnsafeMutableBytes { proc_pidinfo(pid, PROC_PIDLISTFDS, 0, $0.baseAddress, Int32($0.count)) }
        guard used > 0 else { return nil }
        var paths: [String] = []
        for fd in fds.prefix(Int(used) / stride) where fd.proc_fdtype == UInt32(PROX_FDTYPE_VNODE) {
            var info = vnode_fdinfowithpath()
            let size = Int32(MemoryLayout<vnode_fdinfowithpath>.size)
            let got = proc_pidfdinfo(pid, fd.proc_fd, PROC_PIDFDVNODEPATHINFO, &info, size)
            if got != size {
                if errno == EBADF { continue }
                return nil
            }
            let path = withUnsafeBytes(of: &info.pvip.vip_path) { raw in
                String(cString: raw.bindMemory(to: CChar.self).baseAddress!)
            }
            paths.append(path)
        }
        return paths
    }
}
