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

    /// Identity of the descriptor the host actually holds, captured with its pathname.
    struct HeldFile: Hashable, Sendable {
        let path: String
        let device: UInt64
        let inode: UInt64
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
    /// Anything not proven makes the whole count nil (unknown): a host whose open files cannot be listed, a held rollout
    /// that cannot be opened or read, a session record whose originator cannot be read (K5), a marker line that does not
    /// parse, or a file still being caught up (K2).
    final class CodexTurnTracker: @unchecked Sendable {
        enum Turn: Equatable { case none, started, finished, unknown }
        /// What the session record (the first line) says: not complete yet, a hosted thread, not one, or unreadable.
        enum Kind: Equatable { case pending, hosted, notHosted, unknown }

        /// K4: a file is the same file while the open fd's device, inode and birth time match and its first bytes are
        /// the ones read before. A replaced file, a reused inode or an in-place rewrite starts over.
        struct Identity: Equatable {
            var device: UInt64
            var inode: UInt64
            var born: timespec

            static func == (a: Identity, b: Identity) -> Bool {
                a.device == b.device && a.inode == b.inode && a.born.tv_sec == b.born.tv_sec && a.born.tv_nsec == b.born.tv_nsec
            }
        }

        private struct Entry {
            var identity: Identity
            var observedSize: UInt64 = 0
            var stamp: [Int] = [] // nanosecond mtime/ctime at the last sample
            var tail = Data()      // consumed bytes immediately before offset, checked even on growth
            var discardingRecord = false
            var discardedPrefix = Data() // only the first 4 KiB survives an oversized record
            var head = Data()      // the first `headLength` bytes read, compared on every sample (K4)
            var offset: UInt64 = 0 // bytes consumed: complete lines read, plus `pending`
            var pending = Data()   // a line still without its newline, carried to the next read (K2)
            var turn: Turn = .none
            var kind: Kind = .pending
            var caughtUp = false   // every byte up to the size measured this sample is consumed
        }

        private let lock = NSLock()
        private var entries: [String: Entry] = [:]
        private let openFiles: @Sendable (Int32) -> [HeldFile]?
        private let sampleBudget: Int
        /// Bytes per read.
        static let chunk = 1 << 20
        /// K2: bytes read per sample, across every held file; a file further behind resumes on the next sample.
        static let defaultSampleBudget = 8 << 20
        static let headLength = 256
        /// Retained bytes per record. Supports the >1 MiB completion fixture while bounding malformed input.
        static let recordLimit = 2 << 20
        static let discardedPrefixLimit = 4 << 10

        init(openFiles: @escaping @Sendable (Int32) -> [HeldFile]?, sampleBudget: Int = defaultSampleBudget) {
            self.openFiles = openFiles
            self.sampleBudget = sampleBudget
        }

        /// Files whose state is kept (K3: only the ones a live host held at the last sample).
        var trackedFiles: Int {
            lock.lock()
            defer { lock.unlock() }
            return entries.count
        }

        func runningThreads(hosts: [Int32]) -> Int? {
            lock.lock()
            defer { lock.unlock() }
            var held: Set<HeldFile> = []
            // K3: whatever happens below, keep state only for files a live host holds.
            defer {
                let paths = Set(held.map(\.path))
                entries = entries.filter { paths.contains($0.key) }
            }
            for pid in hosts {
                guard let files = openFiles(pid) else { return nil }
                held.formUnion(files.filter { Self.isRollout($0.path) })
            }
            var budget = sampleBudget
            var proven = true
            var running: Set<String> = []
            for file in held.sorted(by: { $0.path < $1.path }) {
                guard let entry = advance(file, budget: &budget), entry.caughtUp else {
                    proven = false  // unreadable, or still catching up: keep reading the others
                    continue
                }
                switch (entry.kind, entry.turn) {
                case (.pending, _), (.notHosted, _): continue
                case (.unknown, _), (.hosted, .unknown): proven = false
                case (.hosted, .started): running.insert(AgentTranscriptActivity.sessionID(URL(fileURLWithPath: file.path)))
                case (.hosted, .none), (.hosted, .finished): continue
                }
            }
            return proven ? running.count : nil
        }

        static func isRollout(_ path: String) -> Bool {
            let name = (path as NSString).lastPathComponent
            return name.hasPrefix("rollout-") && name.hasSuffix(".jsonl")
        }

        /// Reads, within `budget`, what the file gained since the last sample (from the start for a new or different
        /// file); nil when it cannot be opened, measured or read.
        private func advance(_ file: HeldFile, budget: inout Int) -> Entry? {
            let path = file.path
            // K4: open first, then measure the open file, so the bytes read are the bytes measured.
            let fd = open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK | O_NOFOLLOW)
            guard fd >= 0 else { return nil }
            let handle = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
            var st = stat()
            guard fstat(fd, &st) == 0, st.st_mode & S_IFMT == S_IFREG,
                  UInt64(UInt32(bitPattern: st.st_dev)) == file.device, UInt64(st.st_ino) == file.inode else { return nil }
            let size = UInt64(max(0, st.st_size))
            let identity = Identity(device: UInt64(UInt32(bitPattern: st.st_dev)), inode: UInt64(st.st_ino), born: st.st_birthtimespec)
            let stamp = [st.st_mtimespec.tv_sec, st.st_mtimespec.tv_nsec, st.st_ctimespec.tv_sec, st.st_ctimespec.tv_nsec]
            var next = Entry(identity: identity)
            if let kept = entries[path], kept.identity == identity, kept.offset <= size,
               !(size <= kept.observedSize && stamp != kept.stamp) {
                if kept.head.isEmpty { next = kept } else {
                    guard (try? handle.seek(toOffset: 0)) != nil, let head = try? handle.read(upToCount: kept.head.count) else { return nil }
                    guard (try? handle.seek(toOffset: kept.offset - UInt64(kept.tail.count))) != nil,
                          let tail = try? handle.read(upToCount: kept.tail.count) else { return nil }
                    if head == kept.head, tail == kept.tail { next = kept }
                }
            }
            next.observedSize = size
            next.stamp = stamp
            while next.offset < size, budget > 0 {
                let want = Int(min(UInt64(Self.chunk), size - next.offset, UInt64(budget)))
                guard (try? handle.seek(toOffset: next.offset)) != nil,
                      let data = try? handle.read(upToCount: want), !data.isEmpty else { return nil }
                budget -= data.count
                if next.head.count < Self.headLength, next.offset == UInt64(next.head.count) {
                    next.head.append(data.prefix(Self.headLength - next.head.count))
                }
                next.offset += UInt64(data.count)
                next.tail = Self.consumedTail(data: data, previous: next.tail)
                // Frame only the new chunk. Never search an ever-growing pending buffer again.
                var start = data.startIndex
                while start < data.endIndex {
                    let newline = data[start...].firstIndex(of: UInt8(ascii: "\n"))
                    let end = newline ?? data.endIndex
                    if !next.discardingRecord {
                        if end - start > Self.recordLimit - next.pending.count {
                            next.discardedPrefix = Data(next.pending.prefix(Self.discardedPrefixLimit))
                            next.discardedPrefix.append(data[start..<end].prefix(Self.discardedPrefixLimit - next.discardedPrefix.count))
                            next.pending.removeAll(keepingCapacity: false)
                            next.discardingRecord = true
                            if Self.discardedRecordAffectsTurn(next.discardedPrefix) { next.turn = .unknown }
                            if next.kind == .pending { next.kind = .unknown }
                        } else {
                            next.pending.append(data[start..<end])
                        }
                    }
                    guard let newline else { break }
                    if !next.discardingRecord {
                        if next.kind == .pending { next.kind = Self.kind(next.pending) }
                        else { next.turn = Self.turn(after: next.turn, line: next.pending) }
                    }
                    next.pending.removeAll(keepingCapacity: false)
                    next.discardingRecord = false
                    next.discardedPrefix.removeAll(keepingCapacity: false)
                    start = newline + 1
                }
            }
            next.caughtUp = next.offset == size
            entries[path] = next
            return next
        }

        static func consumedTail(data: Data, previous: @autoclosure () -> Data) -> Data {
            if data.count >= headLength { return Data(data.suffix(headLength)) }
            return Data((previous().suffix(headLength - data.count) + data).suffix(headLength))
        }

        // Scalar metadata (including timestamp and ordinal) can precede type in the rollout header before its (possibly enormous) payload. An
        // unfamiliar/truncated header stays unknown; nested payload type/text cannot impersonate the top-level type.
        private static let recordHeader = try! NSRegularExpression(pattern:
            #"^\s*\{\s*(?:"[a-z_]+"\s*:\s*(?:"(?:[^"\\]|\\.)*"|-?\d+)\s*,\s*)*"type"\s*:\s*"([a-z_]+)"\s*[,}]"#)
        private static let eventPayloadHeader = try! NSRegularExpression(pattern:
            #"^\s*"payload"\s*:\s*\{\s*"type"\s*:\s*"([a-z_]+)"\s*[,}]"#)

        static func discardedRecordAffectsTurn(_ prefix: Data) -> Bool {
            // The cap can split a UTF-8 character in tool output after an already recognizable header.
            let text = String(decoding: prefix, as: UTF8.self)
            let string = text as NSString
            guard let header = recordHeader.firstMatch(in: text, range: NSRange(location: 0, length: string.length)) else { return true }
            guard string.substring(with: header.range(at: 1)) == "event_msg" else { return false }
            let rest = string.substring(from: header.range.upperBound)
            guard let payload = eventPayloadHeader.firstMatch(in: rest, range: NSRange(location: 0, length: (rest as NSString).length)) else { return true }
            return ["task_started", "task_complete", "turn_aborted"].contains((rest as NSString).substring(with: payload.range(at: 1)))
        }

        /// K5: the session record decides the kind; one whose originator cannot be read is unknown, never hosted.
        static func kind(_ line: Data.SubSequence) -> Kind {
            if line.range(of: AgentTranscriptActivity.codexSubagentMarker) != nil { return .notHosted }
            guard let originator = AgentTranscriptActivity.codexOriginator(Data(line)) else { return .unknown }
            return codexOwnProcessOriginators.contains(originator) ? .notHosted : .hosted
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

    /// The kernel's fd list call, `proc_pidinfo(pid, PROC_PIDLISTFDS, …)`: bytes written, or ≤ 0 on failure.
    typealias ListFDs = (_ pid: Int32, _ buffer: UnsafeMutableRawPointer?, _ bytes: Int32) -> Int32
    static func kernelListFDs(_ pid: Int32, _ buffer: UnsafeMutableRawPointer?, _ bytes: Int32) -> Int32 {
        proc_pidinfo(pid, PROC_PIDLISTFDS, 0, buffer, bytes)
    }
    static let listRetries = 4

    /// The paths and descriptor identities of files a live process holds open, from the kernel (`proc_pidinfo` / `proc_pidfdinfo`); nil when the
    /// list cannot be read whole. K1: a list that fills its buffer may be cut short, so it is read again into a buffer
    /// twice the size, at most `listRetries` times, then nil. An fd closed between the list and its read is skipped; any
    /// other failure is nil.
    static func openFilePaths(_ pid: Int32, listFDs: ListFDs = kernelListFDs) -> [HeldFile]? {
        let needed = listFDs(pid, nil, 0)
        guard needed > 0 else { return nil }
        let stride = MemoryLayout<proc_fdinfo>.stride
        var capacity = Int(needed) / stride + 32
        for _ in 0..<listRetries {
            var fds = [proc_fdinfo](repeating: proc_fdinfo(), count: capacity)
            let bytes = Int32(capacity * stride)
            let used = fds.withUnsafeMutableBytes { listFDs(pid, $0.baseAddress, bytes) }
            guard used > 0 else { return nil }
            guard used < bytes else {
                capacity *= 2
                continue
            }
            var paths: [HeldFile] = []
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
                paths.append(HeldFile(path: path,
                                      device: UInt64(info.pvip.vip_vi.vi_stat.vst_dev),
                                      inode: UInt64(info.pvip.vip_vi.vi_stat.vst_ino)))
            }
            return paths
        }
        return nil
    }
}
