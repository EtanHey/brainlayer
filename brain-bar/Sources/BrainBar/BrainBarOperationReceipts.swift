import Foundation

/// MCP receipts describe calls served by BrainBarDaemon. The small sidecar lets
/// the separate BrainBar UI process read them without touching the database.
/// Watcher ingestion and later deferred-store replay are outside this receipt.
struct BrainBarOperationReceipt: Codable, Equatable, Sendable {
    enum Kind: String, Codable, Sendable { case search, ingest }

    let kind: Kind
    let durationMillis: Int
    let count: Int?
    let failed: Bool
    let recordedAt: Date

    init(kind: Kind, durationMillis: Int, count: Int?, failed: Bool = false, recordedAt: Date = Date()) {
        self.kind = kind
        self.durationMillis = durationMillis
        self.count = count
        self.failed = failed
        self.recordedAt = recordedAt
    }

    var value: String { value(now: Date()) }

    func value(now: Date) -> String {
        let duration = durationMillis < 1_000
            ? "\(durationMillis) ms"
            : String(format: "%.1f s", Double(durationMillis) / 1_000)
        let ageSeconds = max(0, Int(now.timeIntervalSince(recordedAt)))
        let age: String
        if ageSeconds < 60 { age = "just now" }
        else if ageSeconds < 3_600 { age = "\(ageSeconds / 60) min ago" }
        else if ageSeconds < 86_400 { age = "\(ageSeconds / 3_600) h ago" }
        else { age = "\(ageSeconds / 86_400) d ago" }
        let suffix = " · \(age)"
        if failed {
            return "\(duration) · failed · \(kind == .search ? "results" : "chunks") unavailable\(suffix)"
        }
        guard let count else {
            return "\(duration) · \(kind == .search ? "results" : "chunks") unavailable\(suffix)"
        }
        let unit = kind == .search ? (count == 1 ? "result" : "results") : (count == 1 ? "chunk" : "chunks")
        return "\(duration) · \(count) \(unit)\(suffix)"
    }

    /// Known providers emit this header; unknown shapes stay unavailable.
    static func searchCount(in text: String) -> Int? {
        let lines = text.split(separator: "\n")
        for (index, line) in lines.enumerated() {
            let value = String(line)
            if value.hasPrefix("## \""), index + 1 < lines.count,
               TextFormatter.compactSearchLegends.contains(String(lines[index + 1])),
               let range = value.range(of: "\" ", options: .backwards) {
                let counts = value[range.upperBound...].split(separator: "/", omittingEmptySubsequences: false)
                if counts.count == 2, let shown = Int(counts[0]), let total = Int(counts[1]),
                   shown >= 0, total >= shown { return shown }
            }
            if value.hasPrefix("## Search results for "),
               let range = value.range(of: " - ", options: .backwards),
               let count = Int(value[range.upperBound...].split(separator: " ").first ?? "") {
                return count
            }
        }
        return nil
    }
}

final class BrainBarOperationReceipts: @unchecked Sendable {
    static let shared = BrainBarOperationReceipts(url: fileURL(dbPath: BrainBarServer.defaultDBPath()))
    static let changed = Notification.Name("BrainBarOperationReceipts.changed")

    static func fileURL(dbPath: String) -> URL {
        URL(fileURLWithPath: dbPath).deletingLastPathComponent().appendingPathComponent("operation-receipts.json")
    }

    private struct Snapshot: Codable, Sendable {
        let search: BrainBarOperationReceipt?
        let ingest: BrainBarOperationReceipt?
        // Optional for sidecars written before call counting shipped. No query text.
        let agentReads: [String: Int]?
    }

    private let lock = NSLock()
    private let writeQueue = DispatchQueue(label: "com.brainlayer.brainbar.operation-receipts", qos: .utility)
    private let url: URL?
    private let persist: @Sendable (Data, URL) throws -> Void
    private var latestSearch: BrainBarOperationReceipt?
    private var latestIngest: BrainBarOperationReceipt?
    private var agentReads: [String: Int]?
    private var pendingSnapshot: Snapshot?
    private var writeScheduled = false
    private var loggedReadFailure = false

    init(url: URL? = nil, persist: @escaping @Sendable (Data, URL) throws -> Void = {
        try $0.write(to: $1, options: .atomic)
    }) {
        self.url = url
        self.persist = persist
        reload()
    }

    var isPersistent: Bool { url != nil }

    var search: BrainBarOperationReceipt? {
        lock.lock(); defer { lock.unlock() }
        return latestSearch
    }

    var ingest: BrainBarOperationReceipt? {
        lock.lock(); defer { lock.unlock() }
        return latestIngest
    }

    func record(_ receipt: BrainBarOperationReceipt) {
        lock.lock()
        switch receipt.kind {
        case .search: latestSearch = receipt
        case .ingest: latestIngest = receipt
        }
        var shouldSchedule = false
        if url != nil, receipt.kind != .search {
            pendingSnapshot = Snapshot(search: latestSearch, ingest: latestIngest, agentReads: agentReads)
            if !writeScheduled { writeScheduled = true; shouldSchedule = true }
        }
        lock.unlock()
        // Persist a search receipt together with its call count in one queued snapshot.
        if receipt.kind == .search { recordAgentRead(toolName: "brain_search", at: receipt.recordedAt) }
        if shouldSchedule { writeQueue.async { [weak self] in self?.drainWrites() } }
        NotificationCenter.default.post(name: Self.changed, object: nil)
    }

    /// Count accepted search/recall calls, including failures, independently of result counts.
    /// Both aggregation and atomic persistence run on the utility queue, never on the MCP path.
    func recordAgentRead(toolName: String, at date: Date = Date()) {
        guard toolName == "brain_search" || toolName == "brain_recall" else { return }
        writeQueue.async { [self] in
            lock.lock()
            let minute = Int(date.timeIntervalSince1970 / 60)
            var counts = agentReads ?? [:]
            let newest = max(minute, counts.keys.compactMap(Int.init).max() ?? minute)
            // 48 hours plus the current minute: bounded even under sustained traffic.
            counts = counts.filter { (Int($0.key) ?? Int.min) >= newest - 2_880 }
            if minute >= newest - 2_880 { counts[String(minute), default: 0] += 1 }
            agentReads = counts
            if url != nil {
                pendingSnapshot = Snapshot(search: latestSearch, ingest: latestIngest, agentReads: counts)
                writeScheduled = true
            }
            lock.unlock()
            drainWrites()
        }
    }

    /// Oldest first, on a fixed minute grid; nil means the producer has not supplied counts.
    func agentReadBuckets(minutes: Int, count: Int, end: Date) -> [Int]? {
        agentReadActivity(minutes: minutes, count: count, end: end)?.buckets
    }

    struct AgentReadActivity: Sendable {
        let buckets: [Int]
        let peakPerMinute: Int
    }

    func agentReadActivity(minutes: Int, count: Int, end: Date) -> AgentReadActivity? {
        guard minutes > 0, count > 0 else { return nil }
        lock.lock(); let counts = agentReads; lock.unlock()
        guard let counts else { return nil }
        let width = Double(minutes * 60) / Double(count)
        let start = end.timeIntervalSince1970 - Double(minutes * 60)
        var buckets = Array(repeating: 0, count: count)
        var peak = 0
        for (key, value) in counts {
            guard let minute = Int(key), value >= 0 else { continue }
            let offset = Double(minute * 60) - start
            if offset >= 0, offset < Double(minutes * 60) {
                buckets[min(count - 1, Int(offset / width))] += value
                peak = max(peak, value)
            }
        }
        return AgentReadActivity(buckets: buckets, peakPerMinute: peak)
    }

    private func drainWrites() {
        guard let url else { return }
        while true {
            lock.lock()
            guard let snapshot = pendingSnapshot else {
                writeScheduled = false
                lock.unlock()
                return
            }
            pendingSnapshot = nil
            lock.unlock()
            do {
                try persist(JSONEncoder().encode(snapshot), url)
            } catch {
                NSLog("[BrainBar] Could not write operation receipt: %@", String(describing: error))
            }
        }
    }

    func waitForWritesForTesting() { writeQueue.sync {} }

    func reload() {
        guard let url, FileManager.default.fileExists(atPath: url.path) else { return }
        do {
            let snapshot = try JSONDecoder().decode(Snapshot.self, from: Data(contentsOf: url))
            lock.lock()
            latestSearch = snapshot.search
            latestIngest = snapshot.ingest
            agentReads = snapshot.agentReads
            loggedReadFailure = false
            lock.unlock()
        } catch {
            lock.lock()
            let shouldLog = !loggedReadFailure
            loggedReadFailure = true
            lock.unlock()
            if shouldLog { NSLog("[BrainBar] Could not read operation receipt: %@", String(describing: error)) }
        }
    }
}
