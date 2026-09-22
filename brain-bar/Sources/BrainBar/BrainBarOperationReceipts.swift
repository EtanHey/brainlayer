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
        for line in text.split(separator: "\n") {
            let value = String(line)
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
    }

    private let lock = NSLock()
    private let writeQueue = DispatchQueue(label: "com.brainlayer.brainbar.operation-receipts", qos: .utility)
    private let url: URL?
    private let persist: @Sendable (Data, URL) throws -> Void
    private var latestSearch: BrainBarOperationReceipt?
    private var latestIngest: BrainBarOperationReceipt?
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
        if url != nil {
            pendingSnapshot = Snapshot(search: latestSearch, ingest: latestIngest)
            if !writeScheduled { writeScheduled = true; shouldSchedule = true }
        }
        lock.unlock()
        if shouldSchedule { writeQueue.async { [weak self] in self?.drainWrites() } }
        NotificationCenter.default.post(name: Self.changed, object: nil)
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
