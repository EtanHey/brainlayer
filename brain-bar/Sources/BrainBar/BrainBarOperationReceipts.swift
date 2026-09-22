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

    init(kind: Kind, durationMillis: Int, count: Int?, failed: Bool = false) {
        self.kind = kind
        self.durationMillis = durationMillis
        self.count = count
        self.failed = failed
    }

    var value: String {
        let duration = durationMillis < 1_000
            ? "\(durationMillis) ms"
            : String(format: "%.1f s", Double(durationMillis) / 1_000)
        if failed {
            return "\(duration) · failed · \(kind == .search ? "results" : "chunks") unavailable"
        }
        guard let count else {
            return "\(duration) · \(kind == .search ? "results" : "chunks") unavailable"
        }
        let unit = kind == .search ? (count == 1 ? "result" : "results") : (count == 1 ? "chunk" : "chunks")
        return "\(duration) · \(count) \(unit)"
    }

    /// Both known search providers place the shown count in a specific header.
    /// Unknown response shapes stay unavailable rather than inferring from text.
    static func searchCount(in text: String) -> Int? {
        for line in text.split(separator: "\n") {
            let value = String(line)
            if value.hasPrefix("## Search results for "),
               let range = value.range(of: " - ", options: .backwards),
               let count = Int(value[range.upperBound...].split(separator: " ").first ?? "") {
                return count
            }
            if value.hasPrefix("┌─ brain_search:"),
               let range = value.range(of: " ─ ", options: .backwards),
               let count = Int(value[range.upperBound...].split(separator: " ").first ?? "") {
                return count
            }
        }
        return nil
    }
}

final class BrainBarOperationReceipts: @unchecked Sendable {
    static let shared = BrainBarOperationReceipts(url:
        URL(fileURLWithPath: BrainBarServer.defaultDBPath())
            .deletingLastPathComponent().appendingPathComponent("operation-receipts.json")
    )
    static let changed = Notification.Name("BrainBarOperationReceipts.changed")

    private struct Snapshot: Codable {
        let search: BrainBarOperationReceipt?
        let ingest: BrainBarOperationReceipt?
    }

    private let lock = NSLock()
    private let url: URL?
    private var latestSearch: BrainBarOperationReceipt?
    private var latestIngest: BrainBarOperationReceipt?

    init(url: URL? = nil) {
        self.url = url
        reload()
    }

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
        let snapshot = Snapshot(search: latestSearch, ingest: latestIngest)
        if let url {
            do {
                try JSONEncoder().encode(snapshot).write(to: url, options: .atomic)
            } catch {
                NSLog("[BrainBar] Could not write operation receipt: %@", String(describing: error))
            }
        }
        lock.unlock()
        NotificationCenter.default.post(name: Self.changed, object: nil)
    }

    func reload() {
        guard let url, FileManager.default.fileExists(atPath: url.path) else { return }
        do {
            let snapshot = try JSONDecoder().decode(Snapshot.self, from: Data(contentsOf: url))
            lock.lock()
            latestSearch = snapshot.search
            latestIngest = snapshot.ingest
            lock.unlock()
        } catch {
            NSLog("[BrainBar] Could not read operation receipt: %@", String(describing: error))
        }
    }
}
