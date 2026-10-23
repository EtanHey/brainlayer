import Foundation
import Darwin

/// Content-free, fail-open lifecycle observation; never runs sink I/O on the writer.
enum DeferredStoreTelemetry {
    struct Origin: Codable, Sendable { let epoch: Double; let reason: String }
    private struct Event: Codable, Sendable {
        let event: String
        let queue_id: String
        let chunk_id: String
        let reason: String
        let recorded_at: String
        let queued_at_epoch: Double?
        let queue_to_commit_ms: Double?
    }
    private struct Totals: Codable {
        var count = 0
        var latency_count = 0
        var latency_sum_ms = 0.0
        var latency_max_ms = 0.0
        var histogram: [Int: Int] = [:]
    }
    private static let queue = DispatchQueue(label: "com.brainlayer.deferred-telemetry", qos: .utility)
    private static let slots = DispatchSemaphore(value: 128)

    static func record(_ event: String, dbPath: String, queueID: String, chunkID: String,
                       origin: Origin?, queuedAt: String?) {
        let env = ProcessInfo.processInfo.environment
        guard !["0", "false", "no", "off"].contains(env["BRAINLAYER_WRITER_TELEMETRY", default: "1"].lowercased()) else { return }
        let now = Date()
        let epoch = origin?.epoch ?? queuedAt.flatMap { ISO8601DateFormatter().date(from: $0)?.timeIntervalSince1970 }
        let reason = origin?.reason ?? "UNKNOWN"
        let item = Event(event: event, queue_id: queueID, chunk_id: chunkID,
                         reason: ["DB_BUSY", "DB_NOT_OPEN", "BACKUP_SNAPSHOT"].contains(reason) ? reason : "UNKNOWN",
                         recorded_at: ISO8601DateFormatter().string(from: now), queued_at_epoch: epoch,
                         queue_to_commit_ms: event == "committed" ? epoch.map { (now.timeIntervalSince1970 - $0) * 1000 } : nil)
        let path = env["BRAINLAYER_DEFERRED_TELEMETRY_PATH"].map { URL(fileURLWithPath: $0) }
            ?? URL(fileURLWithPath: dbPath).deletingLastPathComponent().appendingPathComponent("logs/deferred-stores.jsonl")
        guard slots.wait(timeout: .now()) == .success else {
            NSLog("[BrainBar] Deferred telemetry dropped: sink saturated")
            return
        }
        queue.async {
            defer { slots.signal() }
            do { try write(item, path: path) }
            catch { NSLog("[BrainBar] Deferred telemetry unavailable") }
        }
    }

    static func waitForWritesForTesting() { queue.sync {} }

    private static func write(_ item: Event, path: URL) throws {
        let fm = FileManager.default
        let directory = path.deletingLastPathComponent()
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        let fd = open(path.path + ".lock", O_CREAT | O_RDWR, 0o600)
        guard fd >= 0 else { throw CocoaError(.fileWriteUnknown) }
        defer { Darwin.close(fd) }
        guard flock(fd, LOCK_EX) == 0 else { throw CocoaError(.fileWriteUnknown) }
        defer { flock(fd, LOCK_UN) }
        let encoder = JSONEncoder()
        var data = try encoder.encode(item)
        data.append(0x0A)
        guard data.count <= 1024 * 1024 else { throw CocoaError(.fileWriteUnknown) }
        let size = (try? fm.attributesOfItem(atPath: path.path)[.size] as? Int) ?? 0
        if size + data.count > 1024 * 1024 {
            for index in stride(from: 3, through: 1, by: -1) {
                let destination = path.path + ".\(index)"
                if fm.fileExists(atPath: destination) { try fm.removeItem(atPath: destination) }
                let source = index == 1 ? path.path : path.path + ".\(index - 1)"
                if fm.fileExists(atPath: source) { try fm.moveItem(atPath: source, toPath: destination) }
            }
        }
        if !fm.fileExists(atPath: path.path) { fm.createFile(atPath: path.path, contents: nil, attributes: [.posixPermissions: 0o600]) }
        let handle = try FileHandle(forWritingTo: path)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: data)
        let prefix = path.deletingPathExtension().lastPathComponent + "-summary-"
        let summaryPath = directory.appendingPathComponent(prefix + item.recorded_at.prefix(10) + ".json")
        var summary = try fm.fileExists(atPath: summaryPath.path)
            ? JSONDecoder().decode([String: Totals].self, from: Data(contentsOf: summaryPath)) : [:]
        let key = item.event + "|" + item.reason
        var totals = summary[key] ?? Totals()
        totals.count += 1
        if let ms = item.queue_to_commit_ms, ms.isFinite, ms >= 0 {
            totals.latency_count += 1; totals.latency_sum_ms += ms
            totals.latency_max_ms = max(totals.latency_max_ms, ms)
            totals.histogram[min(31, Int(ceil(log2(max(1, ms))))), default: 0] += 1
        }
        summary[key] = totals
        try encoder.encode(summary).write(to: summaryPath, options: .atomic)
        let files = try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.lastPathComponent.hasPrefix(prefix) && $0.pathExtension == "json" }.sorted { $0.path < $1.path }
        for file in files.dropLast(8) { try fm.removeItem(at: file) }
    }
}
