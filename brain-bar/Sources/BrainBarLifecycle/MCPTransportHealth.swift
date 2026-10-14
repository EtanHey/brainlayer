import Foundation

/// Aggregate transport evidence only. Owned by the server request queue; never retains payloads or identities.
public final class MCPTransportHealth {
    public enum Event: String, CaseIterable {
        case connectionsAccepted = "connections_accepted", connectionsClosed = "connections_closed"
        case disconnectsWithInFlight = "disconnects_with_in_flight"
        case requestsReceived = "requests_received", requestsAnswered = "requests_answered"
        case requestsDropped = "requests_dropped", acceptErrors = "accept_errors"
        case readErrors = "read_errors", writeErrors = "write_errors"
    }
    private let path: String
    private let started = Date()
    private let startedUptime = ProcessInfo.processInfo.systemUptime
    private var events: [(Double, Event, Int)] = []
    private var overflowAt: Double?
    public static let windowSeconds: Double = 900

    public init(path: String) { self.path = path }

    public func record(_ event: Event, count: Int = 1) {
        let now = ProcessInfo.processInfo.systemUptime
        events.removeAll { $0.0 <= now - Self.windowSeconds }
        if events.count >= 10_000 {
            events.removeFirst()
            overflowAt = now
        }
        events.append((now, event, count))
    }

    public func publish(dbPath: String, inFlight: Int, stopped: Bool = false) {
        let now = ProcessInfo.processInfo.systemUptime
        events.removeAll { $0.0 <= now - Self.windowSeconds }
        let reason = stopped ? "BrainBar server stopped" :
            ((overflowAt.map { now - $0 < Self.windowSeconds } ?? false) ? "transport event buffer overflow" : "")
        var value: [String: Any] = [
            "schema_version": 1, "state": reason.isEmpty ? "measured" : "unmeasurable", "reason": reason,
            "db_path": URL(fileURLWithPath: dbPath).standardizedFileURL.resolvingSymlinksInPath().path,
            "generated_at": ISO8601DateFormatter().string(from: Date()),
            "producer_started_at": ISO8601DateFormatter().string(from: started),
            "window_seconds": Int(Self.windowSeconds)
        ]
        if reason.isEmpty {
            var counts = Dictionary(uniqueKeysWithValues: Event.allCases.map { ($0.rawValue, 0) })
            for (_, event, count) in events { counts[event.rawValue, default: 0] += count }
            value["counts"] = counts
            value["observed_seconds"] = min(Int(now - startedUptime), Int(Self.windowSeconds))
            value["in_flight_requests"] = inFlight
        }
        do {
            let destination = URL(fileURLWithPath: path)
            try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            let temporary = destination.deletingLastPathComponent().appendingPathComponent(".transport-\(UUID().uuidString).tmp")
            defer { try? FileManager.default.removeItem(at: temporary) }
            let data = try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
            guard FileManager.default.createFile(atPath: temporary.path, contents: data, attributes: [.posixPermissions: 0o600]) else {
                throw CocoaError(.fileWriteUnknown)
            }
            guard rename(temporary.path, destination.path) == 0 else { throw CocoaError(.fileWriteUnknown) }
        } catch {
            NSLog("[BrainBar] transport health export failed: %@", String(describing: error))
        }
    }
}
