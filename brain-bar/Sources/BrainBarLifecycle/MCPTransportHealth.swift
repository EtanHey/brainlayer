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
    private struct Identity: Decodable { let schema_version: Int; let db_path: String }
    private struct Episode: Codable {
        let dbPath: String, generatedAt: Date, producerStartedAt: Date
        let windowSeconds: Int, observedSeconds: Int
        let counts: [String: Int]
        var wire: [String: Any] {
            ["state": "measured", "reason": "", "db_path": dbPath,
             "generated_at": ISO8601DateFormatter().string(from: generatedAt),
             "producer_started_at": ISO8601DateFormatter().string(from: producerStartedAt),
             "window_seconds": windowSeconds, "observed_seconds": observedSeconds, "counts": counts]
        }
        static func parse(_ value: Any, db: String, now: Date) throws -> Episode {
            guard let raw = value as? [String: Any], raw["state"] as? String == "measured", raw["reason"] as? String == "" else {
                throw CocoaError(.fileReadCorruptFile)
            }
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            decoder.dateDecodingStrategy = .iso8601
            let frame = try decoder.decode(Episode.self, from: JSONSerialization.data(withJSONObject: raw))
            guard frame.dbPath == db, frame.generatedAt <= now, frame.producerStartedAt <= frame.generatedAt,
                  frame.windowSeconds == Int(MCPTransportHealth.windowSeconds), (0...Int(MCPTransportHealth.windowSeconds)).contains(frame.observedSeconds),
                  Double(frame.observedSeconds) <= frame.generatedAt.timeIntervalSince(frame.producerStartedAt) + 1,
                  Set(frame.counts.keys) == Set(Event.allCases.map(\.rawValue)), frame.counts.values.allSatisfy({ $0 >= 0 }) else {
                throw CocoaError(.fileReadCorruptFile)
            }
            return frame
        }
    }
    private let path: String
    private let now: () -> Date
    private let uptime: () -> TimeInterval
    private let started: Date
    private let startedUptime: TimeInterval
    private var events: [(Double, Event, Int)] = []
    private var overflowAt: Double?
    private var restored = false
    private var incident: Episode?
    private var incidentUnknown = false
    private let reconnect = MCPReconnectRecovery()
    public static let windowSeconds: Double = 900
    public static let minimumConnectionAgeSeconds: Double = 10

    public init(path: String, now: @escaping () -> Date = Date.init,
                uptime: @escaping () -> TimeInterval = { ProcessInfo.processInfo.systemUptime }) {
        self.path = path; self.now = now; self.uptime = uptime
        started = now(); startedUptime = uptime()
    }

    public func record(_ event: Event, count: Int = 1) {
        let instant = uptime()
        events.removeAll { $0.0 <= instant - Self.windowSeconds }
        if events.count >= 10_000 { events.removeFirst(); overflowAt = instant }
        events.append((instant, event, count))
    }

    private static func failing(_ counts: [String: Int]) -> Bool {
        let errors = ["accept_errors", "read_errors", "write_errors"].reduce(0) { min(5, $0 + min(5, counts[$1, default: 0])) }
        return counts["requests_dropped", default: 0] >= 3 || errors >= 5 ||
            (counts["connections_accepted", default: 0] >= 50 && counts["connections_closed", default: 0] >= 50 &&
             counts["requests_answered", default: 0] < counts["connections_closed", default: 0])
    }

    private func restore(db: String) {
        guard !restored else { return }
        restored = true
        do {
            let data = try TransportReceiptFile.read(path)
            let identity = try JSONDecoder().decode(Identity.self, from: data)
            guard let value = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  identity.schema_version == 1, identity.db_path == db else {
                throw CocoaError(.fileReadCorruptFile)
            }
            reconnect.restore(value, db: db, wall: now())
            if let saved = value["incident"], !(saved is NSNull) {
                let old = try Episode.parse(saved, db: db, now: now())
                guard Self.failing(old.counts) else { throw CocoaError(.fileReadCorruptFile) }
                incident = old
            } else if value["incident"] == nil, value["state"] as? String == "measured" {
                // Upgrade a still-retained PR-2 failure window without inventing absent evidence.
                let old = try Episode.parse(value, db: db, now: now())
                if Self.failing(old.counts) { incident = old }
            }
        } catch TransportReceiptFile.Failure.missing { return }
        catch { incidentUnknown = true; reconnect.refuse() }
    }

    public func publish(dbPath: String, inFlight: Int, stopped: Bool = false, persistentConnections: Int = 0) {
        let instant = uptime()
        let wall = now()
        let db = URL(fileURLWithPath: dbPath).standardizedFileURL.resolvingSymlinksInPath().path
        restore(db: db)
        events.removeAll { $0.0 <= instant - Self.windowSeconds }
        let observed = min(Int(instant - startedUptime), Int(Self.windowSeconds))
        let overflow = overflowAt.map { instant - $0 < Self.windowSeconds } ?? false
        var counts = Dictionary(uniqueKeysWithValues: Event.allCases.map { ($0.rawValue, 0) })
        for (_, event, count) in events { counts[event.rawValue, default: 0] += count }
        if !overflow, Self.failing(counts) {
            incident = Episode(dbPath: db, generatedAt: wall, producerStartedAt: started,
                windowSeconds: Int(Self.windowSeconds), observedSeconds: observed, counts: counts)
            incidentUnknown = false
        } else if !overflow, observed == Int(Self.windowSeconds), counts["requests_dropped"] == 0,
                  ["accept_errors", "read_errors", "write_errors"].allSatisfy({ counts[$0] == 0 }), !Self.failing(counts) {
            incident = nil; incidentUnknown = false
        }
        let recovery = reconnect.publish(db: db, wall: wall, elapsed: instant - startedUptime,
            live: persistentConnections, stopped: stopped)
        let reason = stopped ? "BrainBar server stopped" : (reconnect.unknown ? reconnect.reason : (incidentUnknown ? "previous transport receipt refused or unreadable" : (overflow ? "transport event buffer overflow" : "")))
        var value: [String: Any] = ["schema_version": 1, "state": reason.isEmpty ? "measured" : "unmeasurable", "reason": reason,
            "db_path": db, "generated_at": ISO8601DateFormatter().string(from: wall),
            "producer_started_at": ISO8601DateFormatter().string(from: started), "window_seconds": Int(Self.windowSeconds)]
        value["incident"] = incident?.wire ?? (incidentUnknown ?
            ["state": "unmeasurable", "reason": "previous transport incident unreadable"] : NSNull())
        value["reconnect"] = recovery
        if reason.isEmpty {
            value["counts"] = counts; value["observed_seconds"] = observed; value["in_flight_requests"] = inFlight
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
        } catch { NSLog("[BrainBar] transport health export failed: %@", String(describing: error)) }
    }
}
