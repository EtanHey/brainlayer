import Foundation
import Darwin

/// Queue-owned, payload-free baseline. Recovery measures live aged sockets,
/// not repeated initialization or the identities of the original clients.
final class MCPReconnectRecovery {
    private struct Snapshot: Decodable {
        let dbPath: String, bootTime: String, severedAt: Date
        let minimumConnectionAgeSeconds: Int, graceSeconds: Int, observedSeconds: Int
        let severedPersistent: Int, recoveredPersistent: Int
    }
    private var baseline = 0
    private var severedAt: Date?
    // Five grace periods allow KeepAlive relaunch; older sessions are not reconnect evidence.
    static let maximumBaselineAgeSeconds: Double = 300
    private let bootTime: String? = {
        var stamp = timeval(), size = MemoryLayout<timeval>.size
        guard sysctlbyname("kern.boottime", &stamp, &size, nil, 0) == 0, size == MemoryLayout<timeval>.size else { return nil }
        return "\(stamp.tv_sec):\(stamp.tv_usec)"
    }()
    private var discarded = false
    private(set) var reason = ""
    var unknown: Bool { !reason.isEmpty }
    func refuse() { reason = "previous socket recovery evidence unreadable" }
    private func discard(_ why: String) { baseline = 0; severedAt = nil; discarded = true; reason = why }
    func restore(_ receipt: [String: Any], db: String, wall: Date) {
        guard let saved = receipt["reconnect"], !(saved is NSNull) else { return }
        do {
            if let raw = saved as? [String: Any], raw["state"] as? String == "unmeasurable",
               let why = raw["reason"] as? String, ["baseline from a previous boot; reconnect deficit not measured", "baseline expired; reconnect deficit not measured"].contains(why) { discard(why); return }
            guard let raw = saved as? [String: Any], raw["state"] as? String == "measured", raw["reason"] as? String == "",
                  let generated = receipt["generated_at"] as? String,
                  let started = receipt["producer_started_at"] as? String,
                  let generatedAt = ISO8601DateFormatter().date(from: generated),
                  let startedAt = ISO8601DateFormatter().date(from: started), startedAt <= generatedAt, generatedAt <= wall else {
                throw CocoaError(.fileReadCorruptFile)
            }
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            decoder.dateDecodingStrategy = .iso8601
            let frame = try decoder.decode(Snapshot.self, from: JSONSerialization.data(withJSONObject: raw))
            guard frame.dbPath == db, frame.severedAt <= generatedAt,
                  frame.minimumConnectionAgeSeconds == 10, frame.graceSeconds == 60,
                  (0...60).contains(frame.observedSeconds),
                  Double(frame.observedSeconds) <= generatedAt.timeIntervalSince(startedAt) + 1,
                  frame.severedPersistent > 0, (0...frame.severedPersistent).contains(frame.recoveredPersistent) else {
                throw CocoaError(.fileReadCorruptFile)
            }
            guard let bootTime, !frame.bootTime.isEmpty else { throw CocoaError(.fileReadCorruptFile) }
            if frame.bootTime != bootTime { discard("baseline from a previous boot; reconnect deficit not measured"); return }
            if wall.timeIntervalSince(frame.severedAt) > Self.maximumBaselineAgeSeconds { discard("baseline expired; reconnect deficit not measured"); return }
            baseline = frame.severedPersistent; severedAt = frame.severedAt
        } catch { refuse() }
    }

    func publish(db: String, wall: Date, elapsed: Double, live: Int, stopped: Bool) -> Any {
        if let severedAt, wall.timeIntervalSince(severedAt) > Self.maximumBaselineAgeSeconds { discard("baseline expired; reconnect deficit not measured") }
        if discarded, stopped { discarded = false; reason = "" }
        if discarded {
            defer { discarded = false; reason = "" }
            return ["state": "unmeasurable", "reason": reason]
        }
        if unknown { return ["state": "unmeasurable", "reason": reason] }
        let observed = max(0, min(60, Int(elapsed)))
        // The tripwire allows two ordinary departures; don't accumulate them.
        if observed >= 60, live >= max(0, baseline - 2) { baseline = 0; severedAt = nil }
        if stopped {
            // Preserve an unresolved earlier deficit through another restart.
            if baseline == 0, live > 0 { severedAt = wall }
            baseline = max(baseline, live)
        }
        guard baseline > 0, let severedAt else { return NSNull() }
        guard let bootTime else { refuse(); return ["state": "unmeasurable", "reason": reason] }
        return ["state": "measured", "reason": "", "db_path": db, "boot_time": bootTime,
                "severed_at": ISO8601DateFormatter().string(from: severedAt),
                "minimum_connection_age_seconds": 10, "grace_seconds": 60,
                "observed_seconds": stopped ? 0 : observed,
                "severed_persistent": baseline, "recovered_persistent": stopped ? 0 : min(baseline, max(0, live))]
    }
}
