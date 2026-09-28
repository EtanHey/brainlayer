import Foundation

enum SearchProfileLogger {
    static var isEnabled: Bool {
        ProcessInfo.processInfo.environment["BRAINLAYER_SEARCH_PROFILE"] == "1"
    }

    static func newQueryID() -> String {
        "q-\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(12))"
    }

    /// The profiling sink. Tests swap it to capture lines; production uses NSLog.
    static let defaultSink: @Sendable (String) -> Void = { line in NSLog("%@", line) }
    nonisolated(unsafe) static var sink: @Sendable (String) -> Void = defaultSink

    /// Defense in depth for ids that cross an internal boundary (router → helper
    /// client → Python helper): only the exact shape `newQueryID()` (and Python's
    /// `search_profile.new_query_id()`) produces is accepted, `q-` plus 12 ASCII hex
    /// digits. A shape check cannot prove origin, so a client-supplied id is never
    /// read at all: `MCPRouter` always generates its own.
    static func acceptedQueryID(_ raw: Any?) -> String? {
        guard let raw = raw as? String else { return nil }
        let scalars = Array(raw.unicodeScalars)
        guard scalars.count == 14, raw.hasPrefix("q-") else { return nil }
        let hex = scalars.dropFirst(2)
        guard hex.allSatisfy({ ("0"..."9").contains($0) || ("a"..."f").contains($0) || ("A"..."F").contains($0) }) else {
            return nil
        }
        return raw
    }

    static func now() -> TimeInterval {
        ProcessInfo.processInfo.systemUptime
    }

    static func durationMS(since startedAt: TimeInterval) -> Double {
        ((now() - startedAt) * 1000).rounded(toPlaces: 3)
    }

    static func log(
        scope: String,
        step: String,
        queryID: String?,
        durMS: Double? = nil,
        fields: [String: Any] = [:]
    ) {
        guard isEnabled else { return }

        var event: [String: Any] = [
            "ts": isoTimestamp(),
            "scope": scope,
            "step": step
        ]
        if let queryID {
            event["query_id"] = acceptedQueryID(queryID) ?? "<rejected>"
        }
        if let durMS {
            event["dur_ms"] = durMS
        }
        let reservedKeys: Set<String> = ["ts", "scope", "step", "query_id", "dur_ms"]
        for (key, value) in fields where !reservedKeys.contains(key) {
            event[key] = value
        }

        guard JSONSerialization.isValidJSONObject(event),
              let data = try? JSONSerialization.data(withJSONObject: event, options: [.sortedKeys]),
              let line = String(data: data, encoding: .utf8) else {
            return
        }
        sink(line)
    }

    private static func isoTimestamp() -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.string(from: Date())
    }
}

private extension Double {
    func rounded(toPlaces places: Int) -> Double {
        let divisor = pow(10.0, Double(places))
        return (self * divisor).rounded() / divisor
    }
}
