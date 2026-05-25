import Foundation

enum SearchProfileLogger {
    static var isEnabled: Bool {
        ProcessInfo.processInfo.environment["BRAINLAYER_SEARCH_PROFILE"] == "1"
    }

    static func newQueryID() -> String {
        "q-\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(12))"
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
            event["query_id"] = queryID
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
        NSLog("%@", line)
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
