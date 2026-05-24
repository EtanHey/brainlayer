import Foundation

enum KGTemporalDate {
    static func parse(_ raw: Any?) -> Date? {
        if let date = raw as? Date {
            return date
        }
        guard let text = raw as? String, !text.isEmpty else {
            return nil
        }
        if let date = pythonMicrosecondISOFormatter().date(from: text) {
            return date
        }
        if let date = fractionalISOFormatter().date(from: text) {
            return date
        }
        if let date = plainISOFormatter().date(from: text) {
            return date
        }
        return sqliteTimestampFormatter().date(from: text)
    }

    private static func fractionalISOFormatter() -> ISO8601DateFormatter {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }

    private static func plainISOFormatter() -> ISO8601DateFormatter {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter
    }

    private static func pythonMicrosecondISOFormatter() -> DateFormatter {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXXXX"
        return formatter
    }

    private static func sqliteTimestampFormatter() -> DateFormatter {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
        return formatter
    }
}
