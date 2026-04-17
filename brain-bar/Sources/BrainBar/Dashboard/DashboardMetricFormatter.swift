import Foundation

enum DashboardMetricFormatter {
    static func speedString(ratePerMinute: Double) -> String {
        liveBadgeString(ratePerMinute: ratePerMinute)
    }

    static func indexingString(
        recentActivityBuckets: [Int],
        activityWindowMinutes: Int = 30
    ) -> String {
        guard activityWindowMinutes > 0 else { return "0/min" }
        let perMinute = Double(recentActivityBuckets.reduce(0, +)) / Double(activityWindowMinutes)
        let clamped = max(perMinute, 0)
        if clamped >= 1 {
            return "\(Int(clamped.rounded()))/min"
        }
        return String(format: "%.1f/min", clamped)
    }

    static func activitySummaryString(
        recentActivityBuckets: [Int],
        activityWindowMinutes: Int = 30
    ) -> String {
        let totalWrites = max(recentActivityBuckets.reduce(0, +), 0)
        return "\(totalWrites) in \(activityWindowMinutes)m"
    }

    static func lastCompletionString(
        recentEnrichmentBuckets: [Int],
        activityWindowMinutes: Int = 30
    ) -> String {
        guard !recentEnrichmentBuckets.isEmpty else { return "30m+" }
        guard let lastIndex = recentEnrichmentBuckets.lastIndex(where: { $0 > 0 }) else {
            return "\(activityWindowMinutes)m+"
        }

        let bucketWidthMinutes = Double(activityWindowMinutes) / Double(recentEnrichmentBuckets.count)
        let bucketsAgo = recentEnrichmentBuckets.count - 1 - lastIndex
        let minutesAgo = Double(bucketsAgo) * bucketWidthMinutes

        if minutesAgo < bucketWidthMinutes {
            return "Just now"
        }

        return "\(Int(minutesAgo.rounded()))m ago"
    }

    static func liveBadgeString(ratePerMinute: Double) -> String {
        let clamped = max(ratePerMinute, 0)
        if clamped.rounded(.towardZero) == clamped {
            return "\(Int(clamped))/min"
        }
        return String(format: "%.1f/min", clamped)
    }
}
