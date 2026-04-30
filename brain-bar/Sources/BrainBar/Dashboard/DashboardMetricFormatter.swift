import Foundation

enum DashboardMetricFormatter {
    static func speedString(ratePerMinute: Double) -> String {
        liveBadgeString(ratePerMinute: ratePerMinute)
    }

    static func rateString(totalEvents: Int, activityWindowMinutes: Int) -> String {
        guard activityWindowMinutes > 0 else { return "0/min" }
        return speedString(ratePerMinute: Double(max(totalEvents, 0)) / Double(activityWindowMinutes))
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
        activitySummaryString(
            totalEvents: recentActivityBuckets.reduce(0, +),
            activityWindowMinutes: activityWindowMinutes
        )
    }

    static func activitySummaryString(
        totalEvents: Int,
        activityWindowMinutes: Int = 30
    ) -> String {
        "\(max(totalEvents, 0)) in \(shortWindowLabel(minutes: activityWindowMinutes))"
    }

    static func lastCompletionString(
        lastEventAt: Date?,
        activityWindowMinutes: Int = 30,
        now: Date = Date()
    ) -> String {
        lastEventString(lastEventAt: lastEventAt, activityWindowMinutes: activityWindowMinutes, now: now)
    }

    static func liveBadgeString(ratePerMinute: Double) -> String {
        let clamped = max(ratePerMinute, 0)
        if clamped.rounded(.towardZero) == clamped {
            return "\(Int(clamped))/min"
        }
        return String(format: "%.1f/min", clamped)
    }

    static func lastEventString(
        lastEventAt: Date?,
        activityWindowMinutes: Int = 30,
        now: Date = Date()
    ) -> String {
        guard let lastEventAt else {
            return "\(shortWindowLabel(minutes: activityWindowMinutes))+"
        }

        let secondsAgo = max(now.timeIntervalSince(lastEventAt), 0)
        if secondsAgo < 60 {
            return "Just now"
        }
        if secondsAgo < 3600 {
            let minutesAgo = Int((secondsAgo / 60).rounded(.up))
            return "\(minutesAgo)m ago"
        }

        let hoursAgo = Int((secondsAgo / 3600).rounded(.up))
        return "\(hoursAgo)h ago"
    }

    static func windowLabel(minutes: Int) -> String {
        "Last \(shortWindowLabel(minutes: minutes))"
    }

    static func shortWindowLabel(minutes: Int) -> String {
        guard minutes >= 60 else { return "\(minutes)m" }
        if minutes % 60 == 0 {
            return "\(minutes / 60)h"
        }
        return "\(minutes)m"
    }
}
