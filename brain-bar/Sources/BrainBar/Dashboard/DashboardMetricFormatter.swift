import Foundation

enum DashboardMetricFormatter {
    static func speedString(ratePerMinute: Double) -> String {
        let perSecond = max(ratePerMinute, 0) / 60.0
        return String(format: "%.2f/s", perSecond)
    }
}
