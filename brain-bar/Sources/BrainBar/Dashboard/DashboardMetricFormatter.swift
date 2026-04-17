import Foundation

enum DashboardMetricFormatter {
    struct RateDisplay: Equatable {
        let valueText: String
        let unitText: String

        var text: String {
            "\(valueText)\(unitText)"
        }
    }

    static func speedDisplay(ratePerMinute: Double) -> RateDisplay {
        guard let clampedRate = sanitizedRatePerMinute(ratePerMinute) else {
            return RateDisplay(valueText: "0", unitText: "/min")
        }
        if clampedRate == 0 {
            return RateDisplay(valueText: "0", unitText: "/min")
        }
        let perSecond = clampedRate / 60.0

        if perSecond >= 1 {
            return RateDisplay(valueText: formattedRateValue(perSecond), unitText: "/s")
        }
        if clampedRate >= 1 {
            return RateDisplay(valueText: formattedRateValue(clampedRate), unitText: "/min")
        }
        return RateDisplay(valueText: formattedRateValue(clampedRate * 60.0), unitText: "/hr")
    }

    static func speedString(ratePerMinute: Double) -> String {
        speedDisplay(ratePerMinute: ratePerMinute).text
    }

    static func rateDetailString(ratePerMinute: Double) -> String {
        guard let clampedRate = sanitizedRatePerMinute(ratePerMinute) else {
            return "0/s"
        }
        let perSecond = clampedRate / 60.0
        return "\(formattedRateValue(perSecond))/s"
    }

    private static func sanitizedRatePerMinute(_ ratePerMinute: Double) -> Double? {
        guard ratePerMinute.isFinite else { return nil }
        return max(ratePerMinute, 0)
    }

    private static func formattedRateValue(_ value: Double) -> String {
        if value == 0 {
            return "0"
        }
        if value >= 10 {
            return String(Int(value.rounded()))
        }

        let string = String(format: "%.1f", value)
        if string.hasSuffix(".0") {
            return String(string.dropLast(2))
        }
        return string
    }
}
