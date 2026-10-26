import Foundation

/// The shared status verdict as the surfaces compiled into both BrainBar targets see it (S5).
/// `BrainBarStatusSnapshot` (app target) derives it; the menu, its icon and the Dashboard render it.
struct BrainBarStatusVerdict: Equatable, Sendable {
    /// Dot-worthy faults, most severe first. Non-empty means the menu-bar dot.
    let attention: [String]
    /// Visible neutral notices; never a dot.
    let notices: [String]
    /// Nothing has been measured yet.
    let isChecking: Bool

    var needsAttention: Bool { !attention.isEmpty }

    /// The Dashboard headline: a count only for dot issues; notices are neutral; "All good" only
    /// when everything was measured and nothing is pending.
    var dashboardStatus: BrainBarOnePageStatus {
        if let first = attention.first {
            return .init(headline: attention.count == 1 ? "1 thing needs you" : "\(attention.count) things need you",
                         reason: first, tone: .amber)
        }
        if let first = notices.first { return .init(headline: "No action needed", reason: first, tone: .neutral) }
        if isChecking { return .init(headline: "Checking…", reason: nil, tone: .neutral) }
        return .init(headline: "All good", reason: nil, tone: .green)
    }

    var tooltip: String {
        needsAttention ? "BrainBar — needs attention: " + attention.joined(separator: "; ") : "BrainBar"
    }
}
