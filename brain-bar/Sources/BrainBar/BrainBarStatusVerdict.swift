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

    var dashboardStatus: BrainBarOnePageStatus {
        .init(headline: "All good", reason: nil, tone: .green)
    }
}
