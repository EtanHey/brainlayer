import Foundation

enum BrainBarLivePulse {
    static func shouldPulse(previous: [Int], current: [Int]) -> Bool {
        previous != current
    }
}

struct DashboardHeartbeat: Equatable, Sendable {
    let lastEvent: BrainBusEvent?
    let updatedAt: Date?
    let revision: Int

    static let empty = DashboardHeartbeat(
        lastEvent: nil,
        updatedAt: nil,
        revision: 0
    )

    func recording(event: BrainBusEvent?, at timestamp: Date) -> DashboardHeartbeat {
        DashboardHeartbeat(
            lastEvent: event,
            updatedAt: timestamp,
            revision: revision + 1
        )
    }
}
