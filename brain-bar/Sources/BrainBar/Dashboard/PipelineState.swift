import AppKit
import Foundation

typealias DashboardStats = BrainDatabase.DashboardStats

struct DaemonHealthSnapshot: Sendable, Equatable {
    let pid: pid_t
    let isResponsive: Bool
    let rssBytes: UInt64
    let uptime: TimeInterval
    let openConnections: Int
    let lastSeenAt: Date
}

enum PipelineState: String, Sendable, Equatable {
    case offline
    case degraded
    case indexing
    case enriching
    case idle

    static func derive(daemon: DaemonHealthSnapshot?, stats: DashboardStats) -> PipelineState {
        guard let daemon else { return .offline }
        guard daemon.isResponsive else { return .degraded }

        let recentWrites = stats.recentActivityBuckets.reduce(0, +)
        if recentWrites > 0 {
            return .indexing
        }
        if stats.pendingEnrichmentCount > 0 {
            return .enriching
        }
        return .idle
    }

    var label: String {
        switch self {
        case .offline: return "Offline"
        case .degraded: return "Degraded"
        case .indexing: return "Indexing"
        case .enriching: return "Enriching"
        case .idle: return "Idle"
        }
    }

    var symbolName: String {
        switch self {
        case .offline: return "wifi.slash"
        case .degraded: return "exclamationmark.triangle.fill"
        case .indexing: return "waveform.path.ecg"
        case .enriching: return "sparkles"
        case .idle: return "checkmark.circle.fill"
        }
    }

    var color: NSColor {
        switch self {
        case .offline:
            return .systemGray
        case .degraded:
            return .systemOrange
        case .indexing:
            return .systemBlue
        case .enriching:
            return .systemPurple
        case .idle:
            return .systemGreen
        }
    }
}
