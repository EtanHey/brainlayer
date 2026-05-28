import Foundation

enum KGAtlasMode: String, CaseIterable, Identifiable {
    case importance
    case tieredAltitude

    var id: String { rawValue }

    var title: String {
        switch self {
        case .importance: "Importance"
        case .tieredAltitude: "Tiered"
        }
    }
}

enum KGAltitudeTier: Int, CaseIterable, Identifiable {
    case summit = 0
    case orbit = 1
    case signal = 2
    case field = 3
    case ground = 4

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .summit: "Summit"
        case .orbit: "Orbit"
        case .signal: "Signal"
        case .field: "Field"
        case .ground: "Ground"
        }
    }

    var caption: String {
        switch self {
        case .summit: "Etan + Claude Code"
        case .orbit: "core operators"
        case .signal: "high-signal systems"
        case .field: "active context"
        case .ground: "full graph"
        }
    }

    static func tier(for node: KGNode) -> KGAltitudeTier {
        let canonicalName = node.name.trimmingCharacters(in: .whitespacesAndNewlines).localizedLowercase
        if topAltitudeEntityNames.contains(canonicalName) {
            return .summit
        }

        if node.importance >= 8.5 || (node.entityType == "agent" && node.importance >= 8) {
            return .orbit
        }

        if node.importance >= 7 || node.linkedChunkCount >= 50 {
            return .signal
        }

        if node.importance >= 5 || node.linkedChunkCount >= 15 {
            return .field
        }

        return .ground
    }

    static func visibleTiers(at altitude: Double) -> Set<KGAltitudeTier> {
        let clampedAltitude = max(0, min(Double(KGAltitudeTier.allCases.count - 1), altitude))
        let level = Int(clampedAltitude.rounded())
        return Set(KGAltitudeTier.allCases.filter { $0.rawValue <= level })
    }

    static func tier(at altitude: Double) -> KGAltitudeTier {
        let clampedAltitude = max(0, min(Double(KGAltitudeTier.allCases.count - 1), altitude))
        return KGAltitudeTier(rawValue: Int(clampedAltitude.rounded())) ?? .signal
    }

    private static let topAltitudeEntityNames: Set<String> = [
        "etan",
        "etan heyman",
        "claude code",
    ]
}
