import SwiftUI

struct KGNode: Identifiable, Equatable, Sendable {
    let id: String
    let name: String
    let entityType: String
    let importance: Double
    let linkedChunkCount: Int

    var position: CGPoint
    var velocity: CGVector

    /// Radius scales with importance and linked evidence density.
    var radius: CGFloat {
        let clampedImportance = max(0, min(10, importance))
        let importanceBoost = CGFloat((clampedImportance / 10.0) * 20)
        let evidenceBoost = min(max(CGFloat(linkedChunkCount), 0) / 10.0, 8)
        return 8 + importanceBoost + evidenceBoost
    }

    var color: Color {
        switch entityType {
        case "person": return .blue
        case "project": return .green
        case "tool": return .orange
        case "technology": return .purple
        case "agent": return .cyan
        case "company": return .pink
        case "topic": return .yellow
        case "decision": return .red
        default: return .gray
        }
    }

    init(
        id: String,
        name: String,
        entityType: String,
        importance: Double,
        linkedChunkCount: Int = 0,
        position: CGPoint? = nil,
        velocity: CGVector = .zero
    ) {
        self.id = id
        self.name = name
        self.entityType = entityType
        self.importance = importance
        self.linkedChunkCount = linkedChunkCount
        self.position = position ?? CGPoint(
            x: CGFloat.random(in: 100...500),
            y: CGFloat.random(in: 100...400)
        )
        self.velocity = velocity
    }
}
