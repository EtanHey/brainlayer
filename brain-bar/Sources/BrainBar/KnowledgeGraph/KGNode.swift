import SwiftUI

struct KGNode: Identifiable, Equatable {
    let id: String
    let name: String
    let entityType: String
    let importance: Double

    var position: CGPoint
    var velocity: CGVector

    /// Radius scales with importance (min 8, max 28)
    var radius: CGFloat {
        CGFloat(8 + (importance / 10.0) * 20)
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
        position: CGPoint? = nil,
        velocity: CGVector = .zero
    ) {
        self.id = id
        self.name = name
        self.entityType = entityType
        self.importance = importance
        self.position = position ?? CGPoint(
            x: CGFloat.random(in: 50...750),
            y: CGFloat.random(in: 50...550)
        )
        self.velocity = velocity
    }
}
