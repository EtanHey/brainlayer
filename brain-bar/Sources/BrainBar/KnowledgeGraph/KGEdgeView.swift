import SwiftUI

/// Draw helpers for rendering a KG edge on a Canvas.
enum KGEdgeRenderer {
    static func draw(
        edge: KGEdge,
        sourcePos: CGPoint,
        targetPos: CGPoint,
        isHighlighted: Bool,
        in context: inout GraphicsContext,
        environment: EnvironmentValues
    ) {
        let darkMode = environment.colorScheme == .dark
        var path = Path()
        path.move(to: sourcePos)
        path.addLine(to: targetPos)

        let color: Color = isHighlighted
            ? (darkMode ? .brainBarWhite : .brainBarBlack.opacity(0.8))
            : (darkMode ? .brainBarTextMuted.opacity(0.35) : .brainBarBlack.opacity(0.18))
        context.stroke(path, with: .color(color), lineWidth: isHighlighted ? 2 : 1)

        // Relation label at midpoint
        let mid = CGPoint(
            x: (sourcePos.x + targetPos.x) / 2,
            y: (sourcePos.y + targetPos.y) / 2
        )
        let label = Text(edge.relationType)
            .font(.system(size: 8))
            .foregroundColor(
                isHighlighted
                    ? (darkMode ? Color.brainBarWhite.opacity(0.9) : Color.brainBarBlack.opacity(0.75))
                    : (darkMode ? Color.brainBarTextMuted.opacity(0.5) : Color.brainBarBlack.opacity(0.4))
            )
        context.draw(context.resolve(label), at: mid, anchor: .center)
    }
}
