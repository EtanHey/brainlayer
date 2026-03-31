import SwiftUI

/// Draw helpers for rendering a KG node on a Canvas.
/// Not a standalone SwiftUI View — used inside KGCanvasView's Canvas block.
enum KGNodeRenderer {
    static func draw(
        node: KGNode,
        isSelected: Bool,
        in context: inout GraphicsContext,
        environment: EnvironmentValues
    ) {
        let r = node.radius
        let rect = CGRect(
            x: node.position.x - r,
            y: node.position.y - r,
            width: r * 2,
            height: r * 2
        )

        // Circle fill
        let fillColor = isSelected
            ? Color.white
            : node.color
        context.fill(Circle().path(in: rect), with: .color(fillColor.opacity(0.85)))

        // Border ring
        let strokeColor = isSelected ? node.color : Color.white.opacity(0.4)
        context.stroke(
            Circle().path(in: rect),
            with: .color(strokeColor),
            lineWidth: isSelected ? 3 : 1.5
        )

        // Label
        let label = Text(node.name)
            .font(.system(size: max(9, r * 0.55), weight: .medium))
            .foregroundColor(isSelected ? .black : .white)
        context.draw(
            context.resolve(label),
            at: CGPoint(x: node.position.x, y: node.position.y + r + 12),
            anchor: .top
        )
    }
}
