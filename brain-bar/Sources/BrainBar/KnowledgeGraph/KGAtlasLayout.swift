import CoreGraphics
import Foundation

enum KGAtlasLayout {
    static func seededNodes(_ nodes: [KGNode], canvasSize: CGSize) -> [KGNode] {
        let width = max(canvasSize.width, 640)
        let height = max(canvasSize.height, 480)
        let regionCenters = makeRegionCenters(canvasSize: CGSize(width: width, height: height))
        let grouped = Dictionary(grouping: nodes.enumerated(), by: \.element.entityType)

        var seeded = nodes
        for (entityType, indexedNodes) in grouped {
            let center = regionCenters[entityType] ?? CGPoint(x: width * 0.5, y: height * 0.5)
            let sorted = indexedNodes.sorted {
                if $0.element.importance == $1.element.importance {
                    return $0.element.name.localizedCaseInsensitiveCompare($1.element.name) == .orderedAscending
                }
                return $0.element.importance > $1.element.importance
            }

            for (offset, indexed) in sorted.enumerated() {
                seeded[indexed.offset].position = position(
                    for: offset,
                    total: sorted.count,
                    center: center
                )
                seeded[indexed.offset].velocity = .zero
            }
        }

        return seeded
    }

    private static func makeRegionCenters(canvasSize: CGSize) -> [String: CGPoint] {
        let columns: [CGFloat] = [0.22, 0.5, 0.78]
        let rows: [CGFloat] = [0.24, 0.5, 0.76]

        return [
            "person": CGPoint(x: canvasSize.width * columns[0], y: canvasSize.height * rows[0]),
            "project": CGPoint(x: canvasSize.width * columns[1], y: canvasSize.height * rows[0]),
            "tool": CGPoint(x: canvasSize.width * columns[2], y: canvasSize.height * rows[0]),
            "technology": CGPoint(x: canvasSize.width * columns[0], y: canvasSize.height * rows[1]),
            "agent": CGPoint(x: canvasSize.width * columns[1], y: canvasSize.height * rows[1]),
            "company": CGPoint(x: canvasSize.width * columns[2], y: canvasSize.height * rows[1]),
            "topic": CGPoint(x: canvasSize.width * columns[0], y: canvasSize.height * rows[2]),
            "decision": CGPoint(x: canvasSize.width * columns[1], y: canvasSize.height * rows[2]),
        ]
    }

    private static func position(for offset: Int, total: Int, center: CGPoint) -> CGPoint {
        guard total > 1 else { return center }

        if offset == 0 {
            return center
        }

        let ring = Int(ceil((sqrt(Double(offset + 1)) - 1) / 2))
        let indexInRing = offset - (ring * ring)
        let countInRing = max(ring * 6, 1)
        let angle = (Double(indexInRing) / Double(countInRing)) * Double.pi * 2
        let radius = CGFloat(42 + ring * 30)

        return CGPoint(
            x: center.x + cos(angle) * radius,
            y: center.y + sin(angle) * radius * 0.72
        )
    }
}
