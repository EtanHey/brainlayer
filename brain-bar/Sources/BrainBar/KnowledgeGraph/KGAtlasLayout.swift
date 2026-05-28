import CoreGraphics
import Foundation

enum KGAtlasLayout {
    static func seededNodes(
        _ nodes: [KGNode],
        canvasSize: CGSize,
        mode: KGAtlasMode = .importance
    ) -> [KGNode] {
        let width = max(canvasSize.width, 640)
        let height = max(canvasSize.height, 480)
        switch mode {
        case .importance:
            return seededImportanceNodes(nodes, canvasSize: CGSize(width: width, height: height))
        case .tieredAltitude:
            return seededTieredAltitudeNodes(nodes, canvasSize: CGSize(width: width, height: height))
        }
    }

    private static func seededImportanceNodes(_ nodes: [KGNode], canvasSize: CGSize) -> [KGNode] {
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

    private static func seededTieredAltitudeNodes(_ nodes: [KGNode], canvasSize: CGSize) -> [KGNode] {
        let width = max(canvasSize.width, 640)
        let height = max(canvasSize.height, 480)
        let tierRows = tierRowCenters(canvasSize: CGSize(width: width, height: height))
        let grouped = Dictionary(grouping: nodes.enumerated()) { indexedNode in
            KGAltitudeTier.tier(for: indexedNode.element)
        }

        var seeded = nodes
        for tier in KGAltitudeTier.allCases {
            guard let indexedNodes = grouped[tier], !indexedNodes.isEmpty else { continue }
            let sorted = indexedNodes.sorted {
                if $0.element.importance == $1.element.importance {
                    return $0.element.name.localizedCaseInsensitiveCompare($1.element.name) == .orderedAscending
                }
                return $0.element.importance > $1.element.importance
            }

            for (offset, indexed) in sorted.enumerated() {
                seeded[indexed.offset].position = tieredPosition(
                    for: offset,
                    total: sorted.count,
                    rowY: tierRows[tier] ?? height * 0.5,
                    canvasSize: CGSize(width: width, height: height),
                    tier: tier
                )
                seeded[indexed.offset].velocity = .zero
            }
        }

        return seeded
    }

    private static func makeRegionCenters(canvasSize: CGSize) -> [String: CGPoint] {
        let width = max(canvasSize.width, 640)
        let height = max(canvasSize.height, 480)
        let columns: [CGFloat] = [0.26, 0.5, 0.78]
        let rows: [CGFloat] = [0.30, 0.56, 0.76]

        return [
            "person": CGPoint(x: width * columns[0], y: height * rows[1]),
            "project": CGPoint(x: width * columns[1], y: height * rows[0]),
            "tool": CGPoint(x: width * columns[2], y: height * rows[0]),
            "technology": CGPoint(x: width * columns[0], y: height * rows[2]),
            "agent": CGPoint(x: width * columns[1], y: height * rows[1]),
            "company": CGPoint(x: width * columns[2], y: height * rows[1]),
            "topic": CGPoint(x: width * columns[1], y: height * rows[2]),
            "decision": CGPoint(x: width * columns[2], y: height * rows[2]),
        ]
    }

    private static func tierRowCenters(canvasSize: CGSize) -> [KGAltitudeTier: CGFloat] {
        let height = max(canvasSize.height, 480)
        return [
            .summit: height * 0.26,
            .orbit: height * 0.40,
            .signal: height * 0.55,
            .field: height * 0.70,
            .ground: height * 0.84,
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

    private static func tieredPosition(
        for offset: Int,
        total: Int,
        rowY: CGFloat,
        canvasSize: CGSize,
        tier: KGAltitudeTier
    ) -> CGPoint {
        guard total > 1 else {
            return CGPoint(x: canvasSize.width * (tier == .summit ? 0.42 : 0.5), y: rowY)
        }

        let usableWidth = max(canvasSize.width - 180, 360)
        let step = usableWidth / CGFloat(max(total - 1, 1))
        let x = 90 + CGFloat(offset) * step
        let stagger = (offset % 2 == 0 ? -1.0 : 1.0) * min(18, canvasSize.height * 0.025)

        return CGPoint(x: x, y: rowY + stagger)
    }
}
