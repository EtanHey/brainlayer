import SwiftUI

@MainActor
final class KGViewModel: ObservableObject {
    @Published var nodes: [KGNode] = []
    @Published var edges: [KGEdge] = []
    @Published var selectedNodeId: String?
    @Published var selectedEntity: EntityCard?
    @Published var selectedEntityChunks: [BrainDatabase.KGChunkRow] = []

    /// Set by KGCanvasView via GeometryReader — used for centering force
    var canvasCenter: CGPoint = CGPoint(x: 300, y: 250)

    private let database: BrainDatabase

    // Force simulation parameters
    private let repulsionStrength: CGFloat = 5000
    private let attractionStrength: CGFloat = 0.005
    private let centeringStrength: CGFloat = 0.01
    private let damping: CGFloat = 0.85
    private let idealEdgeLength: CGFloat = 150

    init(database: BrainDatabase) {
        self.database = database
    }

    // MARK: - Data Loading

    func loadGraph() {
        do {
            let entityRows = try database.fetchKGEntities()
            let relationRows = try database.fetchKGRelations()

            let entityIds = Set(entityRows.map(\.id))

            nodes = entityRows.map { row in
                KGNode(
                    id: row.id,
                    name: row.name,
                    entityType: row.entityType,
                    importance: row.importance
                )
            }

            // Only include edges where both endpoints exist
            edges = relationRows.compactMap { row in
                guard entityIds.contains(row.sourceId), entityIds.contains(row.targetId) else {
                    return nil
                }
                return KGEdge(
                    sourceId: row.sourceId,
                    targetId: row.targetId,
                    relationType: row.relationType
                )
            }
        } catch {
            nodes = []
            edges = []
        }
    }

    // MARK: - Selection

    func selectNode(id: String?) {
        selectedNodeId = id
        if let id {
            if let lookup = try? database.lookupEntity(query: nodeById(id)?.name ?? "") {
                selectedEntity = EntityCard(lookupPayload: lookup)
            }
            selectedEntityChunks = (try? database.fetchEntityChunks(entityId: id)) ?? []
        } else {
            selectedEntity = nil
            selectedEntityChunks = []
        }
    }

    // MARK: - Hit Testing

    func nodeAt(point: CGPoint) -> KGNode? {
        for node in nodes {
            let dx = point.x - node.position.x
            let dy = point.y - node.position.y
            let dist = sqrt(dx * dx + dy * dy)
            if dist <= node.radius + 4 { // 4pt tolerance
                return node
            }
        }
        return nil
    }

    // MARK: - Force-Directed Layout

    func tick() {
        guard nodes.count > 1 else { return }

        var forces = Array(repeating: CGVector.zero, count: nodes.count)
        let center = canvasCenter

        // Repulsion: all pairs (Coulomb's law)
        for i in 0..<nodes.count {
            for j in (i + 1)..<nodes.count {
                let dx = nodes[i].position.x - nodes[j].position.x
                let dy = nodes[i].position.y - nodes[j].position.y
                let distSq = max(dx * dx + dy * dy, 1)
                let force = repulsionStrength / distSq
                let dist = sqrt(distSq)
                let fx = (dx / dist) * force
                let fy = (dy / dist) * force
                forces[i].dx += fx
                forces[i].dy += fy
                forces[j].dx -= fx
                forces[j].dy -= fy
            }
        }

        // Attraction: edges (spring force)
        let nodeIndex = Dictionary(uniqueKeysWithValues: nodes.enumerated().map { ($1.id, $0) })
        for edge in edges {
            guard let si = nodeIndex[edge.sourceId], let ti = nodeIndex[edge.targetId] else { continue }
            let dx = nodes[ti].position.x - nodes[si].position.x
            let dy = nodes[ti].position.y - nodes[si].position.y
            let dist = max(sqrt(dx * dx + dy * dy), 1)
            let displacement = dist - idealEdgeLength
            let fx = (dx / dist) * displacement * attractionStrength
            let fy = (dy / dist) * displacement * attractionStrength
            forces[si].dx += fx
            forces[si].dy += fy
            forces[ti].dx -= fx
            forces[ti].dy -= fy
        }

        // Centering force
        for i in 0..<nodes.count {
            forces[i].dx += (center.x - nodes[i].position.x) * centeringStrength
            forces[i].dy += (center.y - nodes[i].position.y) * centeringStrength
        }

        // Apply forces with damping
        for i in 0..<nodes.count {
            nodes[i].velocity.dx = (nodes[i].velocity.dx + forces[i].dx) * damping
            nodes[i].velocity.dy = (nodes[i].velocity.dy + forces[i].dy) * damping
            nodes[i].position.x += nodes[i].velocity.dx
            nodes[i].position.y += nodes[i].velocity.dy
        }
    }

    // MARK: - Helpers

    private func nodeById(_ id: String) -> KGNode? {
        nodes.first { $0.id == id }
    }
}
