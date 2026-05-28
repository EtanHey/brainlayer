import Foundation

struct KGAtlasPresentation {
    struct Region: Equatable, Identifiable {
        let entityType: String
        let title: String
        let nodes: [KGNode]

        var id: String { entityType }
    }

    struct Snapshot: Equatable {
        let regions: [Region]
        let visibleNodes: [KGNode]
        let visibleEdges: [KGEdge]
        let selectedRegion: Region?
    }

    static func snapshot(
        nodes: [KGNode],
        edges: [KGEdge],
        selectedNodeId: String?,
        minimumImportance: Double,
        userDefaults: UserDefaults = .standard
    ) -> Snapshot {
        let visibleNodes = nodes.filter { node in
            node.importance >= minimumImportance
                || node.id == selectedNodeId
                || pinnedEntityNames.contains(node.name.localizedLowercase)
        }

        let grouped = Dictionary(grouping: visibleNodes, by: \.entityType)
        let regions: [Region] = orderedEntityTypes.compactMap { entityType in
            guard let values = grouped[entityType], !values.isEmpty else { return nil }
            return Region(
                entityType: entityType,
                title: title(for: entityType),
                nodes: values.sorted {
                    if $0.importance == $1.importance {
                        return $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending
                    }
                    return $0.importance > $1.importance
                }
            )
        }
        let renderableNodes = regions.flatMap { $0.nodes }
        let renderableIDs = Set(renderableNodes.map(\.id))
        let visibleEdges = virtualizedVisibleEdges(
            from: edges.filter { renderableIDs.contains($0.sourceId) && renderableIDs.contains($0.targetId) },
            maxLinksPerNode: maxLinksPerNode(from: userDefaults),
            selectedNodeId: selectedNodeId
        )

        let selectedRegion = regions.first { region in
            region.nodes.contains { $0.id == selectedNodeId }
        }

        return Snapshot(
            regions: regions,
            visibleNodes: renderableNodes,
            visibleEdges: visibleEdges,
            selectedRegion: selectedRegion
        )
    }

    static func title(for entityType: String) -> String {
        switch entityType {
        case "person": "People"
        case "project": "Projects"
        case "tool": "Tools"
        case "technology": "Technology"
        case "agent": "Agents"
        case "company": "Companies"
        case "topic": "Topics"
        case "decision": "Decisions"
        default: "Other"
        }
    }

    private static let orderedEntityTypes = [
        "person",
        "project",
        "tool",
        "technology",
        "agent",
        "company",
        "topic",
        "decision",
    ]
    private static let pinnedEntityNames: Set<String> = ["etan heyman"]

    private static let maxLinksPerNodeKey = "brainBar.maxLinksPerNode"
    private static let defaultMaxLinksPerNode = 50

    private static func maxLinksPerNode(from userDefaults: UserDefaults) -> Int {
        let configuredValue = userDefaults.integer(forKey: maxLinksPerNodeKey)
        return configuredValue > 0 ? configuredValue : defaultMaxLinksPerNode
    }

    private static func virtualizedVisibleEdges(
        from edges: [KGEdge],
        maxLinksPerNode: Int,
        selectedNodeId: String?
    ) -> [KGEdge] {
        var linkCountsByNode: [String: Int] = [:]
        var visibleEdges: [KGEdge] = []
        let orderedEdges = prioritizedEdges(edges, selectedNodeId: selectedNodeId)

        for edge in orderedEdges {
            let sourceCount = linkCountsByNode[edge.sourceId, default: 0]
            let targetCount = linkCountsByNode[edge.targetId, default: 0]
            guard sourceCount < maxLinksPerNode, targetCount < maxLinksPerNode else {
                continue
            }

            visibleEdges.append(edge)
            linkCountsByNode[edge.sourceId] = sourceCount + 1
            linkCountsByNode[edge.targetId] = targetCount + 1
        }

        return visibleEdges
    }

    private static func prioritizedEdges(_ edges: [KGEdge], selectedNodeId: String?) -> [KGEdge] {
        guard let selectedNodeId else { return edges }
        let incidentEdges = edges.filter { $0.sourceId == selectedNodeId || $0.targetId == selectedNodeId }
        let remainingEdges = edges.filter { $0.sourceId != selectedNodeId && $0.targetId != selectedNodeId }
        return incidentEdges + remainingEdges
    }
}
