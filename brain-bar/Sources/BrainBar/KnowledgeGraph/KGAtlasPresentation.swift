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
        minimumImportance: Double
    ) -> Snapshot {
        let visibleNodes = nodes.filter { node in
            node.importance >= minimumImportance || node.id == selectedNodeId
        }

        let visibleIDs = Set(visibleNodes.map(\.id))
        let visibleEdges = edges.filter { visibleIDs.contains($0.sourceId) && visibleIDs.contains($0.targetId) }

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

        let selectedRegion = regions.first { region in
            region.nodes.contains { $0.id == selectedNodeId }
        }

        return Snapshot(
            regions: regions,
            visibleNodes: regions.flatMap { $0.nodes },
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
}
