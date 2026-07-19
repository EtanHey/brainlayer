import Foundation

struct KGAtlasPresentation {
    enum FindMove {
        case up
        case down
    }

    struct FindResult: Equatable, Identifiable {
        let id: String
        let name: String
        let entityTypeTitle: String
        let degree: Int

        var relationshipSummary: String {
            switch degree {
            case 0: "No visible relationships"
            case 1: "1 visible relationship"
            default: "\(degree) visible relationships"
            }
        }

        var accessibilityLabel: String {
            "\(name), \(entityTypeTitle), \(relationshipSummary)"
        }
    }

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
        let activeAltitudeTier: KGAltitudeTier
    }

    static func snapshot(
        nodes: [KGNode],
        edges: [KGEdge],
        selectedNodeId: String?,
        minimumImportance: Double,
        mode: KGAtlasMode = .importance,
        altitude: Double = Double(KGAltitudeTier.signal.rawValue),
        userDefaults: UserDefaults = .standard
    ) -> Snapshot {
        let activeAltitudeTier = KGAltitudeTier.tier(at: altitude)
        let visibleAltitudeTiers = KGAltitudeTier.visibleTiers(at: altitude)
        let visibleNodes = nodes.filter { node in
            switch mode {
            case .importance:
                node.importance >= minimumImportance
                    || node.id == selectedNodeId
                    || pinnedEntityNames.contains(node.name.localizedLowercase)
            case .tieredAltitude:
                visibleAltitudeTiers.contains(KGAltitudeTier.tier(for: node))
                    || node.id == selectedNodeId
                    || pinnedEntityNames.contains(node.name.localizedLowercase)
            }
        }

        let regions = regions(for: visibleNodes, mode: mode)
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
            selectedRegion: selectedRegion,
            activeAltitudeTier: activeAltitudeTier
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

    static func findResults(
        nodes: [KGNode],
        edges: [KGEdge],
        query: String
    ) -> [FindResult] {
        var degrees: [String: Int] = [:]
        for edge in edges {
            degrees[edge.sourceId, default: 0] += 1
            degrees[edge.targetId, default: 0] += 1
        }

        let terms = query
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .split(whereSeparator: \.isWhitespace)
            .map { $0.localizedLowercase }

        return nodes.compactMap { node in
            let result = FindResult(
                id: node.id,
                name: node.name,
                entityTypeTitle: singularTitle(for: node.entityType),
                degree: degrees[node.id, default: 0]
            )
            let searchableText = "\(node.name) \(node.entityType) \(result.entityTypeTitle)"
                .localizedLowercase
            guard terms.allSatisfy(searchableText.contains) else { return nil }
            return result
        }
        .sorted { lhs, rhs in
            let nameOrder = lhs.name.localizedCaseInsensitiveCompare(rhs.name)
            if nameOrder != .orderedSame {
                return nameOrder == .orderedAscending
            }
            if lhs.entityTypeTitle != rhs.entityTypeTitle {
                return lhs.entityTypeTitle < rhs.entityTypeTitle
            }
            return lhs.id < rhs.id
        }
    }

    static func nextFindResultID(
        currentID: String?,
        move: FindMove,
        results: [FindResult]
    ) -> String? {
        guard !results.isEmpty else { return nil }
        guard let currentIndex = results.firstIndex(where: { $0.id == currentID }) else {
            return move == .down ? results.first?.id : results.last?.id
        }

        switch move {
        case .up:
            return results[max(currentIndex - 1, results.startIndex)].id
        case .down:
            return results[min(currentIndex + 1, results.index(before: results.endIndex))].id
        }
    }

    static func priorityLabelNodeIDs(
        nodes: [KGNode],
        selectedNodeID: String?,
        scale: CGFloat
    ) -> Set<String> {
        let maxLabels = scale < 0.7 ? 18 : (scale < 1.15 ? 36 : 72)
        let orderedNodes = nodes.sorted { lhs, rhs in
            let lhsSelected = lhs.id == selectedNodeID
            let rhsSelected = rhs.id == selectedNodeID
            if lhsSelected != rhsSelected { return lhsSelected }

            let lhsOwner = pinnedEntityNames.contains(lhs.name.localizedLowercase)
            let rhsOwner = pinnedEntityNames.contains(rhs.name.localizedLowercase)
            if lhsOwner != rhsOwner { return lhsOwner }

            if lhs.importance != rhs.importance { return lhs.importance > rhs.importance }
            if lhs.linkedChunkCount != rhs.linkedChunkCount {
                return lhs.linkedChunkCount > rhs.linkedChunkCount
            }
            let nameOrder = lhs.name.localizedCaseInsensitiveCompare(rhs.name)
            if nameOrder != .orderedSame { return nameOrder == .orderedAscending }
            return lhs.id < rhs.id
        }

        var acceptedRects: [CGRect] = []
        var acceptedIDs = Set<String>()
        for node in orderedNodes {
            if acceptedIDs.count >= maxLabels, node.id != selectedNodeID { continue }
            let rect = approximateLabelRect(for: node, scale: scale)
            guard node.id == selectedNodeID || !acceptedRects.contains(where: { $0.intersects(rect) }) else {
                continue
            }
            acceptedRects.append(rect)
            acceptedIDs.insert(node.id)
        }
        return acceptedIDs
    }

    /// The canvas retains relationship topology while the inspector owns the
    /// full, readable edge list. Inline relation text becomes illegible on
    /// dense graphs and competes with priority node labels.
    static func lineOnlyEdge(_ edge: KGEdge) -> KGEdge {
        KGEdge(
            sourceId: edge.sourceId,
            targetId: edge.targetId,
            relationType: ""
        )
    }

    static func modeEffectDescription(
        mode: KGAtlasMode,
        minimumImportance: Double,
        altitudeTier: KGAltitudeTier
    ) -> String {
        switch mode {
        case .importance:
            return "Shows entities with importance \(Int(minimumImportance))/10 or higher. Selected and pinned entities remain visible."
        case .tieredAltitude:
            return "Shows Summit through \(altitudeTier.title). Lower altitude reveals more entities."
        }
    }

    static func canvasAccessibilityValue(
        visibleEntityCount: Int,
        visibleRelationshipCount: Int
    ) -> String {
        let entities = visibleEntityCount == 1 ? "1 visible entity" : "\(visibleEntityCount) visible entities"
        let relationships = visibleRelationshipCount == 1
            ? "1 visible relationship"
            : "\(visibleRelationshipCount) visible relationships"
        return "\(entities) and \(relationships)"
    }

    static let canvasAccessibilityHint =
        "Use Find entity to browse and select every visible entity without spatial pointing."

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

    private static func singularTitle(for entityType: String) -> String {
        switch entityType {
        case "person": "Person"
        case "project": "Project"
        case "tool": "Tool"
        case "technology": "Technology"
        case "agent": "Agent"
        case "company": "Company"
        case "topic": "Topic"
        case "decision": "Decision"
        default: entityType.isEmpty ? "Entity" : entityType.capitalized
        }
    }

    private static func approximateLabelRect(for node: KGNode, scale: CGFloat) -> CGRect {
        let fontSize = max(9, node.radius * 0.55)
        let width = max(28, CGFloat(node.name.count) * fontSize * 0.58)
        let inverseScale = 1 / max(scale, 0.35)
        let horizontalClearance = 8 * inverseScale
        let verticalClearance = 5 * inverseScale
        return CGRect(
            x: node.position.x - width / 2 - horizontalClearance,
            y: node.position.y + node.radius + 12 - verticalClearance,
            width: width + horizontalClearance * 2,
            height: fontSize + verticalClearance * 2
        )
    }

    private static func maxLinksPerNode(from userDefaults: UserDefaults) -> Int {
        let configuredValue = userDefaults.integer(forKey: maxLinksPerNodeKey)
        return configuredValue > 0 ? configuredValue : defaultMaxLinksPerNode
    }

    private static func regions(for visibleNodes: [KGNode], mode: KGAtlasMode) -> [Region] {
        switch mode {
        case .importance:
            let grouped = Dictionary(grouping: visibleNodes, by: \.entityType)
            return orderedEntityTypes.compactMap { entityType in
                guard let values = grouped[entityType], !values.isEmpty else { return nil }
                return Region(
                    entityType: entityType,
                    title: title(for: entityType),
                    nodes: sortNodes(values)
                )
            }
        case .tieredAltitude:
            let grouped = Dictionary(grouping: visibleNodes) { node in
                KGAltitudeTier.tier(for: node)
            }
            return KGAltitudeTier.allCases.compactMap { tier in
                guard let values = grouped[tier], !values.isEmpty else { return nil }
                return Region(
                    entityType: "altitude-\(tier.rawValue)",
                    title: tier.title,
                    nodes: sortNodes(values)
                )
            }
        }
    }

    private static func sortNodes(_ nodes: [KGNode]) -> [KGNode] {
        nodes.sorted {
            if $0.importance == $1.importance {
                return $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending
            }
            return $0.importance > $1.importance
        }
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
