import XCTest
@testable import BrainBar

final class KGAtlasPresentationTests: XCTestCase {
    private let maxLinksKey = "brainBar.maxLinksPerNode"

    override func tearDown() {
        UserDefaults.standard.removeObject(forKey: maxLinksKey)
        super.tearDown()
    }

    func testSnapshotBuildsDeterministicRegions() {
        let nodes = [
            KGNode(id: "p1", name: "Etan Heyman", entityType: "person", importance: 9, position: .zero),
            KGNode(id: "pr1", name: "brainlayer", entityType: "project", importance: 8, position: .zero),
            KGNode(id: "t1", name: "mcp", entityType: "tool", importance: 6, position: .zero),
        ]
        let edges = [
            KGEdge(sourceId: "p1", targetId: "pr1", relationType: "builds"),
            KGEdge(sourceId: "pr1", targetId: "t1", relationType: "uses"),
        ]

        let snapshot = KGAtlasPresentation.snapshot(
            nodes: nodes,
            edges: edges,
            selectedNodeId: nil,
            minimumImportance: 0
        )

        XCTAssertEqual(snapshot.regions.map { $0.title }, ["People", "Projects", "Tools"])
        XCTAssertEqual(snapshot.regions.map { $0.nodes.count }, [1, 1, 1])
        XCTAssertEqual(snapshot.visibleNodes.map { $0.id }, ["p1", "pr1", "t1"])
        XCTAssertEqual(snapshot.visibleEdges.count, 2)
    }

    func testAltitudeFilterKeepsSelectedNodeVisible() {
        let nodes = [
            KGNode(id: "p1", name: "Etan Heyman", entityType: "person", importance: 9, position: .zero),
            KGNode(id: "a1", name: "coachClaude", entityType: "agent", importance: 2, position: .zero),
            KGNode(id: "t1", name: "mcp", entityType: "tool", importance: 7, position: .zero),
        ]
        let edges = [
            KGEdge(sourceId: "p1", targetId: "a1", relationType: "uses"),
            KGEdge(sourceId: "a1", targetId: "t1", relationType: "calls"),
        ]

        let snapshot = KGAtlasPresentation.snapshot(
            nodes: nodes,
            edges: edges,
            selectedNodeId: "a1",
            minimumImportance: 5
        )

        XCTAssertEqual(snapshot.visibleNodes.map { $0.id }, ["p1", "t1", "a1"])
        XCTAssertEqual(snapshot.visibleEdges.map { $0.id }, ["p1-uses-a1", "a1-calls-t1"])
        XCTAssertEqual(snapshot.selectedRegion?.title, "Agents")
    }

    func testSnapshotVirtualizesHubLinksAtDefaultFiftyVisibleRelations() {
        UserDefaults.standard.removeObject(forKey: maxLinksKey)
        let graph = makeHubGraph(edgeCount: 60)

        let snapshot = KGAtlasPresentation.snapshot(
            nodes: graph.nodes,
            edges: graph.edges,
            selectedNodeId: nil,
            minimumImportance: 0
        )

        XCTAssertEqual(snapshot.visibleEdges.count, 50)
        XCTAssertEqual(snapshot.visibleEdges.first?.targetId, "leaf-1")
        XCTAssertEqual(snapshot.visibleEdges.last?.targetId, "leaf-50")
    }

    func testSnapshotUsesConfiguredMaxLinksPerNode() {
        UserDefaults.standard.set(12, forKey: maxLinksKey)
        let graph = makeHubGraph(edgeCount: 30)

        let snapshot = KGAtlasPresentation.snapshot(
            nodes: graph.nodes,
            edges: graph.edges,
            selectedNodeId: nil,
            minimumImportance: 0
        )

        XCTAssertEqual(snapshot.visibleEdges.count, 12)
        XCTAssertEqual(snapshot.visibleEdges.last?.targetId, "leaf-12")
    }

    private func makeHubGraph(edgeCount: Int) -> (nodes: [KGNode], edges: [KGEdge]) {
        let hub = KGNode(
            id: "hub",
            name: "Etan Heyman",
            entityType: "person",
            importance: 10,
            position: .zero
        )
        let leaves = (1...edgeCount).map { index in
            KGNode(
                id: "leaf-\(index)",
                name: "Project \(index)",
                entityType: "project",
                importance: 5,
                position: .zero
            )
        }
        let edges = (1...edgeCount).map { index in
            KGEdge(sourceId: "hub", targetId: "leaf-\(index)", relationType: "owns")
        }

        return ([hub] + leaves, edges)
    }
}
