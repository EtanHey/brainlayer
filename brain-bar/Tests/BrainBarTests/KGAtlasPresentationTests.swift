import XCTest
@testable import BrainBar

final class KGAtlasPresentationTests: XCTestCase {
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
}
