import XCTest
@testable import BrainBar

final class KGAtlasPresentationTests: XCTestCase {
    private let maxLinksKey = "brainBar.maxLinksPerNode"
    private var userDefaultsSuiteName: String!
    private var testUserDefaults: UserDefaults!

    override func setUp() {
        super.setUp()
        userDefaultsSuiteName = "KGAtlasPresentationTests-\(UUID().uuidString)"
        testUserDefaults = UserDefaults(suiteName: userDefaultsSuiteName)
    }

    override func tearDown() {
        testUserDefaults.removePersistentDomain(forName: userDefaultsSuiteName)
        testUserDefaults = nil
        userDefaultsSuiteName = nil
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
            minimumImportance: 0,
            userDefaults: testUserDefaults
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
            minimumImportance: 5,
            userDefaults: testUserDefaults
        )

        XCTAssertEqual(snapshot.visibleNodes.map { $0.id }, ["p1", "t1", "a1"])
        XCTAssertEqual(snapshot.visibleEdges.map { $0.id }, ["p1-uses-a1", "a1-calls-t1"])
        XCTAssertEqual(snapshot.selectedRegion?.title, "Agents")
    }

    func testAltitudeFilterKeepsOwnerEntityVisible() {
        let nodes = [
            KGNode(id: "owner", name: "Etan Heyman", entityType: "person", importance: 0.5, position: .zero),
            KGNode(id: "site", name: "etanheyman.com", entityType: "project", importance: 6, position: .zero),
            KGNode(id: "low", name: "Scratch", entityType: "topic", importance: 0.5, position: .zero),
        ]
        let edges = [
            KGEdge(sourceId: "owner", targetId: "site", relationType: "owns"),
            KGEdge(sourceId: "owner", targetId: "low", relationType: "mentioned"),
        ]

        let snapshot = KGAtlasPresentation.snapshot(
            nodes: nodes,
            edges: edges,
            selectedNodeId: nil,
            minimumImportance: 6,
            userDefaults: testUserDefaults
        )

        XCTAssertEqual(snapshot.visibleNodes.map(\.id), ["owner", "site"])
        XCTAssertEqual(snapshot.visibleEdges.map(\.id), ["owner-owns-site"])
    }

    func testSnapshotVirtualizesHubLinksAtDefaultFiftyVisibleRelations() {
        let graph = makeHubGraph(edgeCount: 60)

        let snapshot = KGAtlasPresentation.snapshot(
            nodes: graph.nodes,
            edges: graph.edges,
            selectedNodeId: nil,
            minimumImportance: 0,
            userDefaults: testUserDefaults
        )

        XCTAssertEqual(snapshot.visibleEdges.count, 50)
        XCTAssertEqual(snapshot.visibleEdges.first?.targetId, "leaf-1")
        XCTAssertEqual(snapshot.visibleEdges.last?.targetId, "leaf-50")
    }

    func testSnapshotUsesConfiguredMaxLinksPerNode() {
        testUserDefaults.set(12, forKey: maxLinksKey)
        let graph = makeHubGraph(edgeCount: 30)

        let snapshot = KGAtlasPresentation.snapshot(
            nodes: graph.nodes,
            edges: graph.edges,
            selectedNodeId: nil,
            minimumImportance: 0,
            userDefaults: testUserDefaults
        )

        XCTAssertEqual(snapshot.visibleEdges.count, 12)
        XCTAssertEqual(snapshot.visibleEdges.last?.targetId, "leaf-12")
    }

    func testSnapshotKeepsSelectedNodeIncidentEdgeInsideLinkCap() {
        let graph = makeHubGraph(edgeCount: 60)

        let snapshot = KGAtlasPresentation.snapshot(
            nodes: graph.nodes,
            edges: graph.edges,
            selectedNodeId: "leaf-60",
            minimumImportance: 0,
            userDefaults: testUserDefaults
        )

        XCTAssertEqual(snapshot.visibleEdges.count, 50)
        XCTAssertTrue(snapshot.visibleEdges.contains { edge in
            edge.sourceId == "hub" && edge.targetId == "leaf-60"
        })
    }

    func testSnapshotCapsLinksAfterRemovingHiddenEntityTypes() {
        testUserDefaults.set(2, forKey: maxLinksKey)
        let nodes = [
            KGNode(id: "hub", name: "Hub", entityType: "person", importance: 10, position: .zero),
            KGNode(id: "hidden-1", name: "Hidden 1", entityType: "library", importance: 5, position: .zero),
            KGNode(id: "hidden-2", name: "Hidden 2", entityType: "library", importance: 5, position: .zero),
            KGNode(id: "project-1", name: "Project 1", entityType: "project", importance: 5, position: .zero),
            KGNode(id: "project-2", name: "Project 2", entityType: "project", importance: 5, position: .zero),
        ]
        let edges = [
            KGEdge(sourceId: "hub", targetId: "hidden-1", relationType: "uses"),
            KGEdge(sourceId: "hub", targetId: "hidden-2", relationType: "uses"),
            KGEdge(sourceId: "hub", targetId: "project-1", relationType: "owns"),
            KGEdge(sourceId: "hub", targetId: "project-2", relationType: "owns"),
        ]

        let snapshot = KGAtlasPresentation.snapshot(
            nodes: nodes,
            edges: edges,
            selectedNodeId: nil,
            minimumImportance: 0,
            userDefaults: testUserDefaults
        )

        XCTAssertEqual(snapshot.visibleNodes.map(\.id), ["hub", "project-1", "project-2"])
        XCTAssertEqual(snapshot.visibleEdges.map(\.targetId), ["project-1", "project-2"])
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
