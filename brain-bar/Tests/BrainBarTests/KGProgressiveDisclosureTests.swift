import XCTest
@testable import BrainBar

final class KGProgressiveDisclosureTests: XCTestCase {
    deinit {}

    func testEmptyAtlasNeverReservesSidebarWidth() {
        XCTAssertFalse(KGCanvasMetrics.sidebarVisible(
            windowWidth: 1_200,
            selectedEntityVisible: false
        ))
        XCTAssertTrue(KGCanvasMetrics.sidebarVisible(
            windowWidth: 760,
            selectedEntityVisible: true
        ))
    }

    func testFindResultsExposeEveryVisibleEntityWithTypeAndRelationshipDegree() throws {
        let nodes = [
            node(id: "person", name: "Etan Heyman", type: "person", importance: 10),
            node(id: "project", name: "BrainLayer", type: "project", importance: 8),
            node(id: "tool", name: "MCP", type: "tool", importance: 7),
            node(id: "hidden", name: "Scratch", type: "topic", importance: 1),
        ]
        let edges = [
            KGEdge(sourceId: "person", targetId: "project", relationType: "builds"),
            KGEdge(sourceId: "project", targetId: "tool", relationType: "uses"),
        ]
        let snapshot = KGAtlasPresentation.snapshot(
            nodes: nodes,
            edges: edges,
            selectedNodeId: nil,
            minimumImportance: 6,
            mode: .importance
        )

        let allResults = KGAtlasPresentation.findResults(
            nodes: snapshot.visibleNodes,
            edges: snapshot.visibleEdges,
            query: ""
        )
        let project = try XCTUnwrap(allResults.first { $0.id == "project" })

        XCTAssertEqual(Set(allResults.map(\.id)), Set(snapshot.visibleNodes.map(\.id)))
        XCTAssertEqual(project.name, "BrainLayer")
        XCTAssertEqual(project.entityTypeTitle, "Project")
        XCTAssertEqual(project.degree, 2)
        XCTAssertEqual(project.relationshipSummary, "2 visible relationships")
        XCTAssertEqual(
            project.accessibilityLabel,
            "BrainLayer, Project, 2 visible relationships"
        )
        XCTAssertEqual(
            KGAtlasPresentation.findResults(
                nodes: snapshot.visibleNodes,
                edges: snapshot.visibleEdges,
                query: "tool"
            ).map(\.id),
            ["tool"]
        )
    }

    func testFindResultKeyboardSelectionIsDeterministicAndClamped() {
        let results = [
            KGAtlasPresentation.FindResult(
                id: "a",
                name: "Alpha",
                entityTypeTitle: "Project",
                degree: 0
            ),
            KGAtlasPresentation.FindResult(
                id: "b",
                name: "Beta",
                entityTypeTitle: "Tool",
                degree: 1
            ),
            KGAtlasPresentation.FindResult(
                id: "c",
                name: "Gamma",
                entityTypeTitle: "Topic",
                degree: 2
            ),
        ]

        XCTAssertEqual(
            KGAtlasPresentation.nextFindResultID(
                currentID: nil,
                move: .down,
                results: results
            ),
            "a"
        )
        XCTAssertEqual(
            KGAtlasPresentation.nextFindResultID(
                currentID: "a",
                move: .down,
                results: results
            ),
            "b"
        )
        XCTAssertEqual(
            KGAtlasPresentation.nextFindResultID(
                currentID: "c",
                move: .down,
                results: results
            ),
            "c"
        )
        XCTAssertEqual(
            KGAtlasPresentation.nextFindResultID(
                currentID: "a",
                move: .up,
                results: results
            ),
            "a"
        )
        XCTAssertEqual(
            KGAtlasPresentation.nextFindResultID(
                currentID: nil,
                move: .up,
                results: results
            ),
            "c"
        )
    }

    func testPriorityLabelsAreStableAcrossRefreshOrderAndSuppressNearbyCollisions() {
        let alpha = node(
            id: "alpha",
            name: "Alpha Platform",
            type: "project",
            importance: 9,
            chunks: 80,
            position: CGPoint(x: 200, y: 200)
        )
        let beta = node(
            id: "beta",
            name: "Beta Platform",
            type: "project",
            importance: 8,
            chunks: 60,
            position: CGPoint(x: 214, y: 204)
        )
        let gamma = node(
            id: "gamma",
            name: "Gamma Tool",
            type: "tool",
            importance: 7,
            chunks: 40,
            position: CGPoint(x: 520, y: 360)
        )

        let forward = KGAtlasPresentation.priorityLabelNodeIDs(
            nodes: [alpha, beta, gamma],
            selectedNodeID: "beta",
            scale: 1
        )
        let refreshed = KGAtlasPresentation.priorityLabelNodeIDs(
            nodes: [gamma, beta, alpha],
            selectedNodeID: "beta",
            scale: 1
        )

        XCTAssertEqual(forward, refreshed)
        XCTAssertTrue(forward.contains("beta"), "The selected label is always disclosed")
        XCTAssertTrue(forward.contains("gamma"), "A distant priority label remains visible")
        XCTAssertFalse(forward.contains("alpha"), "A nearby label yields to the selected entity")
    }

    func testCanvasEdgesKeepTopologyWithoutAlwaysOnRelationText() {
        let edge = KGEdge(
            sourceId: "project",
            targetId: "tool",
            relationType: "uses"
        )

        let lineOnlyEdge = KGAtlasPresentation.lineOnlyEdge(edge)

        XCTAssertEqual(lineOnlyEdge.sourceId, edge.sourceId)
        XCTAssertEqual(lineOnlyEdge.targetId, edge.targetId)
        XCTAssertEqual(lineOnlyEdge.relationType, "")
    }

    func testModeEffectsAndCanvasSummaryExplainNonSpatialControls() {
        XCTAssertEqual(
            KGAtlasPresentation.modeEffectDescription(
                mode: .importance,
                minimumImportance: 6,
                altitudeTier: .signal
            ),
            "Shows entities with importance 6/10 or higher. Selected and pinned entities remain visible."
        )
        XCTAssertEqual(
            KGAtlasPresentation.modeEffectDescription(
                mode: .tieredAltitude,
                minimumImportance: 6,
                altitudeTier: .signal
            ),
            "Shows Summit through Signal. Lower altitude reveals more entities."
        )
        XCTAssertEqual(
            KGAtlasPresentation.canvasAccessibilityValue(
                visibleEntityCount: 3,
                visibleRelationshipCount: 2
            ),
            "3 visible entities and 2 visible relationships"
        )
        XCTAssertEqual(
            KGAtlasPresentation.canvasAccessibilityHint,
            "Use Find entity to browse and select every visible entity without spatial pointing."
        )
    }

    func testFixedFixtureProducesStableNodePositionsRegardlessOfInputOrder() {
        let nodes = [
            node(id: "person", name: "Etan Heyman", type: "person", importance: 10),
            node(id: "project", name: "BrainLayer", type: "project", importance: 8),
            node(id: "tool", name: "MCP", type: "tool", importance: 7),
        ]
        let canvasSize = CGSize(width: 960, height: 640)

        let forward = Dictionary(uniqueKeysWithValues: KGAtlasLayout.seededNodes(
            nodes,
            canvasSize: canvasSize,
            mode: .tieredAltitude
        ).map { ($0.id, $0.position) })
        let refreshed = Dictionary(uniqueKeysWithValues: KGAtlasLayout.seededNodes(
            Array(nodes.reversed()),
            canvasSize: canvasSize,
            mode: .tieredAltitude
        ).map { ($0.id, $0.position) })

        XCTAssertEqual(forward, refreshed)
    }

    func testDetailDisclosuresStartCollapsedAndRemainDismissible() {
        XCTAssertFalse(KGSidebarView.relationsSectionDefaultExpanded)
        XCTAssertFalse(KGSidebarView.filesSectionDefaultExpanded)
        XCTAssertEqual(KGSidebarView.closeAccessibilityLabel, "Close entity details")
    }

    private func node(
        id: String,
        name: String,
        type: String,
        importance: Double,
        chunks: Int = 0,
        position: CGPoint = .zero
    ) -> KGNode {
        KGNode(
            id: id,
            name: name,
            entityType: type,
            importance: importance,
            linkedChunkCount: chunks,
            position: position
        )
    }
}
