import XCTest
@testable import BrainBar

final class KGAtlasLayoutTests: XCTestCase {
    deinit {}

    func testCanvasMetricsSubtractSidebarAndCanvasPadding() {
        // Regression guard: Macroscope PR #315 flagged that resizing used the
        // outer window size, which placed graph nodes behind the visible sidebar.
        let size = KGCanvasMetrics.drawableSize(
            windowSize: CGSize(width: 1_200, height: 800),
            sidebarVisible: true
        )

        XCTAssertEqual(size.width, 844)
        XCTAssertEqual(size.height, 764)
    }

    func testSeededNodesPlaceEntityTypesIntoStableRegions() {
        let nodes = [
            KGNode(id: "p1", name: "Etan Heyman", entityType: "person", importance: 9, position: .zero),
            KGNode(id: "p2", name: "David Heyman", entityType: "person", importance: 6, position: .zero),
            KGNode(id: "pr1", name: "brainlayer", entityType: "project", importance: 8, position: .zero),
            KGNode(id: "t1", name: "mcp", entityType: "tool", importance: 7, position: .zero),
        ]

        let seeded = KGAtlasLayout.seededNodes(
            nodes,
            canvasSize: CGSize(width: 960, height: 640)
        )

        let personNodes = seeded.filter { $0.entityType == "person" }
        let projectNode = seeded.first { $0.entityType == "project" }!
        let toolNode = seeded.first { $0.entityType == "tool" }!

        XCTAssertEqual(personNodes.count, 2)
        XCTAssertLessThan(personNodes[0].position.x, projectNode.position.x)
        XCTAssertLessThan(projectNode.position.x, toolNode.position.x)
        XCTAssertLessThan(abs(personNodes[0].position.x - personNodes[1].position.x), 120)
    }

    func testSeededOwnerEntityStartsBelowAtlasToolbar() {
        let seeded = KGAtlasLayout.seededNodes(
            [
                KGNode(
                    id: "owner",
                    name: "Etan Heyman",
                    entityType: "person",
                    importance: 0.5,
                    position: .zero
                ),
            ],
            canvasSize: CGSize(width: 860, height: 500)
        )

        let owner = seeded[0]
        XCTAssertGreaterThan(owner.position.y, 240)
        XCTAssertLessThan(owner.position.x, 300)
    }

    func testTieredAltitudeLayoutPlacesNodesInDescendingRows() {
        let seeded = KGAtlasLayout.seededNodes(
            [
                KGNode(id: "owner", name: "Etan Heyman", entityType: "person", importance: 10, position: .zero),
                KGNode(id: "claude", name: "Claude Code", entityType: "agent", importance: 6, position: .zero),
                KGNode(id: "brainlayer", name: "brainlayer", entityType: "project", importance: 8, position: .zero),
                KGNode(id: "scratch", name: "Scratch", entityType: "topic", importance: 1, position: .zero),
            ],
            canvasSize: CGSize(width: 1_000, height: 800),
            mode: .tieredAltitude
        )

        let owner = seeded.first { $0.id == "owner" }!
        let claude = seeded.first { $0.id == "claude" }!
        let brainlayer = seeded.first { $0.id == "brainlayer" }!
        let scratch = seeded.first { $0.id == "scratch" }!

        XCTAssertLessThan(owner.position.y, brainlayer.position.y)
        XCTAssertLessThan(claude.position.y, brainlayer.position.y)
        XCTAssertLessThan(brainlayer.position.y, scratch.position.y)
    }
}
