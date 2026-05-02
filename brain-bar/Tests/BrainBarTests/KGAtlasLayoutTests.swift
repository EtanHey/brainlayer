import XCTest
@testable import BrainBar

final class KGAtlasLayoutTests: XCTestCase {
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
}
