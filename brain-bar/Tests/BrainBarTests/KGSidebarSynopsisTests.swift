import XCTest
@testable import BrainBar

final class KGSidebarSynopsisTests: XCTestCase {
    // QA #17: Etan's own node showed "no entity description is stored yet",
    // reading as broken even though it had rich relations/chunks. When data
    // exists, point at it as the drill-down; only call it empty when truly empty.

    func testFallbackPointsAtRelationsAndChunksWhenBothPresent() {
        let text = KGSidebarView.synopsisFallbackText(
            name: "Etan Heyman",
            relationCount: 6,
            chunkCount: 42
        )

        XCTAssertEqual(text, "Explore Etan Heyman through its 6 relations and 42 linked chunks below.")
        XCTAssertFalse(text.lowercased().contains("stored yet"))
    }

    func testFallbackSingularizesCounts() {
        let text = KGSidebarView.synopsisFallbackText(
            name: "Domica",
            relationCount: 1,
            chunkCount: 1
        )

        XCTAssertEqual(text, "Explore Domica through its 1 relation and 1 linked chunk below.")
    }

    func testFallbackUsesOnlyAvailableSignal() {
        let onlyChunks = KGSidebarView.synopsisFallbackText(name: "Claude Code", relationCount: 0, chunkCount: 3)
        XCTAssertEqual(onlyChunks, "Explore Claude Code through its 3 linked chunks below.")

        let onlyRelations = KGSidebarView.synopsisFallbackText(name: "Claude Code", relationCount: 2, chunkCount: 0)
        XCTAssertEqual(onlyRelations, "Explore Claude Code through its 2 relations below.")
    }

    func testFallbackFallsBackToEmptyCopyWhenNothingLinked() {
        let text = KGSidebarView.synopsisFallbackText(name: "Ghost", relationCount: 0, chunkCount: 0)

        XCTAssertEqual(text, "No synopsis yet. This entity has no relations or linked chunks to summarize.")
        XCTAssertFalse(text.lowercased().contains("stored yet"))
    }

    func testFallbackHandlesBlankName() {
        let text = KGSidebarView.synopsisFallbackText(name: "   ", relationCount: 3, chunkCount: 0)

        XCTAssertEqual(text, "Explore this entity through its 3 relations below.")
    }
}
