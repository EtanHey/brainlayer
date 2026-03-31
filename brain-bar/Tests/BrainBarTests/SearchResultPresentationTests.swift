import XCTest
@testable import BrainBar

final class SearchResultPresentationTests: XCTestCase {
    func testSearchResultBuildsCompactMetadataRow() {
        let result = SearchResult(
            chunkID: "rt-123",
            score: 0.87,
            project: "brainlayer",
            date: "2026-03-29T19:30:00Z",
            summary: "BrainBar native formatter parity",
            snippet: "BrainBar native formatter parity",
            importance: 8,
            tags: ["swift", "mcp"]
        )

        XCTAssertEqual(result.displayText, "BrainBar native formatter parity")
        XCTAssertEqual(result.compactMetadata, "brainlayer • 2026-03-29 • imp 8 • score 0.87")
        XCTAssertEqual(result.tagSummary, "swift, mcp")
    }

    func testSearchResultFallsBackToSnippetAndOmitsMissingMetadata() {
        let result = SearchResult(
            chunkID: "rt-456",
            score: 0,
            project: "",
            date: "",
            summary: "",
            snippet: "Fallback snippet text",
            importance: nil,
            tags: []
        )

        XCTAssertEqual(result.displayText, "Fallback snippet text")
        XCTAssertEqual(result.compactMetadata, "score 0.00")
        XCTAssertNil(result.tagSummary)
    }
}
