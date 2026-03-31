import XCTest
@testable import BrainBar

final class InjectionEventTests: XCTestCase {
    func testSummaryUsesQueryTokenCountAndChunkCount() {
        let event = InjectionEvent(
            id: 42,
            sessionID: "sess-1",
            timestamp: "2026-03-31T04:50:00.000Z",
            query: "voicebar daemon sleep recovery",
            chunkIDs: ["c1", "c2", "c3"],
            tokenCount: 128
        )

        XCTAssertEqual(event.chunkCount, 3)
        XCTAssertTrue(event.summaryLine.contains("voicebar daemon sleep recovery"))
        XCTAssertTrue(event.summaryLine.contains("3 chunks"))
        XCTAssertTrue(event.summaryLine.contains("128 tok"))
    }

    func testChunkIDsDecodeFromJSONStringPayload() throws {
        let event = try InjectionEvent(
            row: [
                "id": 7,
                "session_id": "sess-2",
                "timestamp": "2026-03-31T04:50:00.000Z",
                "query": "brainbar search window",
                "chunk_ids": "[\"chunk-a\",\"chunk-b\"]",
                "token_count": 64
            ]
        )

        XCTAssertEqual(event.chunkIDs, ["chunk-a", "chunk-b"])
        XCTAssertEqual(event.tokenCount, 64)
    }
}
