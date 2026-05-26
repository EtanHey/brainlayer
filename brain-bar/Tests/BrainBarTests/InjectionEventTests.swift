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

    func testDisplayTitleUsesChunkSummaryBeforeSourcePrompt() {
        let event = InjectionEvent(
            id: 12,
            sessionID: "session-abcdef",
            timestamp: "2026-03-31T04:50:00.000Z",
            query: "cron loop prompt with operational plumbing",
            chunkIDs: ["chunk-1"],
            tokenCount: 42,
            chunks: [
                InjectionChunk(
                    id: "chunk-1",
                    content: "The dashboard coverage bug comes from excluding terminal duplicate chunks.",
                    summary: "Coverage bug excludes terminal duplicate chunks",
                    source: "precompact-hook",
                    sourceFile: "precompact:abc123",
                    tags: ["pr-merge"],
                    contentType: "memory"
                )
            ]
        )

        XCTAssertEqual(event.displayTitle, "Coverage bug excludes terminal duplicate chunks")
        XCTAssertEqual(event.triggeredByText, "Triggered by: cron loop prompt with operational plumbing")
        XCTAssertEqual(event.primaryKind.label, "Checkpoint")
        XCTAssertEqual(event.primaryKind.glyph, "🏷")
        XCTAssertEqual(event.modalTitle, "Memory Checkpoint")
    }

    func testKindClassificationCoversKnownSources() {
        XCTAssertEqual(InjectionKind.classify(source: "precompact-hook", sourceFile: "", tags: [], content: "").label, "Memory Checkpoint")
        XCTAssertEqual(InjectionKind.classify(source: "brain_store", sourceFile: "", tags: [], content: "").label, "Stored Memory")
        XCTAssertEqual(InjectionKind.classify(source: "claude_code", sourceFile: "", tags: [], content: "").label, "Realtime Capture")
        XCTAssertEqual(InjectionKind.classify(source: "youtube", sourceFile: "", tags: [], content: "").label, "Video Knowledge")
        XCTAssertEqual(InjectionKind.classify(source: "codex_cli", sourceFile: "", tags: [], content: "").label, "Tool Session")
        XCTAssertEqual(InjectionKind.classify(source: "mcp", sourceFile: "", tags: ["pr-merge"], content: "").label, "Checkpoint")
        XCTAssertEqual(InjectionKind.classify(source: "", sourceFile: "", tags: [], content: "[CHECKPOINT] merged PR").label, "Checkpoint")
    }

    func testKindPaletteIndexUsesDeclarationOrder() {
        XCTAssertEqual(InjectionKind.memoryCheckpoint.paletteIndex, 0)
        XCTAssertEqual(InjectionKind.other.paletteIndex, InjectionKind.allCases.count - 1)
    }

    func testSnapshotFiltersByType() {
        let now = Date(timeIntervalSince1970: 1_780_000_000)
        let timestamp = ISO8601DateFormatter().string(from: now.addingTimeInterval(-60))
        let checkpoint = InjectionEvent(
            id: 1,
            sessionID: "s1",
            timestamp: timestamp,
            query: "checkpoint query",
            chunkIDs: ["c1"],
            tokenCount: 10,
            chunks: [
                InjectionChunk(
                    id: "c1",
                    content: "[CHECKPOINT] merged item",
                    summary: "",
                    source: "mcp",
                    sourceFile: "",
                    tags: ["pr-merge"],
                    contentType: "memory"
                )
            ]
        )
        let realtime = InjectionEvent(
            id: 2,
            sessionID: "s2",
            timestamp: timestamp,
            query: "realtime query",
            chunkIDs: ["c2"],
            tokenCount: 10,
            chunks: [
                InjectionChunk(
                    id: "c2",
                    content: "Fresh Claude session capture",
                    summary: "",
                    source: "claude_code",
                    sourceFile: "",
                    tags: [],
                    contentType: "user_message"
                )
            ]
        )

        let snapshot = InjectionPresentation.snapshot(
            events: [checkpoint, realtime],
            filterText: "",
            typeFilter: .checkpoint,
            now: now
        )

        XCTAssertEqual(snapshot.filteredEvents.map(\.id), [1])
        XCTAssertEqual(snapshot.summary.queryCount, 1)
    }

    func testSnapshotTypeFilterMatchesAnyRetrievedChunkKind() {
        let now = Date(timeIntervalSince1970: 1_780_000_000)
        let timestamp = ISO8601DateFormatter().string(from: now.addingTimeInterval(-60))
        let mixed = InjectionEvent(
            id: 3,
            sessionID: "s3",
            timestamp: timestamp,
            query: "mixed query",
            chunkIDs: ["stored", "video"],
            tokenCount: 10,
            chunks: [
                InjectionChunk(
                    id: "stored",
                    content: "Stored memory first",
                    summary: "",
                    source: "mcp",
                    sourceFile: "",
                    tags: [],
                    contentType: "memory"
                ),
                InjectionChunk(
                    id: "video",
                    content: "Video knowledge second",
                    summary: "",
                    source: "youtube",
                    sourceFile: "",
                    tags: [],
                    contentType: "media"
                )
            ]
        )

        let snapshot = InjectionPresentation.snapshot(
            events: [mixed],
            filterText: "",
            typeFilter: .video,
            now: now
        )

        XCTAssertEqual(snapshot.filteredEvents.map(\.id), [3])
    }
}
