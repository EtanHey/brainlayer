// StoreResponseClarityTests.swift — every brain_store outcome must return an
// unambiguous, action-guiding response.
//
// Etan, 2026-08-19: "make sure the tool responses explain the agents exactly
// what's going on so they don't just duplicate — if something is deferred, it
// means it will be stored in the future."
//
// Each response must carry:
//   (a) the OUTCOME WORD, so an agent can branch without parsing prose;
//   (b) the CANONICAL chunk_id it resolved to — or an explicit "no chunk_id"
//       for the outcomes that store nothing;
//   (c) the INSTRUCTION — above all, whether re-storing is right or wrong.
//
// Outcomes reachable on the BrainBar path:
//   STORED / DUPLICATE / DEFERRED / ERROR
// MERGED is Python-only: BrainBar deliberately writes NULL simhash columns
// (see the comment in BrainDatabase.store), so it has no near-duplicate merge
// to report. Its dedupe is exact-content-hash within a conversation, which is
// DUPLICATE, and the formatter still renders MERGED so both paths share one
// vocabulary.

import XCTest
import SQLite3
@testable import BrainBar

final class StoreResponseClarityTests: XCTestCase {

    // MARK: - Formatter, one test per outcome

    func testStoredNewStatesOutcomeIDAndThatItIsNew() {
        let text = Formatters.formatStoreResult(chunkId: "brainbar-abc123", outcome: .stored, useColor: false)
        XCTAssertTrue(text.contains("STORED"), text)
        XCTAssertTrue(text.contains("brainbar-abc123"), text)
        XCTAssertTrue(text.lowercased().contains("new"), text)
        XCTAssertFalse(text.contains("DUPLICATE"), text)
    }

    func testDuplicateStatesOutcomeCanonicalIDAndDoNotReStore() {
        let text = Formatters.formatStoreResult(chunkId: "brainbar-canon1", outcome: .duplicate, useColor: false)
        XCTAssertTrue(text.contains("DUPLICATE"), text)
        XCTAssertTrue(text.contains("brainbar-canon1"), text)
        XCTAssertTrue(text.lowercased().contains("already stored"), text)
        XCTAssertTrue(text.lowercased().contains("do not re-store"), text)
    }

    func testMergedStatesOutcomeCanonicalIDAndDoNotReStore() {
        let text = Formatters.formatStoreResult(chunkId: "brainbar-canon2", outcome: .merged, useColor: false)
        XCTAssertTrue(text.contains("MERGED"), text)
        XCTAssertTrue(text.contains("brainbar-canon2"), text)
        XCTAssertTrue(text.lowercased().contains("merged into"), text)
        XCTAssertTrue(text.lowercased().contains("do not re-store"), text)
    }

    func testDeferredStatesOutcomePromisedIDAndThatItWillBeStored() {
        let text = Formatters.formatStoreResult(chunkId: "brainbar-def456", queued: true, useColor: false)
        // The prefix is "STORED (deferred)", not a bare "DEFERRED:" -- Etan retired
        // that on 2026-08-09 because it read as failure and agents re-stored on it.
        // Machines branch on the structured status, which is "DEFERRED".
        XCTAssertTrue(text.contains("STORED (deferred)"), text)
        XCTAssertFalse(text.contains("DEFERRED:"), text)
        XCTAssertTrue(text.contains("brainbar-def456"), text)
        XCTAssertTrue(text.lowercased().contains("will be stored"), text)
        XCTAssertTrue(text.lowercased().contains("do not retry"), text)
        XCTAssertTrue(text.lowercased().contains("fallback"), text)
    }

    func testDeferredReportsTheReasonItWasDeferred() {
        let text = Formatters.formatStoreResult(
            chunkId: "brainbar-def456",
            queued: true,
            queuedReason: "SCHEMA_FINGERPRINT_MISMATCH",
            useColor: false
        )
        XCTAssertTrue(text.contains("SCHEMA_FINGERPRINT_MISMATCH"), text)
    }

    func testErrorStatesOutcomeAbsenceOfIDAndThatNothingWasStored() {
        let text = Formatters.formatStoreFailure(outcome: .error, reason: "disk I/O error", useColor: false)
        XCTAssertTrue(text.contains("ERROR"), text)
        XCTAssertTrue(text.lowercased().contains("no chunk_id"), text)
        XCTAssertTrue(text.contains("disk I/O error"), text)
        XCTAssertTrue(text.lowercased().contains("not stored"), text)
    }

    func testErrorNeverImpliesADeferredWrite() {
        // An error is the one outcome where the memory is genuinely lost. If it
        // read like DEFERRED the agent would drop the content on the floor.
        let text = Formatters.formatStoreFailure(outcome: .error, reason: "boom", useColor: false)
        XCTAssertFalse(text.lowercased().contains("deferred"), text)
        XCTAssertFalse(text.lowercased().contains("will be stored"), text)
    }

    func testRejectedStatesOutcomeAbsenceOfIDAndDoNotRetry() {
        let text = Formatters.formatStoreFailure(
            outcome: .rejected,
            reason: "content must be non-empty",
            useColor: false
        )
        XCTAssertTrue(text.contains("REJECTED"), text)
        XCTAssertTrue(text.lowercased().contains("no chunk_id"), text)
        XCTAssertTrue(text.lowercased().contains("do not retry"), text)
    }

    // MARK: - Router: a real store, then a real duplicate

    func testFirstStoreReportsSTOREDAndSecondIdenticalStoreReportsDUPLICATE() throws {
        let tempDir = makeClarityTempDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let dbPath = tempDir.appendingPathComponent("brainbar.db").path
        let db = BrainDatabase(path: dbPath)
        defer { db.close() }

        let router = MCPRouter(profile: "full")
        router.setDatabase(db)
        let session = router.makePaletteSession()
        let arguments: [String: Any] = [
            "content": "The canonical DB lives at ~/.local/share/brainlayer/brainlayer.db",
            "tags": ["clarity"],
            "importance": 6,
        ]

        let firstResponse = router.handle(
            clarityToolCall(id: 901, name: "brain_store", arguments: arguments),
            session: session
        )
        let first = try XCTUnwrap(firstResponse["result"] as? [String: Any])
        let firstText = try clarityText(first)
        XCTAssertEqual(first["status"] as? String, "STORED")
        XCTAssertEqual(first["stored_new"] as? Bool, true)
        XCTAssertTrue(firstText.contains("STORED"), firstText)
        let canonicalID = try XCTUnwrap(first["chunk_id"] as? String)
        XCTAssertTrue(firstText.contains(canonicalID), firstText)

        let secondResponse = router.handle(
            clarityToolCall(id: 902, name: "brain_store", arguments: arguments),
            session: session
        )
        let second = try XCTUnwrap(secondResponse["result"] as? [String: Any])
        let secondText = try clarityText(second)

        XCTAssertEqual(second["status"] as? String, "DUPLICATE")
        XCTAssertEqual(second["stored_new"] as? Bool, false)
        XCTAssertEqual(
            second["chunk_id"] as? String,
            canonicalID,
            "a duplicate must resolve to the canonical chunk the agent should reference"
        )
        XCTAssertTrue(secondText.contains("DUPLICATE"), secondText)
        XCTAssertTrue(secondText.contains(canonicalID), secondText)
        XCTAssertTrue(secondText.lowercased().contains("do not re-store"), secondText)

        XCTAssertEqual(
            try clarityChunkCount(path: dbPath),
            1,
            "the duplicate must not have written a second row"
        )
    }

    func testErrorResponseNamesTheOutcomeAndSaysNothingWasStored() throws {
        let router = MCPRouter(profile: "full")
        let session = router.makePaletteSession()

        // No `content` argument — the missing-parameter path.
        let response = router.handle(
            clarityToolCall(id: 903, name: "brain_store", arguments: ["tags": ["clarity"]]),
            session: session
        )
        let result = try XCTUnwrap(response["result"] as? [String: Any])
        XCTAssertEqual(result["isError"] as? Bool, true)
        let content = try XCTUnwrap(result["content"] as? [[String: Any]])
        let text = try XCTUnwrap(content.first?["text"] as? String)

        XCTAssertTrue(text.contains("ERROR") || text.contains("REJECTED"), text)
        XCTAssertTrue(text.lowercased().contains("no chunk_id"), text)
        XCTAssertTrue(
            text.lowercased().contains("nothing was stored") || text.lowercased().contains("not stored"),
            text
        )
    }

    // MARK: - Tool description must list the outcomes up front

    /// The CORE profile is what agents see by default -- the full palette only
    /// appears after expand_palette. A terse core description that does not name
    /// the outcomes leaves the default agent learning DUPLICATE exists only by
    /// receiving one, which is one redundant call too late.
    func testCoreProfileBrainStoreDescriptionEnumeratesEveryOutcome() throws {
        let description = try brainStoreDescription(profile: "core")
        for outcome in ["STORED", "DUPLICATE", "MERGED", "DEFERRED", "REJECTED", "ERROR"] {
            XCTAssertTrue(
                description.contains(outcome),
                "the default-profile brain_store description must name \(outcome); got: \(description)"
            )
        }
        XCTAssertTrue(
            description.lowercased().contains("do not re-store"),
            "the description must state the rule that makes DUPLICATE/MERGED actionable: \(description)"
        )
    }

    func testCoreProfileDescriptionStaysTerse() throws {
        // The core palette is deliberately short on tokens. Naming the outcomes
        // must not turn it into the full paragraph.
        XCTAssertLessThan(try brainStoreDescription(profile: "core").count, 320)
    }

    private func brainStoreDescription(profile: String) throws -> String {
        let router = MCPRouter(profile: profile)
        let response = router.handle(
            ["jsonrpc": "2.0", "id": 905, "method": "tools/list"],
            session: router.makePaletteSession()
        )
        let result = try XCTUnwrap(response["result"] as? [String: Any])
        let tools = try XCTUnwrap(result["tools"] as? [[String: Any]])
        let store = try XCTUnwrap(tools.first { ($0["name"] as? String) == "brain_store" })
        return try XCTUnwrap(store["description"] as? String)
    }

    func testBrainStoreDescriptionEnumeratesEveryOutcome() throws {
        let router = MCPRouter(profile: "full")
        let response = router.handle(
            ["jsonrpc": "2.0", "id": 904, "method": "tools/list"],
            session: router.makePaletteSession()
        )
        let result = try XCTUnwrap(response["result"] as? [String: Any])
        let tools = try XCTUnwrap(result["tools"] as? [[String: Any]])
        let store = try XCTUnwrap(tools.first { ($0["name"] as? String) == "brain_store" })
        let description = try XCTUnwrap(store["description"] as? String)

        for outcome in ["STORED", "DUPLICATE", "MERGED", "DEFERRED", "REJECTED", "ERROR"] {
            XCTAssertTrue(
                description.contains(outcome),
                "brain_store description must name the \(outcome) outcome so agents know it can happen"
            )
        }
        XCTAssertTrue(
            description.lowercased().contains("do not re-store"),
            "the description must state the rule that makes DUPLICATE/MERGED actionable"
        )
    }
}

// MARK: - File-private helpers

private func makeClarityTempDirectory() -> URL {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
}

private func clarityToolCall(id: Int, name: String, arguments: [String: Any]) -> [String: Any] {
    [
        "jsonrpc": "2.0",
        "id": id,
        "method": "tools/call",
        "params": [
            "name": name,
            "arguments": arguments
        ] as [String: Any]
    ]
}

private func clarityText(_ result: [String: Any]) throws -> String {
    XCTAssertNil(result["isError"], String(describing: result))
    let content = try XCTUnwrap(result["content"] as? [[String: Any]])
    return try XCTUnwrap(content.first?["text"] as? String)
}

private func clarityChunkCount(path: String) throws -> Int {
    var db: OpaquePointer?
    let rc = sqlite3_open_v2(path, &db, SQLITE_OPEN_READONLY | SQLITE_OPEN_FULLMUTEX, nil)
    guard rc == SQLITE_OK, let db else {
        throw NSError(domain: "StoreResponseClarityTests", code: Int(rc))
    }
    defer { sqlite3_close(db) }

    var stmt: OpaquePointer?
    guard sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM chunks", -1, &stmt, nil) == SQLITE_OK else {
        throw NSError(domain: "StoreResponseClarityTests", code: 1)
    }
    defer { sqlite3_finalize(stmt) }
    guard sqlite3_step(stmt) == SQLITE_ROW else {
        throw NSError(domain: "StoreResponseClarityTests", code: 2)
    }
    return Int(sqlite3_column_int64(stmt, 0))
}
