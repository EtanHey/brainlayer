// StoreSecretScrubTests.swift — brain_store never writes a raw secret to disk.
//
// Before this, only the Python watcher scrubbed. BrainBar's store path wrote the
// raw content to the chunks row, both FTS tables and, while the DB was busy, the
// pending-stores queue file. Synthetic tokens only.

import CryptoKit
import Foundation
import SQLite3
import XCTest
@testable import BrainBar

final class StoreSecretScrubTests: XCTestCase {
    private let supabase = "sbp_" + String(repeating: "0", count: 40)
    private let github = "ghp_" + String(repeating: "0", count: 36)

    private var tempDir: URL!
    private var dbPath: String!

    override func setUpWithError() throws {
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("brainbar-scrub-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        dbPath = tempDir.appendingPathComponent("brainbar.db").path
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tempDir)
    }

    private func secretText() -> String {
        "deploy note: supabase \(supabase) and github \(github)"
    }

    // Byte-level: String.contains is grapheme aware, so a token followed by a
    // combining mark would not "contain" the token (#962 review N6).
    private func assertNoToken(_ blob: String, _ what: String, file: StaticString = #filePath, line: UInt = #line) {
        let bytes = Data(blob.utf8)
        XCTAssertNil(bytes.range(of: Data(supabase.utf8)), "\(what) holds the synthetic supabase token", file: file, line: line)
        XCTAssertNil(bytes.range(of: Data(github.utf8)), "\(what) holds the synthetic github token", file: file, line: line)
    }

    private func rows(_ sql: String) throws -> [[String]] {
        var handle: OpaquePointer?
        guard sqlite3_open_v2(dbPath, &handle, SQLITE_OPEN_READONLY, nil) == SQLITE_OK else {
            throw NSError(domain: "StoreSecretScrubTests", code: 1)
        }
        defer { sqlite3_close(handle) }
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(handle, sql, -1, &statement, nil) == SQLITE_OK else {
            throw NSError(domain: "StoreSecretScrubTests", code: 2, userInfo: [NSLocalizedDescriptionKey: sql])
        }
        defer { sqlite3_finalize(statement) }
        var out: [[String]] = []
        while sqlite3_step(statement) == SQLITE_ROW {
            var row: [String] = []
            for column in 0..<sqlite3_column_count(statement) {
                if let text = sqlite3_column_text(statement, column) {
                    row.append(String(cString: text))
                } else {
                    row.append("")
                }
            }
            out.append(row)
        }
        return out
    }

    func testStoreScrubsRowTagsFTSAndRecordsProviders() throws {
        let db = BrainDatabase(path: dbPath)
        defer { db.close() }

        let stored = try db.store(content: secretText(), tags: ["deploy", github], importance: 6, source: "mcp")

        let row = try XCTUnwrap(
            try rows("SELECT content, tags, metadata, preview_text, content_hash FROM chunks WHERE id = '\(stored.chunkID)'").first
        )
        assertNoToken(row.joined(separator: "|"), "chunks row")
        XCTAssertTrue(row[0].contains("[REDACTED:supabase]"), row[0])
        let metadata = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(row[2].utf8)) as? [String: Any]
        )
        XCTAssertEqual(metadata["secret_scrub_redactions"] as? [String], ["github", "supabase"])
        let expectedHash = SHA256.hash(data: Data(row[0].utf8)).map { String(format: "%02x", $0) }.joined()
        XCTAssertEqual(row[4], expectedHash, "content_hash must be computed from the scrubbed text")

        assertNoToken(try rows("SELECT * FROM chunks_fts").flatMap { $0 }.joined(separator: "|"), "chunks_fts")
        assertNoToken(
            try rows("SELECT * FROM chunks_fts_trigram").flatMap { $0 }.joined(separator: "|"),
            "chunks_fts_trigram"
        )
    }

    func testDeferredQueueNeverWritesTheRawTokenAndReplayKeepsProviders() throws {
        let db = BrainDatabase(path: dbPath)
        defer { db.close() }
        let queued = try db.queuePendingStore(
            content: secretText(),
            tags: [github],
            importance: 5,
            source: "mcp",
            chunkID: "brainbar-scrub-queued"
        )

        let queuePath = tempDir.appendingPathComponent("pending-stores.jsonl")
        let onDisk = try String(contentsOf: queuePath, encoding: .utf8)
        assertNoToken(onDisk, "pending-stores.jsonl")
        XCTAssertTrue(onDisk.contains("[REDACTED:supabase]"), onDisk)

        let flushed = db.flushPendingStores()
        XCTAssertEqual(flushed.map(\.storedChunk.chunkID), [queued.chunkID])
        assertNoToken(flushed.map { $0.content + $0.tags.joined() }.joined(), "flushed receipt")
        let row = try XCTUnwrap(try rows("SELECT content, metadata FROM chunks WHERE id = '\(queued.chunkID)'").first)
        assertNoToken(row.joined(separator: "|"), "replayed row")
        XCTAssertTrue(row[1].contains("secret_scrub_redactions"), row[1])
    }

    func testProvidersSurviveALineThatStaysQueuedThroughAFlush() throws {
        let db = BrainDatabase(path: dbPath)
        defer { db.close() }
        let queued = try db.queuePendingStore(
            content: secretText(),
            tags: [],
            importance: 5,
            source: "mcp",
            chunkID: "brainbar-scrub-kept-queued"
        )

        // First flush skips it, so its line is rewritten in place.
        XCTAssertTrue(db.flushPendingStores(excludingChunkIDs: [queued.chunkID]).isEmpty)
        let rewritten = try String(
            contentsOf: tempDir.appendingPathComponent("pending-stores.jsonl"),
            encoding: .utf8
        )
        XCTAssertTrue(rewritten.contains("secret_scrub_redactions"), rewritten)

        XCTAssertEqual(db.flushPendingStores().map(\.storedChunk.chunkID), [queued.chunkID])
        let row = try XCTUnwrap(try rows("SELECT metadata FROM chunks WHERE id = '\(queued.chunkID)'").first)
        XCTAssertTrue(row[0].contains("secret_scrub_redactions"), row[0])
    }

    func testLegacyRawQueueLineIsScrubbedOnReplay() throws {
        let queuePath = tempDir.appendingPathComponent("pending-stores.jsonl")
        let legacy: [String: Any] = [
            "content": secretText(),
            "tags": [github],
            "importance": 5,
            "source": "mcp",
            "chunk_id": "brainbar-legacy-raw",
        ]
        var line = try JSONSerialization.data(withJSONObject: legacy)
        line.append(0x0A)
        try line.write(to: queuePath)

        let db = BrainDatabase(path: dbPath)
        defer { db.close() }
        let flushed = db.flushPendingStores()

        XCTAssertEqual(flushed.map(\.storedChunk.chunkID), ["brainbar-legacy-raw"])
        let row = try XCTUnwrap(try rows("SELECT content, tags, metadata FROM chunks WHERE id = 'brainbar-legacy-raw'").first)
        assertNoToken(row.joined(separator: "|"), "row replayed from a legacy raw line")
        XCTAssertTrue(row[2].contains("secret_scrub_redactions"), row[2])
    }

    // #962 review B1: ICU's \b treats a combining mark as a word character, so the
    // provider pattern missed a token followed by U+0301 and the raw token was stored.
    func testStoreRedactsATokenFollowedByACombiningMark() throws {
        let db = BrainDatabase(path: dbPath)
        defer { db.close() }
        let openai = "sk-" + String(repeating: "0", count: 40)
        let stored = try db.store(content: "x \(openai)\u{0301} y", tags: [], importance: 5, source: "mcp")

        let row = try XCTUnwrap(try rows("SELECT content FROM chunks WHERE id = '\(stored.chunkID)'").first)
        XCTAssertNil(Data(row[0].utf8).range(of: Data(openai.utf8)), "raw token bytes reached the row")
        // Bytes, not String.contains: "]" + U+0301 is one grapheme, so the marker
        // would not "contain" as a String even though it is there.
        XCTAssertNotNil(Data(row[0].utf8).range(of: Data("[REDACTED:openai]".utf8)), row[0])
    }

    // #962 review N1: digest extracted entity spans from the raw content, so after
    // the scrub shortened the stored text the start_utf16 offsets pointed past it.
    func testDigestEntityOffsetsIndexTheStoredScrubbedText() throws {
        let db = BrainDatabase(path: dbPath)
        defer { db.close() }
        let openai = "sk-" + String(repeating: "0", count: 40)

        let result = try db.digest(
            content: "key \(openai) then Alice Wonderland arrived",
            title: "notes \(github)"
        )
        let chunkID = try XCTUnwrap(result["chunk_id"] as? String)
        let row = try XCTUnwrap(try rows("SELECT content, metadata FROM chunks WHERE id = '\(chunkID)'").first)
        assertNoToken(row[0], "digest row")
        XCTAssertNil(Data(row[0].utf8).range(of: Data(openai.utf8)), "digest row holds the openai token")
        let metadata = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(row[1].utf8)) as? [String: Any])
        XCTAssertEqual(metadata["secret_scrub_redactions"] as? [String], ["github", "openai"])
        let candidates = try XCTUnwrap(metadata["digest_entity_candidates"] as? [[String: Any]])
        XCTAssertFalse(candidates.isEmpty)
        let stored = row[0] as NSString
        for candidate in candidates {
            let start = try XCTUnwrap(candidate["start_utf16"] as? Int)
            let length = try XCTUnwrap(candidate["length_utf16"] as? Int)
            XCTAssertLessThanOrEqual(start + length, stored.length, "candidate span runs past the stored text")
            XCTAssertEqual(stored.substring(with: NSRange(location: start, length: length)), candidate["surface"] as? String)
        }
    }

    // #962 review N2: brain_update wrote tags raw.
    func testUpdateChunkScrubsTags() throws {
        let db = BrainDatabase(path: dbPath)
        defer { db.close() }
        let stored = try db.store(content: "an ordinary note", tags: ["a"], importance: 5, source: "mcp")

        try db.updateChunk(id: stored.chunkID, tags: ["deploy", github])

        let row = try XCTUnwrap(try rows("SELECT tags, metadata FROM chunks WHERE id = '\(stored.chunkID)'").first)
        assertNoToken(row.joined(separator: "|"), "updated row")
        XCTAssertTrue(row[0].contains("[REDACTED:github]"), row[0])
        XCTAssertTrue(row[1].contains("secret_scrub_redactions"), row[1])
        assertNoToken(try rows("SELECT * FROM chunks_fts").flatMap { $0 }.joined(separator: "|"), "chunks_fts after update")
    }

    // #962 review N3: the store receipt echoed the raw tags back into the transcript.
    func testStoreReceiptEchoesScrubbedTags() throws {
        let db = BrainDatabase(path: dbPath)
        defer { db.close() }
        let router = MCPRouter(profile: "full")
        router.setDatabase(db)
        let response = router.handle(
            [
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": ["name": "brain_store", "arguments": ["content": "note", "tags": ["deploy", github]]],
            ],
            session: router.makePaletteSession()
        )
        let result = try XCTUnwrap(response["result"] as? [String: Any])
        let content = try XCTUnwrap(result["content"] as? [[String: Any]])
        let text = content.compactMap { $0["text"] as? String }.joined()
        assertNoToken(text, "store receipt")
        XCTAssertTrue(text.contains("[REDACTED:github]"), text)
    }

    func testRouterStoreKeepsReceiptContractAndDedupsOnScrubbedText() throws {
        let db = BrainDatabase(path: dbPath)
        defer { db.close() }
        let router = MCPRouter(profile: "full")
        router.setDatabase(db)
        let session = router.makePaletteSession()

        func store(_ content: String, id: Int) throws -> [String: Any] {
            let response = router.handle(
                [
                    "jsonrpc": "2.0",
                    "id": id,
                    "method": "tools/call",
                    "params": ["name": "brain_store", "arguments": ["content": content, "tags": ["scrub"]]],
                ],
                session: session
            )
            return try XCTUnwrap(response["result"] as? [String: Any])
        }

        let first = try store("rotated ghp_" + String(repeating: "0", count: 36) + " today", id: 1)
        XCTAssertEqual(first["status"] as? String, "STORED")
        let chunkID = try XCTUnwrap(first["chunk_id"] as? String)

        // Differs only in the secret value, so it is the same memory once scrubbed.
        let second = try store("rotated ghp_" + String(repeating: "1", count: 36) + " today", id: 2)
        XCTAssertEqual(second["status"] as? String, "DUPLICATE")
        XCTAssertEqual(second["chunk_id"] as? String, chunkID)

        let stored = try rows("SELECT content FROM chunks").flatMap { $0 }
        XCTAssertEqual(stored.count, 1)
        XCTAssertEqual(stored.first, "rotated [REDACTED:github] today")
    }
}
