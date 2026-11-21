import SQLite3
import XCTest
@testable import BrainBar

final class ReadAliasPrecedenceTests: XCTestCase {
    func testNativeReadsKeepOwnRowAndOriginConversationWithoutFollowingAliases() throws {
        for retired in ["active", "archived_at", "superseded_by", "aggregated_into", "invalid_at"] {
            let path = brainBarTestTemporaryPath() + "read-alias-\(UUID().uuidString).db"
            let database = BrainDatabase(path: path)
            defer { database.close(); removeBrainBarTestDatabase(path) }
            try database.insertChunk(id: "survivor", content: "The preserved survivor", sessionId: "survivor-session",
                                     project: "fixture", contentType: "user_message", importance: 5)
            try database.insertChunk(id: "own", content: "The original session message", sessionId: "own-session",
                                     project: "fixture", contentType: "user_message", importance: 5)
            var handle: OpaquePointer?
            XCTAssertEqual(sqlite3_open(path, &handle), SQLITE_OK)
            let db = try XCTUnwrap(handle)
            defer { sqlite3_close(db) }
            XCTAssertEqual(sqlite3_exec(db, """
                CREATE TABLE chunk_id_alias(old_chunk_id TEXT PRIMARY KEY, canonical_chunk_id TEXT, deprecated_at TEXT);
                INSERT INTO chunk_id_alias VALUES ('own','survivor',datetime('now'));
                INSERT INTO chunk_id_alias VALUES ('folded-only','survivor',datetime('now'));
                """, nil, nil, nil), SQLITE_OK)
            if retired != "active" {
                var columns: OpaquePointer?
                XCTAssertEqual(sqlite3_prepare_v2(db, "PRAGMA table_info(chunks)", -1, &columns, nil), SQLITE_OK)
                var present = false
                while sqlite3_step(columns) == SQLITE_ROW {
                    if let name = sqlite3_column_text(columns, 1), String(cString: name) == retired { present = true }
                }
                sqlite3_finalize(columns)
                if !present { XCTAssertEqual(sqlite3_exec(db, "ALTER TABLE chunks ADD COLUMN \(retired) TEXT", nil, nil, nil), SQLITE_OK) }
                XCTAssertEqual(sqlite3_exec(db, "UPDATE chunks SET \(retired)='retired' WHERE id='own'", nil, nil, nil), SQLITE_OK)
            }
            let expected = "own"
            XCTAssertEqual(try database.getChunk(id: "own")?["id"] as? String, expected, retired)
            let expanded = try database.expandChunk(id: "own", before: 0, after: 0)
            XCTAssertEqual((expanded["target"] as? [String: Any])?["chunk_id"] as? String, expected, retired)
            XCTAssertEqual((expanded["target"] as? [String: Any])?["session_id"] as? String, "own-session", retired)
            let recalled = try database.recallSession(sessionId: "own-session")
            XCTAssertEqual(recalled.first?["chunk_id"] as? String, expected, retired)
            XCTAssertEqual(recalled.first?["content"] as? String, "The original session message", retired)
            XCTAssertNil(try database.getChunk(id: "folded-only"), retired)
            XCTAssertFalse(try database.archiveChunk(id: "folded-only"), retired)
            XCTAssertFalse(try database.supersedeChunk(oldChunkID: "folded-only", newChunkID: "survivor"), retired)
            let survivor = try XCTUnwrap(database.getChunk(id: "survivor"))
            XCTAssertNil(survivor["archived_at"] as? String, retired)
            XCTAssertNil(survivor["superseded_by"] as? String, retired)
            var stmt: OpaquePointer?
            XCTAssertEqual(sqlite3_prepare_v2(db, "SELECT canonical_chunk_id FROM chunk_id_alias WHERE old_chunk_id='own'", -1, &stmt, nil), SQLITE_OK)
            XCTAssertEqual(sqlite3_step(stmt), SQLITE_ROW)
            XCTAssertEqual(String(cString: sqlite3_column_text(stmt, 0)), "survivor")
            sqlite3_finalize(stmt)
        }
    }
}
