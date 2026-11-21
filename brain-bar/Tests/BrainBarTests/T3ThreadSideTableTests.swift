// T3ThreadSideTableTests.swift — T3 PR 3.1: brain_expand shows a session chunk's T3 thread.
//
// The side tables are written by Python t3-ingest; BrainBar only reads them. These tests
// create them with the same DDL as `src/brainlayer/t3_thread_tables.py`.

import Darwin
import XCTest
import SQLite3
@testable import BrainBar

final class T3ThreadSideTableTests: XCTestCase {
    private var path: String!
    private var database: BrainDatabase!

    override func setUpWithError() throws {
        path = brainBarTestTemporaryPath() + "brainbar-t3-side-\(UUID().uuidString).db"
        database = BrainDatabase(path: path)
        try database.insertChunk(
            id: "session-chunk", content: "A session line from the Claude run",
            sessionId: "sess-claude", project: "brainlayer", contentType: "assistant_text", importance: 5
        )
    }

    override func tearDownWithError() throws {
        database.close()
        removeBrainBarTestDatabase(path)
    }

    private func createSideTables() throws {
        try exec("""
            CREATE TABLE IF NOT EXISTS t3_threads (
                thread_id TEXT PRIMARY KEY, state_db TEXT NOT NULL, title TEXT NOT NULL, project_id TEXT,
                project_name TEXT, forked_from TEXT, created_at TEXT NOT NULL, deleted_at TEXT,
                text_owner TEXT NOT NULL CHECK (text_owner IN ('session', 't3')), owner_reason TEXT NOT NULL,
                last_message_at TEXT, session_high_water TEXT, updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS t3_thread_sessions (
                thread_id TEXT NOT NULL, provider_thread_id TEXT NOT NULL, driver TEXT NOT NULL,
                provider TEXT NOT NULL, native_session_id TEXT, conversation_id TEXT,
                is_active INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL,
                PRIMARY KEY (thread_id, provider_thread_id));
            CREATE INDEX IF NOT EXISTS idx_t3_thread_sessions_conversation ON t3_thread_sessions(conversation_id);
        """)
    }

    private func addThread(
        _ id: String, title: String, project: String?, createdAt: String = "2026-11-19T00:00:00Z",
        deletedAt: String? = nil, conversation: String? = nil, active: Bool = true
    ) throws {
        let projectSQL = project.map { "'\($0)'" } ?? "NULL"
        let deletedSQL = deletedAt.map { "'\($0)'" } ?? "NULL"
        try exec("""
            INSERT INTO t3_threads VALUES ('\(id)', 'statev2.sqlite', '\(title)', 'p-\(id)', \(projectSQL), NULL,
                '\(createdAt)', \(deletedSQL), 'session', 'session_chunks', NULL, NULL, '2026-11-19T00:00:00Z');
        """)
        if let conversation {
            try exec("""
                INSERT INTO t3_thread_sessions VALUES ('\(id)', 'pt-\(id)', 'claudeAgent', 'claudeAgent',
                    '\(conversation)', '\(conversation)', \(active ? 1 : 0), '2026-11-19T00:00:00Z');
            """)
        }
    }

    private func exec(_ sql: String) throws {
        var handle: OpaquePointer?
        guard sqlite3_open_v2(path, &handle, SQLITE_OPEN_READWRITE, nil) == SQLITE_OK, let handle else {
            throw NSError(domain: "T3ThreadSideTableTests", code: 1)
        }
        defer { sqlite3_close(handle) }
        let rc = sqlite3_exec(handle, sql, nil, nil, nil)
        guard rc == SQLITE_OK else {
            throw NSError(domain: "T3ThreadSideTableTests", code: Int(rc),
                          userInfo: [NSLocalizedDescriptionKey: String(cString: sqlite3_errmsg(handle))])
        }
    }

    // Fail-first: on main, expandChunk never returns a t3_thread.
    func testExpandOfASessionChunkReturnsItsT3Thread() throws {
        try createSideTables()
        try addThread("th-claude", title: "Claude thread", project: "brainlayer", conversation: "sess-claude")

        let expanded = try database.expandChunk(id: "session-chunk", before: 0, after: 0)
        let thread = try XCTUnwrap(expanded["t3_thread"] as? [String: Any])
        XCTAssertEqual(thread["id"] as? String, "th-claude")
        XCTAssertEqual(thread["title"] as? String, "Claude thread")
        XCTAssertEqual(thread["project"] as? String, "brainlayer")
        XCTAssertEqual(thread["link"] as? String, "t3://thread/th-claude")
    }

    func testExpandWithoutSideTablesStillWorksAndHasNoThread() throws {
        let expanded = try database.expandChunk(id: "session-chunk", before: 0, after: 0)
        XCTAssertNil(expanded["t3_thread"])
        XCTAssertEqual((expanded["target"] as? [String: Any])?["chunk_id"] as? String, "session-chunk")
    }

    func testExpandOfAnUnmappedSessionHasNoThread() throws {
        try createSideTables()
        try addThread("th-other", title: "Other", project: "golems", conversation: "sess-other")
        let expanded = try database.expandChunk(id: "session-chunk", before: 0, after: 0)
        XCTAssertNil(expanded["t3_thread"])
    }

    func testProjectFallsBackToProjectIdWhenT3HasNoProjectName() throws {
        try createSideTables()
        try addThread("th-claude", title: "No project row", project: nil, conversation: "sess-claude")
        let thread = try XCTUnwrap(
            try database.expandChunk(id: "session-chunk", before: 0, after: 0)["t3_thread"] as? [String: Any]
        )
        XCTAssertEqual(thread["project"] as? String, "p-th-claude")
    }

    func testAT3ChunkResolvesByItsDigestedConversationThroughMetadata() throws {
        try createSideTables()
        try addThread("th-digested", title: "Digested id thread", project: "brainlayer")
        try database.insertChunk(
            id: "t3-chunk", content: "A message T3 stored itself",
            sessionId: "t3:sha256:abc", project: "brainlayer", contentType: "user_message", importance: 5
        )
        try exec("ALTER TABLE chunks ADD COLUMN source_class TEXT")
        try exec("UPDATE chunks SET metadata = '{\"t3_thread_id\": \"th-digested\"}', source_class = 'agent-thinking' WHERE id = 't3-chunk'")
        let expanded = try database.expandChunk(id: "t3-chunk", before: 0, after: 0)
        XCTAssertEqual((expanded["target"] as? [String: Any])?["source_class"] as? String, "agent-thinking")
        let thread = try XCTUnwrap(
            expanded["t3_thread"] as? [String: Any]
        )
        XCTAssertEqual(thread["id"] as? String, "th-digested")
        let router = MCPRouter(profile: "full")
        router.setDatabase(database)
        let response = router.handle([
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": ["name": "brain_expand", "arguments": ["chunk_id": "t3-chunk"]] as [String: Any],
        ])
        let result = try XCTUnwrap(response["result"] as? [String: Any])
        let content = try XCTUnwrap(result["content"] as? [[String: Any]])
        let text = content.compactMap { $0["text"] as? String }.joined(separator: "\n")
        XCTAssertTrue(text.contains("t3://thread/th-digested"), text)
        XCTAssertTrue(text.contains(SearchResult.thinkingLabel), text)
    }

    func testASharedSessionPrefersTheLiveThread() throws {
        try createSideTables()
        try addThread("th-newer-deleted", title: "Deleted", project: "brainlayer",
                      createdAt: "2026-11-19T02:00:00Z", deletedAt: "2026-11-19T03:00:00Z", conversation: "sess-claude")
        try addThread("th-live", title: "Live", project: "brainlayer",
                      createdAt: "2026-11-19T01:00:00Z", conversation: "sess-claude", active: false)
        let thread = try XCTUnwrap(
            try database.expandChunk(id: "session-chunk", before: 0, after: 0)["t3_thread"] as? [String: Any]
        )
        XCTAssertEqual(thread["id"] as? String, "th-live")
    }

    func testBrainExpandToolTextNamesTheThread() throws {
        try createSideTables()
        try addThread("th-claude", title: "Claude thread", project: "brainlayer", conversation: "sess-claude")
        let router = MCPRouter(profile: "full")
        router.setDatabase(database)
        let response = router.handle([
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": ["name": "brain_expand", "arguments": ["chunk_id": "session-chunk"]] as [String: Any],
        ])
        let result = try XCTUnwrap(response["result"] as? [String: Any])
        let content = try XCTUnwrap(result["content"] as? [[String: Any]])
        let text = content.compactMap { $0["text"] as? String }.joined(separator: "\n")
        XCTAssertTrue(
            text.contains("T3 thread: Claude thread \u{00b7} brainlayer \u{00b7} t3://thread/th-claude"), text
        )
    }
}
