// DatabaseTests.swift — RED tests for BrainBar SQLite database layer.
//
// BrainBar embeds SQLite with:
// - WAL mode
// - FTS5 for full-text search
// - busy_timeout=5000
// - cache_size=-64000 (64MB)
// - synchronous=NORMAL
// - Single-writer architecture (no concurrent writes)

import XCTest
import SQLite3
@testable import BrainBar

final class DatabaseTests: XCTestCase {
    var db: BrainDatabase!
    var tempDBPath: String!

    override func setUp() {
        super.setUp()
        tempDBPath = NSTemporaryDirectory() + "brainbar-test-\(UUID().uuidString).db"
        db = BrainDatabase(path: tempDBPath)
    }

    override func tearDown() {
        db.close()
        try? FileManager.default.removeItem(atPath: tempDBPath)
        try? FileManager.default.removeItem(atPath: tempDBPath + "-wal")
        try? FileManager.default.removeItem(atPath: tempDBPath + "-shm")
        super.tearDown()
    }

    // MARK: - PRAGMAs

    func testWALModeEnabled() throws {
        let mode = try db.pragma("journal_mode")
        XCTAssertEqual(mode, "wal")
    }

    func testBusyTimeoutSet() throws {
        let timeout = try db.pragma("busy_timeout")
        XCTAssertEqual(timeout, "30000")
    }

    func testCacheSizeSet() throws {
        let cacheSize = try db.pragma("cache_size")
        XCTAssertEqual(cacheSize, "-64000")
    }

    func testSynchronousNormal() throws {
        let sync = try db.pragma("synchronous")
        // NORMAL = 1
        XCTAssertEqual(sync, "1")
    }

    // MARK: - Schema

    func testChunksTableExists() throws {
        let exists = try db.tableExists("chunks")
        XCTAssertTrue(exists, "chunks table must exist")
    }

    func testFTSTableExists() throws {
        let exists = try db.tableExists("chunks_fts")
        XCTAssertTrue(exists, "chunks_fts FTS5 table must exist")
    }

    func testFTSTableUsesPrefixIndexAndUnicodeTokenizer() throws {
        let sql = try sqliteMasterSQL(name: "chunks_fts", path: tempDBPath)

        XCTAssertTrue(sql.contains("prefix='2 3 4'"))
        XCTAssertTrue(sql.contains("tokenize='unicode61 remove_diacritics 2'"))
    }

    func testBrainbarAgentsTableExists() throws {
        let exists = try db.tableExists("brainbar_agents")
        XCTAssertTrue(exists, "brainbar_agents table must exist")
    }

    func testBrainbarSubscriptionsTableExists() throws {
        let exists = try db.tableExists("brainbar_subscriptions")
        XCTAssertTrue(exists, "brainbar_subscriptions table must exist")
    }

    func testInjectionEventsTableExists() throws {
        let exists = try db.tableExists("injection_events")
        XCTAssertTrue(exists, "injection_events table must exist")
    }

    func testQueueIDExpressionIndexExists() throws {
        let sql = try sqliteMasterSQL(name: "idx_chunks_brainbar_queue_id", path: tempDBPath)

        XCTAssertTrue(sql.contains("json_extract(metadata, '$.brainbar_queue_id')"))
        XCTAssertTrue(sql.contains("json_valid(metadata)"))
        XCTAssertTrue(sql.contains("json_extract(metadata, '$.brainbar_queue_id') IS NOT NULL"))
    }

    func testQueueIDExpressionIndexMigratesExistingSchema() throws {
        let legacyPath = NSTemporaryDirectory() + "brainbar-legacy-\(UUID().uuidString).db"
        try sqliteExecWrite(
            path: legacyPath,
            sql: """
                CREATE TABLE chunks (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
            """
        )
        defer {
            try? FileManager.default.removeItem(atPath: legacyPath)
            try? FileManager.default.removeItem(atPath: legacyPath + "-wal")
            try? FileManager.default.removeItem(atPath: legacyPath + "-shm")
        }

        let legacyDB = BrainDatabase(path: legacyPath)
        defer { legacyDB.close() }

        let sql = try sqliteMasterSQL(name: "idx_chunks_brainbar_queue_id", path: legacyPath)
        XCTAssertTrue(sql.contains("json_extract(metadata, '$.brainbar_queue_id')"))
    }

    func testQueueIDExpressionIndexMigratesExistingSchemaWithMalformedMetadata() throws {
        let legacyPath = NSTemporaryDirectory() + "brainbar-legacy-malformed-\(UUID().uuidString).db"
        try sqliteExecWrite(
            path: legacyPath,
            sql: """
                CREATE TABLE chunks (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                );
                INSERT INTO chunks (id, content, metadata)
                VALUES ('legacy-bad-metadata', 'Legacy malformed metadata row', '{not json');
            """
        )
        defer {
            try? FileManager.default.removeItem(atPath: legacyPath)
            try? FileManager.default.removeItem(atPath: legacyPath + "-wal")
            try? FileManager.default.removeItem(atPath: legacyPath + "-shm")
        }

        let legacyDB = BrainDatabase(path: legacyPath)
        defer { legacyDB.close() }

        XCTAssertTrue(legacyDB.isOpen)
        let sql = try sqliteMasterSQL(name: "idx_chunks_brainbar_queue_id", path: legacyPath)
        XCTAssertTrue(sql.contains("json_valid(metadata)"))
    }

    func testQueueIDLookupUsesExpressionIndex() throws {
        db.exec("""
            INSERT INTO chunks (
                id, content, metadata, source_file, project, content_type, char_count, source, tags, importance, preview_text
            ) VALUES (
                'queued-indexed',
                'Queued chunk metadata lookup',
                '{"brainbar_queue_id":"brainbar-pending-lookup"}',
                'brainbar-store',
                'brainbar',
                'assistant_text',
                28,
                'mcp',
                '[]',
                5,
                'Queued chunk metadata lookup'
            )
        """)

        let plan = try sqliteQueryPlan(
            path: tempDBPath,
            sql: """
                EXPLAIN QUERY PLAN
                SELECT 1
                FROM chunks
                WHERE json_valid(metadata)
                  AND json_extract(metadata, '$.brainbar_queue_id') = ?
                LIMIT 1
            """,
            binds: ["brainbar-pending-lookup"]
        )

        XCTAssertTrue(
            plan.contains(where: { $0.localizedCaseInsensitiveContains("idx_chunks_brainbar_queue_id") }),
            "Queue ID dedupe lookups should use the dedicated expression index"
        )
    }

    func testLargeTrigramDesyncDoesNotForceSynchronousStartupRebuild() {
        let decision = BrainDatabase.trigramStartupRepairDecision(
            tableExists: true,
            schemaIsValid: true,
            chunkCount: 359_890,
            trigramCount: 0
        )

        XCTAssertEqual(decision, .skipBackfill)
    }

    func testTriggerTrigramRebuildBackfillsInBatchesWithProgress() throws {
        try seedTrigramMaintenanceRows(count: 5)
        try sqliteExecWrite(path: tempDBPath, sql: "DELETE FROM chunks_fts_trigram")
        XCTAssertEqual(try sqliteCount(path: tempDBPath, table: "chunks_fts_trigram"), 0)

        var events: [BrainDatabase.TrigramMaintenanceProgress] = []
        let final = try db.triggerTrigramRebuild(batchSize: 2, progress: { event in
            events.append(event)
        })

        XCTAssertEqual(final.state, .done)
        XCTAssertEqual(final.total, 5)
        XCTAssertEqual(final.processed, 5)
        XCTAssertEqual(try sqliteCount(path: tempDBPath, table: "chunks_fts_trigram"), 5)
        XCTAssertGreaterThanOrEqual(events.filter { $0.state == .running }.count, 2)
        XCTAssertEqual(events.last?.state, .done)
    }

    func testTriggerTrigramRebuildHonorsCancellationBetweenBatches() throws {
        try seedTrigramMaintenanceRows(count: 5)
        try sqliteExecWrite(path: tempDBPath, sql: "DELETE FROM chunks_fts_trigram")

        var events: [BrainDatabase.TrigramMaintenanceProgress] = []
        let final = try db.triggerTrigramRebuild(
            batchSize: 2,
            shouldCancel: { !events.isEmpty },
            progress: { event in events.append(event) }
        )

        XCTAssertEqual(final.state, .cancelled)
        XCTAssertEqual(final.processed, 2)
        XCTAssertEqual(final.total, 5)
        XCTAssertEqual(try sqliteCount(path: tempDBPath, table: "chunks_fts_trigram"), 2)
        XCTAssertEqual(events.last?.state, .cancelled)
    }

    func testTriggerTrigramRebuildCancellationPreservesUnprocessedLiveRows() throws {
        try seedTrigramMaintenanceRows(count: 5)
        XCTAssertEqual(try sqliteCount(path: tempDBPath, table: "chunks_fts_trigram"), 5)

        var events: [BrainDatabase.TrigramMaintenanceProgress] = []
        let final = try db.triggerTrigramRebuild(
            batchSize: 2,
            shouldCancel: { !events.isEmpty },
            progress: { event in events.append(event) }
        )

        XCTAssertEqual(final.state, .cancelled)
        XCTAssertEqual(final.processed, 2)
        XCTAssertEqual(try sqliteCount(path: tempDBPath, table: "chunks_fts_trigram"), 5)
    }

    func testTriggerTrigramRebuildAllowsWritersBetweenBatches() throws {
        try seedTrigramMaintenanceRows(count: 4)
        try sqliteExecWrite(path: tempDBPath, sql: "DELETE FROM chunks_fts_trigram")

        var insertedDuringProgress = false
        var concurrentInsertError: Error?
        let final = try db.triggerTrigramRebuild(batchSize: 2, progress: { event in
            guard event.state == .running, !insertedDuringProgress else { return }
            do {
                try sqliteExecWrite(
                    path: self.tempDBPath,
                    sql: """
                        INSERT INTO chunks (
                            id, content, metadata, source_file, project, content_type, importance, conversation_id, char_count, tags, summary, preview_text
                        ) VALUES (
                            'trigram-concurrent-writer',
                            'Concurrent writer should acquire the lock between trigram batches',
                            '{}',
                            'brainbar',
                            'brainlayer',
                            'assistant_text',
                            5,
                            'trigram-maintenance',
                            67,
                            '[]',
                            '',
                            'Concurrent writer should acquire the lock between trigram batches'
                        )
                    """
                )
                insertedDuringProgress = true
            } catch {
                concurrentInsertError = error
            }
        })

        XCTAssertEqual(final.state, .done)
        XCTAssertNil(concurrentInsertError)
        XCTAssertTrue(insertedDuringProgress)
        XCTAssertEqual(try sqliteCount(path: tempDBPath, table: "chunks"), 5)
        XCTAssertEqual(try sqliteCount(path: tempDBPath, table: "chunks_fts_trigram"), 5)
    }

    func testTriggerTrigramRebuildDoesNotDuplicateRowsWhenChunkUpdatesBetweenBatches() throws {
        try seedTrigramMaintenanceRows(count: 4)
        try sqliteExecWrite(path: tempDBPath, sql: "DELETE FROM chunks_fts_trigram")

        var updatedDuringProgress = false
        var concurrentUpdateError: Error?
        let final = try db.triggerTrigramRebuild(batchSize: 2, progress: { event in
            guard event.state == .running, !updatedDuringProgress else { return }
            do {
                try sqliteExecWrite(
                    path: self.tempDBPath,
                    sql: """
                        UPDATE chunks
                        SET content = 'Updated not-yet-processed chunk during trigram maintenance'
                        WHERE id = 'trigram-maintenance-3'
                    """
                )
                updatedDuringProgress = true
            } catch {
                concurrentUpdateError = error
            }
        })

        XCTAssertEqual(final.state, .done)
        XCTAssertNil(concurrentUpdateError)
        XCTAssertTrue(updatedDuringProgress)
        XCTAssertEqual(
            try sqliteCount(
                path: tempDBPath,
                sql: "SELECT COUNT(*) FROM chunks_fts_trigram WHERE chunk_id = 'trigram-maintenance-3'"
            ),
            1
        )
        XCTAssertEqual(try sqliteCount(path: tempDBPath, table: "chunks_fts_trigram"), 4)
    }

    func testUpsertSubscriptionRecoversMissingPubSubTables() throws {
        db.exec("DROP TABLE IF EXISTS brainbar_subscriptions")
        db.exec("DROP TABLE IF EXISTS brainbar_agents")

        let record = try db.upsertSubscription(agentID: "agent-a", tags: ["agent-message"])

        XCTAssertEqual(record.agentID, "agent-a")
        XCTAssertEqual(record.tags, ["agent-message"])
        XCTAssertTrue(try db.tableExists("brainbar_agents"))
        XCTAssertTrue(try db.tableExists("brainbar_subscriptions"))
    }

    // MARK: - Search (FTS5)

    func testFTSSearchReturnsResults() throws {
        // Insert a test chunk
        try db.insertChunk(
            id: "test-chunk-1",
            content: "Authentication was implemented using JWT tokens with refresh rotation",
            sessionId: "session-1",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 7
        )

        let results = try db.search(query: "authentication JWT", limit: 10)
        XCTAssertFalse(results.isEmpty, "FTS search should find the inserted chunk")
        XCTAssertEqual(results.first?["chunk_id"] as? String, "test-chunk-1")
    }

    func testSearchReturnsEmptyForNoMatch() throws {
        let results = try db.search(query: "xyznonexistent123", limit: 10)
        XCTAssertTrue(results.isEmpty)
    }

    func testSearchReturnsEmptyForBlankQuery() throws {
        try db.insertChunk(
            id: "blank-query-decoy",
            content: "Blank searches should not hit arbitrary content.",
            sessionId: "session-blank",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 5
        )

        let results = try db.search(query: "   ", limit: 10)

        XCTAssertTrue(results.isEmpty)
    }

    func testSearchExactChunkIDShortCircuitsFTS() throws {
        try db.insertChunk(
            id: "brainbar-5ac50aa1-ed5",
            content: "Exact chunk lookup should not require the identifier to appear in content.",
            sessionId: "session-exact",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 9
        )
        try db.insertChunk(
            id: "brainbar-d01",
            content: "Decoy chunk mentions brainbar-5ac50aa1-ed5 but must not outrank the exact chunk id.",
            sessionId: "session-decoy",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 9
        )

        let results = try db.search(query: "brainbar-5ac50aa1-ed5", limit: 10)

        XCTAssertEqual(results.first?["chunk_id"] as? String, "brainbar-5ac50aa1-ed5")
    }

    func testSearchExactChunkIDRespectsProjectScope() throws {
        try db.insertChunk(
            id: "brainbar-scoped-001",
            content: "Project-scoped exact id lookup should return only matching project chunks.",
            sessionId: "session-scope",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 8
        )

        let scoped = try db.search(query: "brainbar-scoped-001", limit: 10, project: "brainlayer")
        let wrongScope = try db.search(query: "brainbar-scoped-001", limit: 10, project: "voicelayer")

        XCTAssertEqual(scoped.first?["chunk_id"] as? String, "brainbar-scoped-001")
        XCTAssertTrue(wrongScope.isEmpty)
    }

    func testSearchUsesTrigramFTSForIdentifierSubstrings() throws {
        try db.insertChunk(
            id: "trigram-hit",
            content: "stalker-golem queue note",
            sessionId: "session-trigram",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 7
        )

        let results = try db.search(query: "alker-go", limit: 10)

        XCTAssertEqual(results.first?["chunk_id"] as? String, "trigram-hit")
    }

    func testSearchAliasExpansionPreservesMultiwordSemantics() throws {
        try db.insertChunk(
            id: "alias-good",
            content: "Hershkovits reviewed the release plan yesterday.",
            sessionId: "session-alias-good",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 8
        )
        try db.insertChunk(
            id: "alias-bad",
            content: "Met with Hershkovits yesterday.",
            sessionId: "session-alias-bad",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 8
        )

        let results = try db.search(query: "Hershkovitz release plan", limit: 10)
        let resultIDs = results.compactMap { $0["chunk_id"] as? String }

        XCTAssertTrue(resultIDs.contains("alias-good"))
        XCTAssertFalse(resultIDs.contains("alias-bad"))
    }

    func testSearchAliasExpansionNormalizesDotsConsistently() throws {
        db.exec("""
            INSERT INTO kg_entities (id, entity_type, name)
            VALUES ('entity-openai', 'org', 'Open.AI');
        """)
        db.exec("""
            INSERT INTO kg_entity_aliases (alias, entity_id)
            VALUES ('OpenAI', 'entity-openai');
        """)
        try db.insertChunk(
            id: "alias-dot",
            content: "Open.AI roadmap review notes.",
            sessionId: "session-alias-dot",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 8
        )

        let results = try db.search(query: "OpenAI roadmap", limit: 10)

        XCTAssertEqual(results.first?["chunk_id"] as? String, "alias-dot")
    }

    // MARK: - Store

    func testStoreCreatesChunk() throws {
        let stored = try db.store(
            content: "Decision: Use GRDB for SQLite access",
            tags: ["decision", "architecture"],
            importance: 8,
            source: "mcp"
        )

        XCTAssertFalse(stored.chunkID.isEmpty, "store should return a chunk ID")
        XCTAssertGreaterThan(stored.rowID, 0)

        // Verify it's searchable
        let results = try db.search(query: "GRDB SQLite", limit: 10)
        XCTAssertFalse(results.isEmpty)
    }

    func testAnalyzePopulatesStatsForSearchIndexes() throws {
        try db.insertChunk(
            id: "analyze-1",
            content: "Cmd K search should use analyzed prefix indexes",
            sessionId: "session-1",
            project: "brainbar",
            contentType: "assistant_text",
            importance: 7
        )

        XCTAssertGreaterThan(try sqliteStatCount(for: ["chunks", "chunks_fts"], path: tempDBPath), 0, "ANALYZE should populate sqlite_stat1 for search tables")
    }

    func testSearchCandidatesReturnPrecomputedPreviewText() throws {
        try db.insertChunk(
            id: "preview-1",
            content: """
            Search previews must be precomputed ahead of keystrokes so the UI never calls snippet() while typing.
            This content is intentionally long enough to exercise truncation and whitespace normalization.
            """,
            sessionId: "session-1",
            project: "brainbar",
            contentType: "assistant_text",
            importance: 6
        )

        let results = try db.searchCandidates(query: "precomputed keystrokes", limit: 10)

        XCTAssertEqual(results.first?.id, "preview-1")
        XCTAssertFalse(results.first?.previewText.isEmpty ?? true)
        XCTAssertFalse(results.first?.previewText.contains("\n") ?? false)
    }

    func testStoreRetriesThroughTransientWriteLock() throws {
        var lockDB: OpaquePointer?
        let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_FULLMUTEX
        XCTAssertEqual(sqlite3_open_v2(tempDBPath, &lockDB, flags, nil), SQLITE_OK)
        guard let lockDB else {
            XCTFail("Failed to open secondary lock connection")
            return
        }
        defer { sqlite3_close(lockDB) }

        XCTAssertEqual(sqlite3_exec(lockDB, "BEGIN IMMEDIATE", nil, nil, nil), SQLITE_OK)

        let releaseExpectation = expectation(description: "release write lock")
        DispatchQueue.global().asyncAfter(deadline: .now() + 5.5, execute: DispatchWorkItem {
            sqlite3_exec(lockDB, "COMMIT", nil, nil, nil)
            releaseExpectation.fulfill()
        })

        let startedAt = Date()
        let stored = try db.store(
            content: "Store after transient lock",
            tags: ["retry"],
            importance: 5,
            source: "mcp"
        )

        XCTAssertFalse(stored.chunkID.isEmpty)
        XCTAssertGreaterThan(Date().timeIntervalSince(startedAt), 5.0)
        wait(for: [releaseExpectation], timeout: 7.0)
    }

    func testRecordInjectionEventPersistsSessionQueryChunkIDsAndTokenCount() throws {
        try db.recordInjectionEvent(
            sessionID: "claude-session-1",
            query: "voicebar sleep recovery",
            chunkIDs: ["chunk-1", "chunk-2"],
            tokenCount: 77
        )

        let events = try db.listInjectionEvents(limit: 5)

        XCTAssertEqual(events.count, 1)
        XCTAssertEqual(events.first?.sessionID, "claude-session-1")
        XCTAssertEqual(events.first?.query, "voicebar sleep recovery")
        XCTAssertEqual(events.first?.chunkIDs, ["chunk-1", "chunk-2"])
        XCTAssertEqual(events.first?.tokenCount, 77)
    }

    func testListInjectionEventsFiltersBySessionAndNewestFirst() throws {
        try db.recordInjectionEvent(
            sessionID: "session-a",
            query: "older event",
            chunkIDs: ["old-1"],
            tokenCount: 10,
            timestamp: "2026-03-31T04:00:00.000Z"
        )
        try db.recordInjectionEvent(
            sessionID: "session-b",
            query: "other session",
            chunkIDs: ["other-1"],
            tokenCount: 20,
            timestamp: "2026-03-31T04:01:00.000Z"
        )
        try db.recordInjectionEvent(
            sessionID: "session-a",
            query: "newer event",
            chunkIDs: ["new-1", "new-2"],
            tokenCount: 30,
            timestamp: "2026-03-31T04:02:00.000Z"
        )

        let filtered = try db.listInjectionEvents(sessionID: "session-a", limit: 10)

        XCTAssertEqual(filtered.map(\.query), ["newer event", "older event"])
    }

    // MARK: - Filter: project

    func testSearchFiltersByProject() throws {
        try db.insertChunk(id: "proj-a-1", content: "Authentication uses JWT tokens", sessionId: "s1", project: "alpha", contentType: "assistant_text", importance: 5)
        try db.insertChunk(id: "proj-b-1", content: "Authentication uses OAuth tokens", sessionId: "s2", project: "beta", contentType: "assistant_text", importance: 5)

        let filtered = try db.search(query: "authentication tokens", limit: 10, project: "alpha")
        XCTAssertEqual(filtered.count, 1, "Should return only alpha project chunk")
        XCTAssertEqual(filtered.first?["project"] as? String, "alpha")
    }

    func testSearchWithoutProjectReturnsAll() throws {
        try db.insertChunk(id: "all-a", content: "Database migration script", sessionId: "s1", project: "alpha", contentType: "assistant_text", importance: 5)
        try db.insertChunk(id: "all-b", content: "Database migration tool", sessionId: "s2", project: "beta", contentType: "assistant_text", importance: 5)

        let all = try db.search(query: "database migration", limit: 10)
        XCTAssertEqual(all.count, 2, "Without filter, both projects should be returned")
    }

    // MARK: - Filter: importance_min

    func testSearchFiltersByImportanceMin() throws {
        try db.insertChunk(id: "imp-low", content: "Logging configuration setup", sessionId: "s1", project: "test", contentType: "assistant_text", importance: 3)
        try db.insertChunk(id: "imp-high", content: "Logging security audit", sessionId: "s2", project: "test", contentType: "assistant_text", importance: 8)

        let filtered = try db.search(query: "logging", limit: 10, importanceMin: 7.0)
        XCTAssertEqual(filtered.count, 1, "Should return only high-importance chunk")
        XCTAssertEqual(filtered.first?["chunk_id"] as? String, "imp-high")
    }

    // MARK: - Filter: tag

    func testSearchFiltersByTag() throws {
        try db.insertChunk(id: "tag-1", content: "Fixed the authentication bug in login flow", sessionId: "s1", project: "test", contentType: "assistant_text", importance: 5, tags: "[\"bug-fix\", \"auth\"]")
        try db.insertChunk(id: "tag-2", content: "Fixed the authentication bug in signup flow", sessionId: "s2", project: "test", contentType: "assistant_text", importance: 5)

        let filtered = try db.search(query: "authentication bug", limit: 10, tag: "bug-fix")
        XCTAssertEqual(filtered.count, 1, "Should return only tagged chunk")
        XCTAssertEqual(filtered.first?["chunk_id"] as? String, "tag-1")
    }

    // MARK: - Filter: combined

    func testSearchCombinesFilters() throws {
        try db.insertChunk(id: "combo-1", content: "API rate limiting implementation", sessionId: "s1", project: "alpha", contentType: "assistant_text", importance: 9)
        try db.insertChunk(id: "combo-2", content: "API rate limiting design", sessionId: "s2", project: "beta", contentType: "assistant_text", importance: 9)
        try db.insertChunk(id: "combo-3", content: "API rate limiting notes", sessionId: "s3", project: "alpha", contentType: "assistant_text", importance: 3)

        let filtered = try db.search(query: "API rate limiting", limit: 10, project: "alpha", importanceMin: 7.0)
        XCTAssertEqual(filtered.count, 1, "Should match only alpha + high importance")
        XCTAssertEqual(filtered.first?["chunk_id"] as? String, "combo-1")
    }

    // MARK: - Production rowid divergence

    /// Simulates production DB where FTS5 rowids don't match chunks rowids.
    /// In production, Python's trigger doesn't set explicit rowid, so after
    /// FTS5 table rebuilds, rowids diverge. The JOIN must use chunk_id, not rowid.
    func testImportanceFilterWorksWithDivergedRowids() throws {
        // Insert two chunks normally (synced rowids via trigger)
        try db.insertChunk(id: "div-1", content: "Unimportant chatter about weather", sessionId: "s1", project: "test", contentType: "assistant_text", importance: 2)
        try db.insertChunk(id: "div-2", content: "Critical architecture decision about caching", sessionId: "s2", project: "test", contentType: "assistant_text", importance: 9)

        // Simulate production rebuild: drop FTS5 table + triggers, recreate, re-populate
        // This creates divergent rowids: FTS5 rows get new rowids 1,2 but chunks keep original rowids
        db.exec("DROP TRIGGER IF EXISTS chunks_fts_insert")
        db.exec("DROP TRIGGER IF EXISTS chunks_fts_delete")
        db.exec("DROP TRIGGER IF EXISTS chunks_fts_update")
        db.exec("DROP TABLE IF EXISTS chunks_fts")
        db.exec("""
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                content, summary, tags, resolved_query, chunk_id UNINDEXED
            )
        """)
        // Re-populate WITHOUT explicit rowid (matches production trigger behavior).
        // FTS5 auto-assigns rowid 1 to div-2 and rowid 2 to div-1 (or vice versa),
        // which WON'T match the chunks table rowids.
        // Insert in REVERSE order to guarantee mismatch: FTS5 rowid 1 = high-importance chunk.
        db.exec("""
            INSERT INTO chunks_fts(content, summary, tags, resolved_query, chunk_id)
            SELECT content, summary, tags, NULL, id FROM chunks ORDER BY id DESC
        """)
        // Recreate trigger matching production (no explicit rowid)
        db.exec("""
            CREATE TRIGGER chunks_fts_insert AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(content, summary, tags, resolved_query, chunk_id)
                VALUES (new.content, new.summary, new.tags, NULL, new.id);
            END
        """)

        // Search with importance_min=7 — should return ONLY the high-importance chunk
        let filtered = try db.search(query: "architecture decision caching", limit: 10, importanceMin: 7.0)
        XCTAssertEqual(filtered.count, 1, "Should return only high-importance chunk")
        XCTAssertEqual(filtered.first?["chunk_id"] as? String, "div-2")

        // Also verify the returned importance is correct (not from wrong row)
        let importance = filtered.first?["importance"]
        XCTAssertNotNil(importance, "importance should be present")
        // Should be 9, not 2 (which would happen if rowids mapped to wrong chunk)
        if let imp = importance as? Int {
            XCTAssertGreaterThanOrEqual(imp, 7, "Returned importance must be >= 7")
        } else if let imp = importance as? Double {
            XCTAssertGreaterThanOrEqual(imp, 7.0, "Returned importance must be >= 7")
        }
    }

    /// Verify ALL results from importance_min search actually have importance >= threshold.
    func testAllResultsRespectImportanceMin() throws {
        try db.insertChunk(id: "all-1", content: "Vector database indexing strategies", sessionId: "s1", project: "test", contentType: "assistant_text", importance: 3)
        try db.insertChunk(id: "all-2", content: "Vector database performance tuning", sessionId: "s2", project: "test", contentType: "assistant_text", importance: 7)
        try db.insertChunk(id: "all-3", content: "Vector database scaling patterns", sessionId: "s3", project: "test", contentType: "assistant_text", importance: 9)

        let results = try db.search(query: "vector database", limit: 10, importanceMin: 7.0)
        XCTAssertEqual(results.count, 2, "Should return only chunks with importance >= 7")
        for result in results {
            if let imp = result["importance"] as? Int {
                XCTAssertGreaterThanOrEqual(imp, 7, "Every result must have importance >= 7")
            } else if let imp = result["importance"] as? Double {
                XCTAssertGreaterThanOrEqual(imp, 7.0, "Every result must have importance >= 7")
            }
        }
    }

    // MARK: - Concurrent reads

    func testConcurrentReadsDoNotBlock() throws {
        // Insert test data
        try db.insertChunk(
            id: "concurrent-1",
            content: "Test chunk for concurrent reads",
            sessionId: "session-1",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 5
        )

        let expectation = XCTestExpectation(description: "concurrent reads")
        expectation.expectedFulfillmentCount = 10
        let database = db!

        for _ in 0..<10 {
            DispatchQueue.global().async {
                do {
                    let results = try database.search(query: "concurrent", limit: 5)
                    XCTAssertFalse(results.isEmpty)
                } catch {
                    XCTFail("Concurrent read failed: \(error)")
                }
                expectation.fulfill()
            }
        }

        wait(for: [expectation], timeout: 5.0)
    }

    // MARK: - Search Ranking (FTS5 BM25)

    func testSearchResultsHaveNonZeroScore() throws {
        try db.insertChunk(id: "score-1", content: "BrainBar Swift daemon formatting", sessionId: "s1", project: "test", contentType: "assistant_text", importance: 5)

        let results = try db.search(query: "BrainBar daemon", limit: 5)
        XCTAssertFalse(results.isEmpty)
        let score = results.first?["score"] as? Double ?? 0
        XCTAssertGreaterThan(score, 0, "Search results should have a non-zero relevance score")
    }

    func testMultiWordSearchUsesAND() throws {
        // "overnight hardening sprint" should match chunk with ALL three words,
        // not chunks with just "sprint" alone (OR would match both)
        try db.insertChunk(id: "and-1", content: "The overnight hardening sprint produced great results with improved stability.", sessionId: "s1", project: "test", contentType: "assistant_text", importance: 5)
        try db.insertChunk(id: "and-2", content: "We had a quick sprint planning session this morning.", sessionId: "s2", project: "test", contentType: "assistant_text", importance: 5)

        let results = try db.search(query: "overnight hardening sprint", limit: 5)
        // Only and-1 should match (has all 3 words). and-2 only has "sprint".
        XCTAssertEqual(results.count, 1, "AND mode: only chunks with ALL query terms should match")
        XCTAssertEqual(results.first?["chunk_id"] as? String, "and-1")
    }

    func testSearchResultsOrderedByRelevance() throws {
        // "sprint" appears in content of both, but the first has it more prominently
        try db.insertChunk(id: "rel-1", content: "The overnight hardening sprint was a success. Sprint results show improvements.", sessionId: "s1", project: "test", contentType: "assistant_text", importance: 5)
        try db.insertChunk(id: "rel-2", content: "We discussed various topics including weather and sprint planning briefly.", sessionId: "s2", project: "test", contentType: "assistant_text", importance: 5)

        let results = try db.search(query: "sprint", limit: 5)
        XCTAssertEqual(results.count, 2)
        let score1 = results[0]["score"] as? Double ?? 0
        let score2 = results[1]["score"] as? Double ?? 0
        XCTAssertGreaterThanOrEqual(score1, score2, "Results should be ordered by relevance (highest score first)")
    }

    // MARK: - brain_tags (list unique tags with counts)

    func testListTagsReturnsUniqueTags() throws {
        try db.insertChunk(id: "tag-1", content: "First chunk", sessionId: "s1", project: "test", contentType: "assistant_text", importance: 5, tags: "[\"swift\", \"macos\"]")
        try db.insertChunk(id: "tag-2", content: "Second chunk", sessionId: "s2", project: "test", contentType: "assistant_text", importance: 5, tags: "[\"swift\", \"daemon\"]")
        try db.insertChunk(id: "tag-3", content: "Third chunk", sessionId: "s3", project: "test", contentType: "assistant_text", importance: 5, tags: "[\"daemon\"]")

        let tags = try db.listTags(limit: 10)
        XCTAssertFalse(tags.isEmpty)
        // swift=2, daemon=2, macos=1
        let swiftTag = tags.first(where: { $0["tag"] as? String == "swift" })
        XCTAssertNotNil(swiftTag)
        XCTAssertEqual(swiftTag?["count"] as? Int, 2)
        let daemonTag = tags.first(where: { $0["tag"] as? String == "daemon" })
        XCTAssertEqual(daemonTag?["count"] as? Int, 2)
    }

    func testListTagsFiltersByQuery() throws {
        try db.insertChunk(id: "tq-1", content: "A", sessionId: "s1", project: "test", contentType: "assistant_text", importance: 5, tags: "[\"architecture\", \"swift\"]")
        try db.insertChunk(id: "tq-2", content: "B", sessionId: "s2", project: "test", contentType: "assistant_text", importance: 5, tags: "[\"archival\", \"python\"]")

        let tags = try db.listTags(query: "arch", limit: 10)
        let tagNames = tags.compactMap { $0["tag"] as? String }
        XCTAssertTrue(tagNames.contains("architecture"))
        XCTAssertTrue(tagNames.contains("archival"))
        XCTAssertFalse(tagNames.contains("swift"))
    }

    // MARK: - brain_update (update chunk content/tags/importance)

    func testUpdateChunkImportance() throws {
        try db.insertChunk(id: "upd-1", content: "Original content", sessionId: "s1", project: "test", contentType: "assistant_text", importance: 5)

        try db.updateChunk(id: "upd-1", importance: 9)

        let results = try db.search(query: "Original content", limit: 1)
        XCTAssertEqual(results.first?["importance"] as? Double, 9.0)
    }

    func testUpdateChunkTags() throws {
        try db.insertChunk(id: "upd-2", content: "Tag update test", sessionId: "s1", project: "test", contentType: "assistant_text", importance: 5, tags: "[\"old-tag\"]")

        try db.updateChunk(id: "upd-2", tags: ["new-tag", "updated"])

        let results = try db.search(query: "Tag update test", limit: 1)
        let tagsStr = results.first?["tags"] as? String ?? ""
        XCTAssertTrue(tagsStr.contains("new-tag"))
        XCTAssertTrue(tagsStr.contains("updated"))
        XCTAssertFalse(tagsStr.contains("old-tag"))
    }

    func testUpdateChunkThrowsOnNonExistentChunk() throws {
        XCTAssertThrowsError(try db.updateChunk(id: "nonexistent-chunk-id", importance: 9)) { error in
            XCTAssertTrue(error is BrainDatabase.DBError, "Should throw DBError")
            if case BrainDatabase.DBError.noResult = error {
                // Expected error type
            } else {
                XCTFail("Expected DBError.noResult, got \(error)")
            }
        }
    }

    // MARK: - brain_expand (get chunk + surrounding context)

    func testExpandChunkReturnsSurroundingContext() throws {
        // Insert 5 chunks in same session
        for i in 1...5 {
            try db.insertChunk(id: "exp-\(i)", content: "Chunk number \(i) in session", sessionId: "expand-session", project: "test", contentType: "assistant_text", importance: 5)
        }

        let expanded = try db.expandChunk(id: "exp-3", before: 2, after: 2)
        XCTAssertNotNil(expanded["target"])
        let context = expanded["context"] as? [[String: Any]] ?? []
        // Should have surrounding chunks from same session
        XCTAssertGreaterThanOrEqual(context.count, 2, "Should return at least 2 surrounding chunks")
    }

    // MARK: - brain_entity (lookup entity + relations)

    func testEntityLookup() throws {
        try db.insertEntity(id: "ent-bl", type: "project", name: "BrainLayer", metadata: "{\"description\": \"Local knowledge pipeline\"}")
        try db.insertEntity(id: "ent-eh", type: "person", name: "Etan Heyman", metadata: "{}")
        try db.insertRelation(sourceId: "ent-eh", targetId: "ent-bl", relationType: "works_on")

        let entity = try db.lookupEntity(query: "BrainLayer")
        XCTAssertNotNil(entity)
        XCTAssertEqual(entity?["name"] as? String, "BrainLayer")
        XCTAssertEqual(entity?["entity_type"] as? String, "project")
        let relations = entity?["relations"] as? [[String: Any]] ?? []
        XCTAssertFalse(relations.isEmpty, "Should include relations")
    }

    func testEntityLookupNotFound() throws {
        let entity = try db.lookupEntity(query: "NonexistentEntity12345")
        XCTAssertNil(entity)
    }

    // MARK: - brain_recall (session context)

    func testRecallStatsMode() throws {
        try db.insertChunk(id: "rc-1", content: "Stats test chunk", sessionId: "s1", project: "test", contentType: "assistant_text", importance: 5)

        let stats = try db.recallStats()
        XCTAssertNotNil(stats["total_chunks"])
        let total = stats["total_chunks"] as? Int ?? 0
        XCTAssertGreaterThan(total, 0)
    }

    // MARK: - brain_digest (rule-based entity extraction)

    func testDigestExtractsEntities() throws {
        let content = "Etan Heyman discussed BrainLayer architecture with Claude. The project uses SQLite and Swift."
        let result = try db.digest(content: content)
        let entities = result["entities"] as? [String] ?? []
        // Should extract capitalized multi-word names
        XCTAssertTrue(entities.contains(where: { $0.contains("Etan") }), "Should extract 'Etan Heyman'")
        XCTAssertTrue(entities.contains(where: { $0.contains("BrainLayer") }), "Should extract 'BrainLayer'")
    }

    func testDigestExtractsKeyPhrases() throws {
        let content = "Decision: Use SQLite for storage. The architecture should support real-time indexing."
        let result = try db.digest(content: content)
        XCTAssertNotNil(result["chunks_created"])
    }

    private func seedTrigramMaintenanceRows(count: Int) throws {
        for index in 0..<count {
            try db.insertChunk(
                id: "trigram-maintenance-\(index)",
                content: "Trigram maintenance fixture \(index)",
                sessionId: "trigram-maintenance",
                project: "brainlayer",
                contentType: "assistant_text",
                importance: 5
            )
        }
    }
}

private func sqliteMasterSQL(name: String, path: String) throws -> String {
    try withSQLiteConnection(path: path) { db in
        try scalarString(
            db: db,
            sql: "SELECT sql FROM sqlite_master WHERE name = ?",
            bind: { stmt in
                sqlite3_bind_text(stmt, 1, name, -1, unsafeBitCast(-1, to: sqlite3_destructor_type.self))
            }
        ) ?? ""
    }
}

private func sqliteStatCount(for tables: [String], path: String) throws -> Int {
    try withSQLiteConnection(path: path) { db in
        let placeholders = Array(repeating: "?", count: tables.count).joined(separator: ", ")
        return try scalarInt(
            db: db,
            sql: "SELECT COUNT(*) FROM sqlite_stat1 WHERE tbl IN (\(placeholders))",
            bind: { stmt in
                for (index, table) in tables.enumerated() {
                    sqlite3_bind_text(
                        stmt,
                        Int32(index + 1),
                        table,
                        -1,
                        unsafeBitCast(-1, to: sqlite3_destructor_type.self)
                    )
                }
            }
        )
    }
}

private func sqliteQueryPlan(path: String, sql: String, binds: [String]) throws -> [String] {
    try withSQLiteConnection(path: path) { db in
        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else {
            throw NSError(domain: "DatabaseTests", code: Int(rc))
        }
        defer { sqlite3_finalize(stmt) }

        for (index, value) in binds.enumerated() {
            sqlite3_bind_text(
                stmt,
                Int32(index + 1),
                value,
                -1,
                unsafeBitCast(-1, to: sqlite3_destructor_type.self)
            )
        }

        var details: [String] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            if let text = sqlite3_column_text(stmt, 3) {
                details.append(String(cString: text))
            }
        }
        return details
    }
}

private func withSQLiteConnection<T>(path: String, body: (OpaquePointer) throws -> T) throws -> T {
    var db: OpaquePointer?
    let rc = sqlite3_open_v2(path, &db, SQLITE_OPEN_READONLY | SQLITE_OPEN_FULLMUTEX, nil)
    guard rc == SQLITE_OK, let db else {
        throw NSError(domain: "DatabaseTests", code: Int(rc))
    }
    defer { sqlite3_close(db) }
    return try body(db)
}

private func sqliteExecWrite(path: String, sql: String) throws {
    var db: OpaquePointer?
    let rc = sqlite3_open_v2(
        path,
        &db,
        SQLITE_OPEN_CREATE | SQLITE_OPEN_READWRITE | SQLITE_OPEN_FULLMUTEX,
        nil
    )
    guard rc == SQLITE_OK, let db else {
        throw NSError(domain: "DatabaseTests", code: Int(rc))
    }
    defer { sqlite3_close(db) }

    let execRC = sqlite3_exec(db, sql, nil, nil, nil)
    guard execRC == SQLITE_OK else {
        throw NSError(domain: "DatabaseTests", code: Int(execRC))
    }
}

private func sqliteCount(path: String, table: String) throws -> Int {
    try sqliteCount(path: path, sql: "SELECT COUNT(*) FROM \(table)")
}

private func sqliteCount(path: String, sql: String) throws -> Int {
    try withSQLiteConnection(path: path) { db in
        try scalarInt(
            db: db,
            sql: sql,
            bind: { _ in }
        )
    }
}

private func scalarString(
    db: OpaquePointer,
    sql: String,
    bind: (OpaquePointer?) -> Void
) throws -> String? {
    var stmt: OpaquePointer?
    let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
    guard rc == SQLITE_OK else {
        throw NSError(domain: "DatabaseTests", code: Int(rc))
    }
    defer { sqlite3_finalize(stmt) }
    bind(stmt)
    guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
    guard let value = sqlite3_column_text(stmt, 0) else { return nil }
    return String(cString: value)
}

private func scalarInt(
    db: OpaquePointer,
    sql: String,
    bind: (OpaquePointer?) -> Void
) throws -> Int {
    var stmt: OpaquePointer?
    let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
    guard rc == SQLITE_OK else {
        throw NSError(domain: "DatabaseTests", code: Int(rc))
    }
    defer { sqlite3_finalize(stmt) }
    bind(stmt)
    guard sqlite3_step(stmt) == SQLITE_ROW else { return 0 }
    return Int(sqlite3_column_int(stmt, 0))
}
