// BrainDatabase.swift — SQLite database layer for BrainBar.
//
// Wraps SQLite3 directly. BrainBar now keeps its pub/sub metadata in the main
// BrainLayer database so agent state and chunk writes share one durable store.

import Darwin
import Foundation
import SQLite3

final class BrainDatabase: @unchecked Sendable {
    static let dashboardDidChangeNotification = "com.brainlayer.brainbar.database-changed"

    struct DashboardStats: Sendable, Equatable {
        let chunkCount: Int
        let enrichedChunkCount: Int
        let pendingEnrichmentCount: Int
        let enrichmentPercent: Double
        let databaseSizeBytes: Int64
        let recentActivityBuckets: [Int]
    }

    struct SubscriberRecord: Sendable {
        let agentID: String
        let generation: Int
        let connectionState: String
        let tags: [String]
        let lastDeliveredSeq: Int64
        let lastAckedSeq: Int64
        let lastConnectedAt: String?
        let disconnectedAt: String?
    }

    struct StoredChunk: Sendable {
        let chunkID: String
        let rowID: Int64
    }

    private var db: OpaquePointer?
    private let path: String
    private(set) var isOpen = false

    init(path: String) {
        self.path = path
        openAndConfigure()
    }

    private func openAndConfigure() {
        do {
            db = try openConnection(path: path)
            try configureConnection(db)
            try ensureSchema()
            isOpen = true
        } catch {
            NSLog("[BrainBar] Failed to open/configure database at %@: %@", path, String(describing: error))
        }
    }

    private func ensureSchema() throws {
        if (try? tableExists("chunks")) != true {
            try execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    source_file TEXT NOT NULL DEFAULT 'brainbar',
                    project TEXT,
                    content_type TEXT DEFAULT 'assistant_text',
                    value_type TEXT,
                    char_count INTEGER DEFAULT 0,
                    source TEXT DEFAULT 'claude_code',
                    sender TEXT,
                    language TEXT,
                    conversation_id TEXT,
                    position INTEGER,
                    context_summary TEXT,
                    tags TEXT DEFAULT '[]',
                    tag_confidence REAL,
                    summary TEXT,
                    importance REAL DEFAULT 5,
                    intent TEXT,
                    enriched_at TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)

            try execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    content, summary, tags, resolved_query, chunk_id UNINDEXED
                )
            """)

            try execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(content, summary, tags, resolved_query, chunk_id)
                    VALUES (new.content, new.summary, new.tags, NULL, new.id);
                END
            """)

            try execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
                    DELETE FROM chunks_fts WHERE chunk_id = old.id;
                END
            """)

            try execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON chunks BEGIN
                    DELETE FROM chunks_fts WHERE chunk_id = old.id;
                    INSERT INTO chunks_fts(content, summary, tags, resolved_query, chunk_id)
                    VALUES (new.content, new.summary, new.tags, NULL, new.id);
                END
            """)
        }

        try execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_created_at
            ON chunks(created_at)
        """)

        try ensureAuxiliarySchema()
    }

    private func ensureAuxiliarySchema() throws {
        try execute("""
            CREATE TABLE IF NOT EXISTS brainbar_agents (
                agent_id TEXT PRIMARY KEY,
                generation INTEGER NOT NULL DEFAULT 1,
                connection_state TEXT NOT NULL DEFAULT 'disconnected',
                last_connected_at TEXT,
                disconnected_at TEXT,
                last_delivered_seq INTEGER NOT NULL DEFAULT 0,
                last_acked_seq INTEGER NOT NULL DEFAULT 0
            )
        """)

        try execute("""
            CREATE TABLE IF NOT EXISTS brainbar_subscriptions (
                agent_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                subscribed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                PRIMARY KEY (agent_id, tag),
                FOREIGN KEY (agent_id) REFERENCES brainbar_agents(agent_id) ON DELETE CASCADE
            ) WITHOUT ROWID
        """)

        try execute("""
            CREATE INDEX IF NOT EXISTS idx_brainbar_subscriptions_tag
            ON brainbar_subscriptions(tag)
        """)

        try execute("""
            CREATE TABLE IF NOT EXISTS kg_entities (
                id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                name TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                description TEXT,
                created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                UNIQUE(entity_type, name)
            )
        """)

        try execute("""
            CREATE TABLE IF NOT EXISTS kg_relations (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                properties TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                UNIQUE(source_id, target_id, relation_type)
            )
        """)
    }

    func close() {
        if let db {
            sqlite3_close(db)
            self.db = nil
        }
    }

    private static let allowedPragmas: Set<String> = [
        "journal_mode", "busy_timeout", "cache_size", "synchronous",
        "wal_checkpoint", "page_count", "page_size", "freelist_count"
    ]

    func pragma(_ name: String) throws -> String {
        guard Self.allowedPragmas.contains(name) else {
            throw DBError.invalidPragma(name)
        }
        guard let db else { throw DBError.notOpen }
        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(db, "PRAGMA \(name)", -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }

        guard sqlite3_step(stmt) == SQLITE_ROW else { throw DBError.noResult }
        guard let cStr = sqlite3_column_text(stmt, 0) else { throw DBError.noResult }
        return String(cString: cStr)
    }

    func tableExists(_ name: String) throws -> Bool {
        guard let db else { throw DBError.notOpen }
        var stmt: OpaquePointer?
        let sql = "SELECT count(*) FROM sqlite_master WHERE type IN ('table','view') AND name = ?"
        let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }

        bindText(name, to: stmt, index: 1)
        guard sqlite3_step(stmt) == SQLITE_ROW else { return false }
        return sqlite3_column_int(stmt, 0) > 0
    }

    func insertChunk(
        id: String,
        content: String,
        sessionId: String,
        project: String,
        contentType: String,
        importance: Int,
        tags: String = "[]"
    ) throws {
        guard let db else { throw DBError.notOpen }
        let sql = """
            INSERT OR REPLACE INTO chunks (id, content, metadata, source_file, project, content_type, importance, conversation_id, char_count, tags, summary)
            VALUES (?, ?, '{}', 'brainbar', ?, ?, ?, ?, ?, ?, '')
        """
        try runWriteStatement(on: db, sql: sql, retries: 3) { stmt in
            bindText(id, to: stmt, index: 1)
            bindText(content, to: stmt, index: 2)
            bindText(project, to: stmt, index: 3)
            bindText(contentType, to: stmt, index: 4)
            sqlite3_bind_int(stmt, 5, Int32(importance))
            bindText(sessionId, to: stmt, index: 6)
            sqlite3_bind_int(stmt, 7, Int32(content.count))
            bindText(tags, to: stmt, index: 8)
        }
    }

    func search(
        query: String,
        limit: Int,
        project: String? = nil,
        tag: String? = nil,
        importanceMin: Double? = nil,
        subscriberID: String? = nil,
        unreadOnly: Bool = false
    ) throws -> [[String: Any]] {
        guard let db else { throw DBError.notOpen }
        let sanitized = sanitizeFTS5Query(query)

        var subscribedTags: [String] = []
        var ackFloor: Int64 = 0
        if unreadOnly, let subscriberID, let record = try subscription(agentID: subscriberID) {
            subscribedTags = record.tags
            ackFloor = record.lastAckedSeq
        }

        var conditions = ["chunks_fts MATCH ?"]
        if project != nil { conditions.append("c.project = ?") }
        if let explicitTag = tag {
            conditions.append("c.tags LIKE ?")
            subscribedTags = [explicitTag]
        } else if unreadOnly, !subscribedTags.isEmpty {
            let tagTerms = Array(repeating: "c.tags LIKE ?", count: subscribedTags.count).joined(separator: " OR ")
            conditions.append("(\(tagTerms))")
        }
        if importanceMin != nil { conditions.append("c.importance >= ?") }
        if unreadOnly { conditions.append("c.rowid > ?") }

        // Unread mode needs contiguous rowid ordering for watermark semantics.
        // Normal search uses FTS5 BM25 rank for relevance ordering.
        let orderByClause = unreadOnly ? "c.rowid ASC" : "f.rank"
        let sql = """
            SELECT c.rowid, c.id, c.content, c.project, c.content_type, c.importance,
                   c.created_at, c.summary, c.tags, c.conversation_id, f.rank
            FROM chunks_fts f
            JOIN chunks c ON c.id = f.chunk_id
            WHERE \(conditions.joined(separator: " AND "))
            ORDER BY \(orderByClause)
            LIMIT ?
        """

        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }

        var paramIdx: Int32 = 1
        bindText(sanitized, to: stmt, index: paramIdx)
        paramIdx += 1
        if let project {
            bindText(project, to: stmt, index: paramIdx)
            paramIdx += 1
        }
        if let explicitTag = tag {
            bindText("%\"\(explicitTag)\"%", to: stmt, index: paramIdx)
            paramIdx += 1
        } else if unreadOnly, !subscribedTags.isEmpty {
            for subscribedTag in subscribedTags {
                bindText("%\"\(subscribedTag)\"%", to: stmt, index: paramIdx)
                paramIdx += 1
            }
        }
        if let importanceMin {
            sqlite3_bind_double(stmt, paramIdx, importanceMin)
            paramIdx += 1
        }
        if unreadOnly {
            sqlite3_bind_int64(stmt, paramIdx, ackFloor)
            paramIdx += 1
        }
        sqlite3_bind_int(stmt, paramIdx, Int32(limit))

        var results: [[String: Any]] = []
        var maxRowID: Int64 = 0
        while sqlite3_step(stmt) == SQLITE_ROW {
            let rowID = sqlite3_column_int64(stmt, 0)
            maxRowID = max(maxRowID, rowID)
            // FTS5 rank is negative (lower = better match). Negate for a positive score.
            let rawRank = sqlite3_column_double(stmt, 10)
            let score = max(0, -rawRank)
            results.append([
                "rowid": Int(rowID),
                "chunk_id": columnText(stmt, 1) as Any,
                "content": columnText(stmt, 2) as Any,
                "project": columnText(stmt, 3) as Any,
                "content_type": columnText(stmt, 4) as Any,
                "importance": sqlite3_column_double(stmt, 5),
                "created_at": columnText(stmt, 6) as Any,
                "summary": columnText(stmt, 7) as Any,
                "tags": columnText(stmt, 8) as Any,
                "session_id": columnText(stmt, 9) as Any,
                "score": score
            ])
        }

        if unreadOnly, let subscriberID, maxRowID > 0 {
            try markDelivered(agentID: subscriberID, seq: maxRowID)
        }

        return results
    }

    func store(content: String, tags: [String], importance: Int, source: String) throws -> StoredChunk {
        guard let db else { throw DBError.notOpen }
        let chunkID = "brainbar-\(UUID().uuidString.lowercased().prefix(12))"
        let tagsJSON = (try? encodeJSON(tags)) ?? "[]"
        let sql = """
            INSERT INTO chunks (id, content, metadata, source_file, tags, importance, source, content_type, char_count)
            VALUES (?, ?, '{}', 'brainbar-store', ?, ?, ?, 'user_message', ?)
        """
        try runWriteStatement(on: db, sql: sql, retries: 3) { stmt in
            bindText(chunkID, to: stmt, index: 1)
            bindText(content, to: stmt, index: 2)
            bindText(tagsJSON, to: stmt, index: 3)
            sqlite3_bind_int(stmt, 4, Int32(importance))
            bindText(source, to: stmt, index: 5)
            sqlite3_bind_int(stmt, 6, Int32(content.count))
        }
        return StoredChunk(chunkID: chunkID, rowID: sqlite3_last_insert_rowid(db))
    }

    func upsertSubscription(agentID: String, tags: [String], incrementGeneration: Bool = false) throws -> SubscriberRecord {
        try ensureAuxiliarySchema()
        let now = Self.timestamp()
        try withImmediateTransaction {
            if let existing = try subscription(agentID: agentID) {
                let nextGeneration = incrementGeneration ? existing.generation + 1 : existing.generation
                try executeUpdate(
                    """
                    UPDATE brainbar_agents
                    SET generation = ?, connection_state = 'connected', last_connected_at = ?, disconnected_at = NULL
                    WHERE agent_id = ?
                    """,
                    binds: { stmt in
                        sqlite3_bind_int(stmt, 1, Int32(nextGeneration))
                        bindText(now, to: stmt, index: 2)
                        bindText(agentID, to: stmt, index: 3)
                    }
                )
            } else {
                try executeUpdate(
                    """
                    INSERT INTO brainbar_agents (
                        agent_id, generation, connection_state, last_connected_at, disconnected_at, last_delivered_seq, last_acked_seq
                    ) VALUES (?, 1, 'connected', ?, NULL, 0, 0)
                    """,
                    binds: { stmt in
                        bindText(agentID, to: stmt, index: 1)
                        bindText(now, to: stmt, index: 2)
                    }
                )
            }

            for tag in Set(tags).sorted() where !tag.isEmpty {
                try executeUpdate(
                    """
                    INSERT OR IGNORE INTO brainbar_subscriptions (agent_id, tag)
                    VALUES (?, ?)
                    """,
                    binds: { stmt in
                        bindText(agentID, to: stmt, index: 1)
                        bindText(tag, to: stmt, index: 2)
                    }
                )
            }
        }

        return try subscription(agentID: agentID) ?? SubscriberRecord(
            agentID: agentID,
            generation: 1,
            connectionState: "connected",
            tags: tags.sorted(),
            lastDeliveredSeq: 0,
            lastAckedSeq: 0,
            lastConnectedAt: now,
            disconnectedAt: nil
        )
    }

    func removeSubscription(agentID: String, tags: [String]?) throws -> SubscriberRecord {
        try ensureAuxiliarySchema()
        try withImmediateTransaction {
            try ensureAgentRow(agentID: agentID)
            if let tags, !tags.isEmpty {
                for tag in tags where !tag.isEmpty {
                    try executeUpdate(
                        "DELETE FROM brainbar_subscriptions WHERE agent_id = ? AND tag = ?",
                        binds: { stmt in
                            bindText(agentID, to: stmt, index: 1)
                            bindText(tag, to: stmt, index: 2)
                        }
                    )
                }
            } else {
                try executeUpdate(
                    "DELETE FROM brainbar_subscriptions WHERE agent_id = ?",
                    binds: { stmt in bindText(agentID, to: stmt, index: 1) }
                )
            }
        }

        return try subscription(agentID: agentID) ?? SubscriberRecord(
            agentID: agentID,
            generation: 1,
            connectionState: "disconnected",
            tags: [],
            lastDeliveredSeq: 0,
            lastAckedSeq: 0,
            lastConnectedAt: nil,
            disconnectedAt: nil
        )
    }

    func subscription(agentID: String) throws -> SubscriberRecord? {
        try ensureAuxiliarySchema()
        guard let db else { throw DBError.notOpen }

        var stmt: OpaquePointer?
        let sql = """
            SELECT agent_id, generation, connection_state, last_connected_at, disconnected_at,
                   last_delivered_seq, last_acked_seq
            FROM brainbar_agents
            WHERE agent_id = ?
        """
        let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }
        bindText(agentID, to: stmt, index: 1)

        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
        let tags = try subscriptionTags(agentID: agentID)
        return SubscriberRecord(
            agentID: columnText(stmt, 0) ?? agentID,
            generation: Int(sqlite3_column_int(stmt, 1)),
            connectionState: columnText(stmt, 2) ?? "disconnected",
            tags: tags,
            lastDeliveredSeq: sqlite3_column_int64(stmt, 5),
            lastAckedSeq: sqlite3_column_int64(stmt, 6),
            lastConnectedAt: columnText(stmt, 3),
            disconnectedAt: columnText(stmt, 4)
        )
    }

    func markSubscriberDisconnected(agentID: String) throws {
        try ensureAuxiliarySchema()
        try ensureAgentRow(agentID: agentID)
        try executeUpdate(
            """
            UPDATE brainbar_agents
            SET connection_state = 'disconnected', disconnected_at = ?
            WHERE agent_id = ?
            """,
            binds: { stmt in
                bindText(Self.timestamp(), to: stmt, index: 1)
                bindText(agentID, to: stmt, index: 2)
            }
        )
    }

    func markDelivered(agentID: String, seq: Int64) throws {
        try ensureAuxiliarySchema()
        try ensureAgentRow(agentID: agentID)
        try executeUpdate(
            """
            UPDATE brainbar_agents
            SET last_delivered_seq = MAX(last_delivered_seq, ?), connection_state = 'connected', disconnected_at = NULL
            WHERE agent_id = ?
            """,
            binds: { stmt in
                sqlite3_bind_int64(stmt, 1, seq)
                bindText(agentID, to: stmt, index: 2)
            }
        )
    }

    func acknowledge(agentID: String, seq: Int64) throws {
        try ensureAuxiliarySchema()
        try ensureAgentRow(agentID: agentID)
        try executeUpdate(
            """
            UPDATE brainbar_agents
            SET last_delivered_seq = MAX(last_delivered_seq, ?),
                last_acked_seq = MAX(last_acked_seq, ?),
                connection_state = 'connected',
                disconnected_at = NULL
            WHERE agent_id = ?
            """,
            binds: { stmt in
                sqlite3_bind_int64(stmt, 1, seq)
                sqlite3_bind_int64(stmt, 2, seq)
                bindText(agentID, to: stmt, index: 3)
            }
        )
    }

    func chunkRowID(forChunkID chunkID: String) throws -> Int64? {
        guard let db else { throw DBError.notOpen }
        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(db, "SELECT rowid FROM chunks WHERE id = ?", -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }
        bindText(chunkID, to: stmt, index: 1)
        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
        return sqlite3_column_int64(stmt, 0)
    }

    func unreadCount(agentID: String, tags: [String]? = nil) throws -> Int {
        try ensureAuxiliarySchema()
        let ackFloor = try subscription(agentID: agentID)?.lastAckedSeq ?? 0
        guard let db else { throw DBError.notOpen }

        var conditions = ["rowid > ?"]
        let filterTags = (tags?.isEmpty == false) ? tags! : try subscription(agentID: agentID)?.tags ?? []
        if !filterTags.isEmpty {
            let tagTerms = Array(repeating: "tags LIKE ?", count: filterTags.count).joined(separator: " OR ")
            conditions.append("(\(tagTerms))")
        }

        let sql = "SELECT COUNT(*) FROM chunks WHERE \(conditions.joined(separator: " AND "))"
        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }

        var paramIdx: Int32 = 1
        sqlite3_bind_int64(stmt, paramIdx, ackFloor)
        paramIdx += 1
        for tag in filterTags {
            bindText("%\"\(tag)\"%", to: stmt, index: paramIdx)
            paramIdx += 1
        }

        guard sqlite3_step(stmt) == SQLITE_ROW else { throw DBError.noResult }
        return Int(sqlite3_column_int(stmt, 0))
    }

    func dashboardStats(activityWindowMinutes: Int = 30, bucketCount: Int = 12) throws -> DashboardStats {
        guard bucketCount > 0 else {
            return DashboardStats(
                chunkCount: 0,
                enrichedChunkCount: 0,
                pendingEnrichmentCount: 0,
                enrichmentPercent: 0,
                databaseSizeBytes: databaseSizeBytes(),
                recentActivityBuckets: []
            )
        }

        let counts = try dashboardCounts()
        let chunkCount = counts.chunkCount
        let enrichedChunkCount = counts.enrichedChunkCount
        let pendingEnrichmentCount = max(0, chunkCount - enrichedChunkCount)
        let enrichmentPercent = chunkCount == 0 ? 0 : (Double(enrichedChunkCount) / Double(chunkCount)) * 100
        let recentActivityBuckets = try recentActivityBuckets(
            activityWindowMinutes: activityWindowMinutes,
            bucketCount: bucketCount
        )

        return DashboardStats(
            chunkCount: chunkCount,
            enrichedChunkCount: enrichedChunkCount,
            pendingEnrichmentCount: pendingEnrichmentCount,
            enrichmentPercent: enrichmentPercent,
            databaseSizeBytes: databaseSizeBytes(),
            recentActivityBuckets: recentActivityBuckets
        )
    }

    func dataVersion() throws -> Int {
        guard let db else { throw DBError.notOpen }
        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(db, "PRAGMA data_version", -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_step(stmt) == SQLITE_ROW else { throw DBError.noResult }
        return Int(sqlite3_column_int(stmt, 0))
    }

    func exec(_ sql: String) {
        do {
            try execute(sql)
        } catch {
            NSLog("[BrainBar] SQL error: %@", String(describing: error))
        }
    }

    private func ensureAgentRow(agentID: String) throws {
        if try subscription(agentID: agentID) != nil {
            return
        }
        try executeUpdate(
            """
            INSERT OR IGNORE INTO brainbar_agents (
                agent_id, generation, connection_state, last_connected_at, disconnected_at, last_delivered_seq, last_acked_seq
            ) VALUES (?, 1, 'disconnected', NULL, NULL, 0, 0)
            """,
            binds: { stmt in bindText(agentID, to: stmt, index: 1) }
        )
    }

    private func scalarInt(_ sql: String) throws -> Int {
        guard let db else { throw DBError.notOpen }
        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_step(stmt) == SQLITE_ROW else { throw DBError.noResult }
        return Int(sqlite3_column_int(stmt, 0))
    }

    private func dashboardCounts() throws -> (chunkCount: Int, enrichedChunkCount: Int) {
        guard let db else { throw DBError.notOpen }
        let sql = """
            SELECT
                COUNT(*) AS chunk_count,
                SUM(CASE WHEN enriched_at IS NOT NULL AND TRIM(enriched_at) != '' THEN 1 ELSE 0 END) AS enriched_count
            FROM chunks
        """
        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_step(stmt) == SQLITE_ROW else { throw DBError.noResult }
        return (
            chunkCount: Int(sqlite3_column_int(stmt, 0)),
            enrichedChunkCount: Int(sqlite3_column_int(stmt, 1))
        )
    }

    private func recentActivityBuckets(activityWindowMinutes: Int, bucketCount: Int) throws -> [Int] {
        guard activityWindowMinutes > 0 else { return Array(repeating: 0, count: bucketCount) }
        guard let db else { throw DBError.notOpen }

        let bucketWidthSeconds = max(1, Double(activityWindowMinutes * 60) / Double(bucketCount))
        let windowStart = Date().addingTimeInterval(Double(-activityWindowMinutes * 60))

        var stmt: OpaquePointer?
        let sql = """
            SELECT datetime(created_at)
            FROM chunks
            WHERE datetime(created_at) >= datetime('now', ?)
            ORDER BY datetime(created_at) ASC
        """
        let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }
        bindText("-\(activityWindowMinutes) minutes", to: stmt, index: 1)

        var buckets = Array(repeating: 0, count: bucketCount)
        let formatter = Self.sqliteDateFormatter

        while sqlite3_step(stmt) == SQLITE_ROW {
            guard let createdAtText = columnText(stmt, 0),
                  let createdAt = formatter.date(from: createdAtText) else {
                continue
            }

            let offset = createdAt.timeIntervalSince(windowStart)
            if offset < 0 { continue }
            if offset > Double(activityWindowMinutes * 60) { continue }

            let rawIndex = Int(offset / bucketWidthSeconds)
            let clampedIndex = min(max(rawIndex, 0), bucketCount - 1)
            buckets[clampedIndex] += 1
        }

        return buckets
    }

    private func databaseSizeBytes() -> Int64 {
        let candidates = [path, "\(path)-wal", "\(path)-shm"]
        return candidates.reduce(into: Int64(0)) { total, candidate in
            let attributes = try? FileManager.default.attributesOfItem(atPath: candidate)
            total += (attributes?[.size] as? NSNumber)?.int64Value ?? 0
        }
    }

    private func subscriptionTags(agentID: String) throws -> [String] {
        guard let db else { throw DBError.notOpen }
        var stmt: OpaquePointer?
        let sql = "SELECT tag FROM brainbar_subscriptions WHERE agent_id = ? ORDER BY tag"
        let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }
        bindText(agentID, to: stmt, index: 1)

        var tags: [String] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            if let tag = columnText(stmt, 0) {
                tags.append(tag)
            }
        }
        return tags
    }

    private func withImmediateTransaction<T>(_ body: () throws -> T) throws -> T {
        try execute("BEGIN IMMEDIATE", retries: 3)
        do {
            let result = try body()
            try execute("COMMIT", retries: 3)
            return result
        } catch {
            try? execute("ROLLBACK")
            throw error
        }
    }

    private func executeUpdate(
        _ sql: String,
        binds: (OpaquePointer) -> Void
    ) throws {
        guard let db else { throw DBError.notOpen }
        try runWriteStatement(on: db, sql: sql, retries: 3, bind: binds)
    }

    private func execute(_ sql: String, retries: Int = 0, retryDelayMillis: UInt32 = 250) throws {
        guard let db else { throw DBError.notOpen }
        var attempts = 0
        while true {
            var errMsg: UnsafeMutablePointer<CChar>?
            let rc = sqlite3_exec(db, sql, nil, nil, &errMsg)
            if rc == SQLITE_OK {
                sqlite3_free(errMsg)
                if Self.isMutationStatement(sql) {
                    Self.postDashboardChangeNotification()
                }
                return
            }

            let message = errMsg.map { String(cString: $0) } ?? "unknown error"
            sqlite3_free(errMsg)

            if (rc == SQLITE_BUSY || rc == SQLITE_LOCKED), attempts < retries {
                attempts += 1
                usleep(retryDelayMillis * 1_000)
                continue
            }

            throw DBError.exec(rc, message)
        }
    }

    private func runWriteStatement(
        on handle: OpaquePointer?,
        sql: String,
        retries: Int = 0,
        retryDelayMillis: UInt32 = 250,
        bind: (OpaquePointer) -> Void
    ) throws {
        guard let handle else { throw DBError.notOpen }
        var attempts = 0

        while true {
            var stmt: OpaquePointer?
            let prepareRC = sqlite3_prepare_v2(handle, sql, -1, &stmt, nil)
            guard prepareRC == SQLITE_OK, let stmt else {
                if (prepareRC == SQLITE_BUSY || prepareRC == SQLITE_LOCKED), attempts < retries {
                    attempts += 1
                    usleep(retryDelayMillis * 1_000)
                    continue
                }
                throw DBError.prepare(prepareRC)
            }

            bind(stmt)
            let stepRC = sqlite3_step(stmt)
            sqlite3_finalize(stmt)

            if stepRC == SQLITE_DONE {
                Self.postDashboardChangeNotification()
                return
            }

            if (stepRC == SQLITE_BUSY || stepRC == SQLITE_LOCKED), attempts < retries {
                attempts += 1
                usleep(retryDelayMillis * 1_000)
                continue
            }

            throw DBError.step(stepRC)
        }
    }

    private func openConnection(path: String) throws -> OpaquePointer {
        var handle: OpaquePointer?
        let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX
        let rc = sqlite3_open_v2(path, &handle, flags, nil)
        guard rc == SQLITE_OK, let handle else { throw DBError.open(path, rc) }
        return handle
    }

    private func configureConnection(_ handle: OpaquePointer?) throws {
        guard let handle else { throw DBError.notOpen }
        try executeOnHandle(handle, sql: "PRAGMA journal_mode = WAL")
        try executeOnHandle(handle, sql: "PRAGMA busy_timeout = 5000")
        try executeOnHandle(handle, sql: "PRAGMA cache_size = -64000")
        try executeOnHandle(handle, sql: "PRAGMA synchronous = NORMAL")
        try executeOnHandle(handle, sql: "PRAGMA foreign_keys = ON")
    }

    private func executeOnHandle(_ handle: OpaquePointer, sql: String) throws {
        var errMsg: UnsafeMutablePointer<CChar>?
        let rc = sqlite3_exec(handle, sql, nil, nil, &errMsg)
        if rc == SQLITE_OK {
            sqlite3_free(errMsg)
            return
        }
        let message = errMsg.map { String(cString: $0) } ?? "unknown error"
        sqlite3_free(errMsg)
        throw DBError.exec(rc, message)
    }

    private func encodeJSON(_ array: [String]) throws -> String {
        let data = try JSONSerialization.data(withJSONObject: array)
        return String(data: data, encoding: .utf8) ?? "[]"
    }

    private func columnText(_ stmt: OpaquePointer?, _ col: Int32) -> String? {
        guard let cStr = sqlite3_column_text(stmt, col) else { return nil }
        return String(cString: cStr)
    }

    private func bindText(_ value: String, to stmt: OpaquePointer?, index: Int32) {
        sqlite3_bind_text(stmt, index, value, -1, unsafeBitCast(-1, to: sqlite3_destructor_type.self))
    }

    private static func isMutationStatement(_ sql: String) -> Bool {
        let keyword = sql
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .split(whereSeparator: \.isWhitespace)
            .first?
            .lowercased()

        return [
            "insert", "update", "delete", "replace",
            "create", "drop", "alter", "begin", "commit", "rollback"
        ].contains(keyword ?? "")
    }

    private static func postDashboardChangeNotification() {
        CFNotificationCenterPostNotification(
            CFNotificationCenterGetDarwinNotifyCenter(),
            CFNotificationName(dashboardDidChangeNotification as CFString),
            nil,
            nil,
            true
        )
    }

    private func sanitizeFTS5Query(_ query: String) -> String {
        let tokens = query.split(separator: " ").compactMap { token -> String? in
            let cleaned = token
                .replacingOccurrences(of: "\"", with: "")
                .replacingOccurrences(of: "*", with: "")
                .trimmingCharacters(in: .whitespaces)
            guard !cleaned.isEmpty else { return nil }
            return "\"\(cleaned)\""
        }
        guard !tokens.isEmpty else { return "\"\"" }
        // Implicit AND (space-separated) — matches Python _escape_fts5_query default.
        // FTS5 treats space as AND. Semantic recall comes from vector search (future);
        // FTS5 should maximize precision.
        return tokens.joined(separator: " ")
    }

    private static func timestamp() -> String {
        ISO8601DateFormatter().string(from: Date())
    }

    // MARK: - brain_tags: list unique tags with counts

    func listTags(query: String? = nil, limit: Int = 50) throws -> [[String: Any]] {
        guard let db else { throw DBError.notOpen }
        
        // Use SQL aggregation for better performance with large datasets.
        // json_each extracts individual tags from the JSON array.
        let sql: String
        if query != nil {
            sql = """
                SELECT LOWER(TRIM(json_each.value)) AS tag, COUNT(*) AS count
                FROM chunks, json_each(chunks.tags)
                WHERE json_each.type = 'text'
                  AND LOWER(TRIM(json_each.value)) LIKE ?
                  AND TRIM(json_each.value) != ''
                GROUP BY LOWER(TRIM(json_each.value))
                ORDER BY count DESC
                LIMIT ?
            """
        } else {
            sql = """
                SELECT LOWER(TRIM(json_each.value)) AS tag, COUNT(*) AS count
                FROM chunks, json_each(chunks.tags)
                WHERE json_each.type = 'text'
                  AND TRIM(json_each.value) != ''
                GROUP BY LOWER(TRIM(json_each.value))
                ORDER BY count DESC
                LIMIT ?
            """
        }
        
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DBError.prepare(sqlite3_errcode(db))
        }
        defer { sqlite3_finalize(stmt) }
        
        var paramIdx: Int32 = 1
        if let query {
            bindText("%\(query.lowercased())%", to: stmt, index: paramIdx)
            paramIdx += 1
        }
        sqlite3_bind_int(stmt, paramIdx, Int32(limit))

        var results: [[String: Any]] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            guard let tag = columnText(stmt, 0), !tag.isEmpty else { continue }
            let count = Int(sqlite3_column_int(stmt, 1))
            results.append(["tag": tag as Any, "count": count as Any])
        }
        return results
    }

    // MARK: - brain_update: update chunk importance/tags

    func updateChunk(id: String, importance: Int? = nil, tags: [String]? = nil) throws {
        guard let db else { throw DBError.notOpen }
        var rowsChanged = 0

        if let importance {
            let sql = "UPDATE chunks SET importance = ? WHERE id = ?"
            try runWriteStatement(on: db, sql: sql, retries: 3) { stmt in
                sqlite3_bind_int(stmt, 1, Int32(importance))
                bindText(id, to: stmt, index: 2)
            }
            rowsChanged += Int(sqlite3_changes(db))
        }

        if let tags {
            let tagsJSON = try encodeJSON(tags)
            let sql = "UPDATE chunks SET tags = ? WHERE id = ?"
            try runWriteStatement(on: db, sql: sql, retries: 3) { stmt in
                bindText(tagsJSON, to: stmt, index: 1)
                bindText(id, to: stmt, index: 2)
            }
            rowsChanged += Int(sqlite3_changes(db))
        }

        if rowsChanged == 0 {
            throw DBError.noResult
        }
    }

    // MARK: - brain_expand: get chunk + surrounding session context

    func expandChunk(id: String, before: Int = 3, after: Int = 3) throws -> [String: Any] {
        guard let db else { throw DBError.notOpen }

        // Get the target chunk with its session_id and rowid
        let targetSQL = "SELECT rowid, id, content, conversation_id, project, content_type, importance, created_at, summary, tags FROM chunks WHERE id = ?"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, targetSQL, -1, &stmt, nil) == SQLITE_OK else {
            throw DBError.prepare(sqlite3_errcode(db))
        }
        defer { sqlite3_finalize(stmt) }
        bindText(id, to: stmt, index: 1)
        guard sqlite3_step(stmt) == SQLITE_ROW else { throw DBError.noResult }

        let targetRowID = sqlite3_column_int64(stmt, 0)
        let sessionId = columnText(stmt, 3) ?? ""
        let target: [String: Any] = [
            "chunk_id": columnText(stmt, 1) as Any,
            "content": columnText(stmt, 2) as Any,
            "session_id": sessionId,
            "project": columnText(stmt, 4) as Any,
            "content_type": columnText(stmt, 5) as Any,
            "importance": sqlite3_column_double(stmt, 6),
            "created_at": columnText(stmt, 7) as Any,
            "summary": columnText(stmt, 8) as Any,
            "tags": columnText(stmt, 9) as Any
        ]

        // Get surrounding chunks from same session using two separate queries
        var context: [[String: Any]] = []

        // Before chunks (reverse order, then flip)
        let beforeSQL = "SELECT id, content, content_type, importance, created_at, summary FROM chunks WHERE conversation_id = ? AND rowid < ? ORDER BY rowid DESC LIMIT ?"
        var beforeStmt: OpaquePointer?
        if sqlite3_prepare_v2(db, beforeSQL, -1, &beforeStmt, nil) == SQLITE_OK {
            defer { sqlite3_finalize(beforeStmt) }
            bindText(sessionId, to: beforeStmt, index: 1)
            sqlite3_bind_int64(beforeStmt, 2, targetRowID)
            sqlite3_bind_int(beforeStmt, 3, Int32(before))
            var beforeChunks: [[String: Any]] = []
            while sqlite3_step(beforeStmt) == SQLITE_ROW {
                beforeChunks.append([
                    "chunk_id": columnText(beforeStmt, 0) as Any,
                    "content": columnText(beforeStmt, 1) as Any,
                    "content_type": columnText(beforeStmt, 2) as Any,
                    "importance": sqlite3_column_double(beforeStmt, 3),
                    "created_at": columnText(beforeStmt, 4) as Any,
                    "summary": columnText(beforeStmt, 5) as Any
                ])
            }
            context.append(contentsOf: beforeChunks.reversed())
        }

        // After chunks
        let afterSQL = "SELECT id, content, content_type, importance, created_at, summary FROM chunks WHERE conversation_id = ? AND rowid > ? ORDER BY rowid ASC LIMIT ?"
        var afterStmt: OpaquePointer?
        if sqlite3_prepare_v2(db, afterSQL, -1, &afterStmt, nil) == SQLITE_OK {
            defer { sqlite3_finalize(afterStmt) }
            bindText(sessionId, to: afterStmt, index: 1)
            sqlite3_bind_int64(afterStmt, 2, targetRowID)
            sqlite3_bind_int(afterStmt, 3, Int32(after))
            while sqlite3_step(afterStmt) == SQLITE_ROW {
                context.append([
                    "chunk_id": columnText(afterStmt, 0) as Any,
                    "content": columnText(afterStmt, 1) as Any,
                    "content_type": columnText(afterStmt, 2) as Any,
                    "importance": sqlite3_column_double(afterStmt, 3),
                    "created_at": columnText(afterStmt, 4) as Any,
                    "summary": columnText(afterStmt, 5) as Any
                ])
            }
        }

        return ["target": target, "context": context]
    }

    // MARK: - brain_entity: insert + lookup entities

    func insertEntity(id: String, type: String, name: String, metadata: String = "{}") throws {
        guard let db else { throw DBError.notOpen }
        let sql = "INSERT OR REPLACE INTO kg_entities (id, entity_type, name, metadata) VALUES (?, ?, ?, ?)"
        try runWriteStatement(on: db, sql: sql, retries: 3) { stmt in
            bindText(id, to: stmt, index: 1)
            bindText(type, to: stmt, index: 2)
            bindText(name, to: stmt, index: 3)
            bindText(metadata, to: stmt, index: 4)
        }
    }

    func insertRelation(sourceId: String, targetId: String, relationType: String) throws {
        guard let db else { throw DBError.notOpen }
        let relId = "\(sourceId)-\(relationType)-\(targetId)"
        let sql = "INSERT OR REPLACE INTO kg_relations (id, source_id, target_id, relation_type) VALUES (?, ?, ?, ?)"
        try runWriteStatement(on: db, sql: sql, retries: 3) { stmt in
            bindText(relId, to: stmt, index: 1)
            bindText(sourceId, to: stmt, index: 2)
            bindText(targetId, to: stmt, index: 3)
            bindText(relationType, to: stmt, index: 4)
        }
    }

    func lookupEntity(query: String) throws -> [String: Any]? {
        guard let db else { throw DBError.notOpen }

        // First try exact name match
        let exactSQL = "SELECT id, entity_type, name, metadata, description FROM kg_entities WHERE name = ? LIMIT 1"
        var exactStmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, exactSQL, -1, &exactStmt, nil) == SQLITE_OK else {
            throw DBError.prepare(sqlite3_errcode(db))
        }
        bindText(query, to: exactStmt, index: 1)

        var entityId: String?
        var result: [String: Any]?

        if sqlite3_step(exactStmt) == SQLITE_ROW {
            entityId = columnText(exactStmt, 0)
            result = [
                "entity_id": entityId as Any,
                "entity_type": columnText(exactStmt, 1) as Any,
                "name": columnText(exactStmt, 2) as Any,
                "metadata": columnText(exactStmt, 3) as Any,
                "description": columnText(exactStmt, 4) as Any
            ]
        }
        sqlite3_finalize(exactStmt)

        // If no exact match, try LIKE
        if result == nil {
            let likeSQL = "SELECT id, entity_type, name, metadata, description FROM kg_entities WHERE name LIKE ? LIMIT 1"
            var likeStmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, likeSQL, -1, &likeStmt, nil) == SQLITE_OK else {
                throw DBError.prepare(sqlite3_errcode(db))
            }
            bindText("%\(query)%", to: likeStmt, index: 1)

            if sqlite3_step(likeStmt) == SQLITE_ROW {
                entityId = columnText(likeStmt, 0)
                result = [
                    "entity_id": entityId as Any,
                    "entity_type": columnText(likeStmt, 1) as Any,
                    "name": columnText(likeStmt, 2) as Any,
                    "metadata": columnText(likeStmt, 3) as Any,
                    "description": columnText(likeStmt, 4) as Any
                ]
            }
            sqlite3_finalize(likeStmt)
        }

        // Get relations for found entity
        if let entityId, result != nil {
            let relSQL = """
                SELECT r.relation_type, r.target_id, e.name
                FROM kg_relations r
                LEFT JOIN kg_entities e ON e.id = r.target_id
                WHERE r.source_id = ?
                UNION ALL
                SELECT r.relation_type, r.source_id, e.name
                FROM kg_relations r
                LEFT JOIN kg_entities e ON e.id = r.source_id
                WHERE r.target_id = ?
            """
            var relStmt: OpaquePointer?
            if sqlite3_prepare_v2(db, relSQL, -1, &relStmt, nil) == SQLITE_OK {
                defer { sqlite3_finalize(relStmt) }
                bindText(entityId, to: relStmt, index: 1)
                bindText(entityId, to: relStmt, index: 2)
                var relations: [[String: Any]] = []
                while sqlite3_step(relStmt) == SQLITE_ROW {
                    let targetName = columnText(relStmt, 2) ?? ""
                    relations.append([
                        "relation_type": columnText(relStmt, 0) as Any,
                        "target_id": columnText(relStmt, 1) as Any,
                        "target_name": targetName as Any,
                        "target": ["name": targetName] as [String: Any]
                    ])
                }
                result?["relations"] = relations
            }
        }

        return result
    }

    // MARK: - brain_recall

    func recallSession(sessionId: String, limit: Int = 20) throws -> [[String: Any]] {
        guard let db else { throw DBError.notOpen }
        let sql = """
            SELECT id, content, project, content_type, importance, created_at, summary, tags
            FROM chunks WHERE conversation_id = ? ORDER BY rowid DESC LIMIT ?
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DBError.prepare(sqlite3_errcode(db))
        }
        defer { sqlite3_finalize(stmt) }
        bindText(sessionId, to: stmt, index: 1)
        sqlite3_bind_int(stmt, 2, Int32(limit))

        var results: [[String: Any]] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            results.append([
                "chunk_id": columnText(stmt, 0) as Any,
                "content": columnText(stmt, 1) as Any,
                "project": columnText(stmt, 2) as Any,
                "content_type": columnText(stmt, 3) as Any,
                "importance": sqlite3_column_double(stmt, 4),
                "created_at": columnText(stmt, 5) as Any,
                "summary": columnText(stmt, 6) as Any,
                "tags": columnText(stmt, 7) as Any,
                "score": 1.0  // session recall has no relevance scoring
            ])
        }
        return results
    }

    func recallStats() throws -> [String: Any] {
        guard let db else { throw DBError.notOpen }

        func queryInt(_ sql: String) throws -> Int {
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw DBError.prepare(sqlite3_errcode(db))
            }
            defer { sqlite3_finalize(stmt) }
            guard sqlite3_step(stmt) == SQLITE_ROW else { return 0 }
            return Int(sqlite3_column_int64(stmt, 0))
        }

        let totalChunks = try queryInt("SELECT COUNT(*) FROM chunks")
        let totalEntities = try queryInt("SELECT COUNT(*) FROM kg_entities")
        let totalRelations = try queryInt("SELECT COUNT(*) FROM kg_relations")
        let enrichedChunks = try queryInt("SELECT COUNT(*) FROM chunks WHERE enriched_at IS NOT NULL")
        let totalProjects = try queryInt("SELECT COUNT(DISTINCT project) FROM chunks")
        let projects = try queryStrings("SELECT DISTINCT project FROM chunks WHERE project IS NOT NULL AND project != '' ORDER BY project ASC LIMIT 12")
        let contentTypes = try queryStrings("SELECT DISTINCT content_type FROM chunks WHERE content_type IS NOT NULL AND content_type != '' ORDER BY content_type ASC")

        return [
            "total_chunks": totalChunks,
            "total_entities": totalEntities,
            "total_relations": totalRelations,
            "enriched_chunks": enrichedChunks,
            "total_projects": totalProjects,
            "enrichment_pct": totalChunks > 0 ? Double(enrichedChunks) / Double(totalChunks) * 100.0 : 0.0,
            "projects": projects,
            "content_types": contentTypes
        ]
    }

    private func queryStrings(_ sql: String) throws -> [String] {
        guard let db else { throw DBError.notOpen }
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DBError.prepare(sqlite3_errcode(db))
        }
        defer { sqlite3_finalize(stmt) }

        var values: [String] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            if let value = columnText(stmt, 0), !value.isEmpty {
                values.append(value)
            }
        }
        return values
    }

    // MARK: - brain_digest: rule-based entity extraction

    func digest(content: String) throws -> [String: Any] {
        guard db != nil else { throw DBError.notOpen }

        // Rule-based entity extraction
        var entities: [String] = []
        var urls: [String] = []
        var codeIds: [String] = []

        // Extract capitalized multi-word names (2-3 words, each capitalized)
        let namePattern = try NSRegularExpression(pattern: "\\b([A-Z][a-z]+(?:\\s+[A-Z][a-z]+){1,2})\\b")
        let nsContent = content as NSString
        let nameMatches = namePattern.matches(in: content, range: NSRange(location: 0, length: nsContent.length))
        for match in nameMatches {
            let name = nsContent.substring(with: match.range)
            // Filter common non-entity phrases
            let skip = ["The", "This", "That", "These", "Those", "Here", "There", "When", "What", "Which", "Where", "How"]
            if !skip.contains(where: { name.hasPrefix($0 + " ") }) {
                entities.append(name)
            }
        }

        // Extract PascalCase identifiers (code names like BrainLayer, MCPRouter)
        let pascalPattern = try NSRegularExpression(pattern: "\\b([A-Z][a-z]+[A-Z][a-zA-Z]+)\\b")
        let pascalMatches = pascalPattern.matches(in: content, range: NSRange(location: 0, length: nsContent.length))
        for match in pascalMatches {
            entities.append(nsContent.substring(with: match.range))
        }

        // Extract URLs
        let urlPattern = try NSRegularExpression(pattern: "https?://[^\\s,)]+")
        let urlMatches = urlPattern.matches(in: content, range: NSRange(location: 0, length: nsContent.length))
        for match in urlMatches {
            urls.append(nsContent.substring(with: match.range))
        }

        // Extract code identifiers (snake_case, dotted paths)
        let codePattern = try NSRegularExpression(pattern: "\\b([a-z][a-z_]+\\.[a-z_]+)\\b")
        let codeMatches = codePattern.matches(in: content, range: NSRange(location: 0, length: nsContent.length))
        for match in codeMatches {
            codeIds.append(nsContent.substring(with: match.range))
        }

        // Deduplicate
        entities = Array(Set(entities))
        urls = Array(Set(urls))
        codeIds = Array(Set(codeIds))

        // Store the digest as a chunk
        let digestSummary = "Digest: \(entities.count) entities, \(urls.count) URLs, \(codeIds.count) code refs"
        
        do {
            let stored = try store(
                content: content.prefix(500) + (content.count > 500 ? "..." : ""),
                tags: ["digest"] + entities.prefix(5).map { $0 },
                importance: 5,
                source: "digest"
            )
            
            return [
                "mode": "digest",
                "entities": entities,
                "entities_created": entities.count,
                "urls": urls,
                "code_identifiers": codeIds,
                "chunks_created": 1,
                "relations_created": 0,
                "chunk_id": stored.chunkID,
                "summary": digestSummary
            ]
        } catch {
            return [
                "mode": "digest",
                "entities": entities,
                "entities_created": entities.count,
                "urls": urls,
                "code_identifiers": codeIds,
                "chunks_created": 0,
                "relations_created": 0,
                "error": "Storage failed: \(error.localizedDescription)",
                "summary": "\(digestSummary) (storage failed)"
            ]
        }
    }

    private static let sqliteDateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
        return formatter
    }()

    enum DBError: LocalizedError {
        case notOpen
        case open(String, Int32)
        case prepare(Int32)
        case step(Int32)
        case exec(Int32, String)
        case noResult
        case invalidPragma(String)

        var errorDescription: String? {
            switch self {
            case .notOpen: return "Database not open"
            case .open(let path, let rc): return "SQLite open failed at \(path): \(rc)"
            case .prepare(let rc): return "SQLite prepare failed: \(rc)"
            case .step(let rc): return "SQLite step failed: \(rc)"
            case .exec(let rc, let message): return "SQLite exec failed: \(rc) (\(message))"
            case .noResult: return "No result"
            case .invalidPragma(let name): return "PRAGMA '\(name)' not in allowlist"
            }
        }
    }

    deinit {
        close()
    }
}
