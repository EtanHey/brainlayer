// BrainDatabase.swift — SQLite database layer for BrainBar.
//
// Wraps SQLite3 directly. BrainBar now keeps its pub/sub metadata in the main
// BrainLayer database so agent state and chunk writes share one durable store.

import Darwin
import Foundation
import SQLite3

final class BrainDatabase: @unchecked Sendable {
    static let dashboardDidChangeNotification = "com.brainlayer.brainbar.database-changed"
    private static let previewExpression = """
        trim(substr(replace(replace(replace(coalesce(nullif(summary, ''), content), char(10), ' '), char(13), ' '), char(9), ' '), 1, 220))
    """
    private static let ftsColumns = "content, summary, tags, resolved_query, chunk_id UNINDEXED"
    private static let ftsOptions = "prefix='2 3 4', tokenize='unicode61 remove_diacritics 2'"

    struct DashboardStats: Sendable, Equatable {
        let chunkCount: Int
        let enrichedChunkCount: Int
        let pendingEnrichmentCount: Int
        let enrichmentPercent: Double
        let enrichmentRatePerMinute: Double
        let databaseSizeBytes: Int64
        let recentActivityBuckets: [Int]
        let recentEnrichmentBuckets: [Int]
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

    struct FlushedPendingStore: Sendable {
        let storedChunk: StoredChunk
        let content: String
        let tags: [String]
        let importance: Int
    }

    struct PendingStoreItem: Codable, Sendable {
        let content: String
        let tags: [String]
        let importance: Int
        let source: String
        let queueID: String?
        let queuedAt: String?

        enum CodingKeys: String, CodingKey {
            case content
            case tags
            case importance
            case source
            case queueID = "queue_id"
            case queuedAt = "queued_at"
        }

        init(
            content: String,
            tags: [String],
            importance: Int,
            source: String,
            queueID: String? = nil,
            queuedAt: String? = nil
        ) {
            self.content = content
            self.tags = tags
            self.importance = importance
            self.source = source
            self.queueID = queueID
            self.queuedAt = queuedAt
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            content = try container.decode(String.self, forKey: .content)
            tags = try container.decodeIfPresent([String].self, forKey: .tags) ?? []
            importance = try container.decodeIfPresent(Int.self, forKey: .importance) ?? 5
            source = try container.decodeIfPresent(String.self, forKey: .source) ?? "mcp"
            queueID = try container.decodeIfPresent(String.self, forKey: .queueID)
            queuedAt = try container.decodeIfPresent(String.self, forKey: .queuedAt)
        }
    }

    struct ConversationChunk: Sendable, Equatable, Identifiable {
        let chunkID: String
        let content: String
        let contentType: String
        let importance: Double
        let createdAt: String
        let summary: String
        let isTarget: Bool

        var id: String { chunkID }
    }

    struct ExpandedConversation: Sendable, Equatable, Identifiable {
        let target: ConversationChunk
        let entries: [ConversationChunk]

        var id: String { target.chunkID }
    }

    private var db: OpaquePointer?
    private let path: String
    private static let pendingStoreFileLock = NSLock()
    private(set) var isOpen = false

    init(path: String) {
        self.path = path
        openAndConfigure()
    }

    private func openAndConfigure() {
        do {
            db = try openConnection(path: path)
            NSLog("[BrainBar] Connection opened, configuring...")
            try configureConnection(db)
            NSLog("[BrainBar] Connection configured, checking schema...")
            // AIDEV-NOTE: Skip ensureSchema if chunks table already exists (Python creates all tables).
            // CREATE TABLE IF NOT EXISTS still acquires a RESERVED lock which blocks on watch agent.
            if (try? tableExists("chunks")) == true {
                NSLog("[BrainBar] Schema already exists — skipping ensureSchema")
                try ensureMigrations()
            } else {
                try ensureSchema()
                NSLog("[BrainBar] Schema created")
            }
            isOpen = true
            NSLog("[BrainBar] Database ready")
        } catch {
            NSLog("[BrainBar] Failed to open/configure database at %@: %@", path, String(describing: error))
        }
    }

    // tableExists defined at line ~237 (existing method)

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
                    preview_text TEXT,
                    importance REAL DEFAULT 5,
                    intent TEXT,
                    enriched_at TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
        }

        try ensureChunkColumns()
        try ensurePreviewTextTriggers()
        try rebuildFTSTableIfNeeded()
        try backfillPreviewText()

        try execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_created_at
            ON chunks(created_at)
        """)

        try ensurePendingStoreQueueIndex()

        try ensureAuxiliarySchema()
        try refreshSearchStatistics()
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

        try execute("""
            CREATE TABLE IF NOT EXISTS kg_entity_chunks (
                entity_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                relevance REAL DEFAULT 1.0,
                PRIMARY KEY (entity_id, chunk_id)
            )
        """)

        try execute("""
            CREATE INDEX IF NOT EXISTS idx_kg_ec_entity
            ON kg_entity_chunks(entity_id)
        """)

        try execute("""
            CREATE TABLE IF NOT EXISTS injection_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                query TEXT NOT NULL,
                chunk_ids TEXT NOT NULL DEFAULT '[]',
                token_count INTEGER NOT NULL DEFAULT 0
            )
        """)

        try execute("""
            CREATE INDEX IF NOT EXISTS idx_injection_events_session_timestamp
            ON injection_events(session_id, timestamp DESC)
        """)

        // Filtered VIEW excludes co_occurs_with noise from KG queries
        try execute("""
            CREATE VIEW IF NOT EXISTS kg_relations_typed AS
            SELECT id, source_id, target_id, relation_type, properties, created_at
            FROM kg_relations
            WHERE relation_type != 'co_occurs_with'
        """)
    }

    /// Lightweight migrations that run even when the full schema already exists.
    /// Safe to call repeatedly — all statements use IF NOT EXISTS.
    private func ensureMigrations() throws {
        // PR 1: Filtered VIEW for KG relations (excludes co_occurs_with)
        try execute("""
            CREATE VIEW IF NOT EXISTS kg_relations_typed AS
            SELECT id, source_id, target_id, relation_type, properties, created_at
            FROM kg_relations
            WHERE relation_type != 'co_occurs_with'
        """)

        try ensurePendingStoreQueueIndex()
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
            INSERT OR REPLACE INTO chunks (id, content, metadata, source_file, project, content_type, importance, conversation_id, char_count, tags, summary, preview_text)
            VALUES (?, ?, '{}', 'brainbar', ?, ?, ?, ?, ?, ?, '', ?)
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
            bindText(Self.previewText(summary: "", content: content), to: stmt, index: 9)
        }
        try refreshSearchStatistics()
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
                "preview_text": Self.previewText(summary: columnText(stmt, 7), content: columnText(stmt, 2)),
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

    func store(
        content: String,
        tags: [String],
        importance: Int,
        source: String,
        queueID: String? = nil,
        refreshStatistics: Bool = true
    ) throws -> StoredChunk {
        guard let db else { throw DBError.notOpen }
        let chunkID = "brainbar-\(UUID().uuidString.lowercased().prefix(12))"
        let tagsJSON = (try? encodeJSON(tags)) ?? "[]"
        let metadataJSON = Self.storeMetadataJSON(queueID: queueID)
        let sql = """
            INSERT INTO chunks (id, content, metadata, source_file, tags, importance, source, content_type, char_count, preview_text)
            VALUES (?, ?, ?, 'brainbar-store', ?, ?, ?, 'user_message', ?, ?)
        """
        try runWriteStatement(on: db, sql: sql, retries: 3) { stmt in
            bindText(chunkID, to: stmt, index: 1)
            bindText(content, to: stmt, index: 2)
            bindText(metadataJSON, to: stmt, index: 3)
            bindText(tagsJSON, to: stmt, index: 4)
            sqlite3_bind_int(stmt, 5, Int32(importance))
            bindText(source, to: stmt, index: 6)
            sqlite3_bind_int(stmt, 7, Int32(content.count))
            bindText(Self.previewText(summary: "", content: content), to: stmt, index: 8)
        }
        if refreshStatistics {
            refreshSearchStatisticsBestEffort()
        }
        return StoredChunk(chunkID: chunkID, rowID: sqlite3_last_insert_rowid(db))
    }

    /// Async wrapper for store() — runs DB write off the main thread.
    func storeAsync(content: String, tags: [String], importance: Int, source: String) async throws -> StoredChunk {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async { [self] in
                do {
                    let result = try self.store(content: content, tags: tags, importance: importance, source: source)
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    func shouldQueueStoreError(_ error: Error) -> Bool {
        guard let dbError = error as? DBError else { return false }
        switch dbError {
        case .prepare(let rc), .step(let rc):
            return Self.isRetryableQueueErrorCode(rc)
        case .exec(let rc, _):
            return Self.isRetryableQueueErrorCode(rc)
        case .notOpen, .open, .noResult, .invalidPragma:
            return false
        }
    }

    func queuePendingStore(content: String, tags: [String], importance: Int, source: String) throws {
        Self.pendingStoreFileLock.lock()
        defer { Self.pendingStoreFileLock.unlock() }

        let path = pendingStorePath()
        try FileManager.default.createDirectory(
            at: path.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let item = PendingStoreItem(
            content: content,
            tags: tags,
            importance: importance,
            source: source,
            queueID: UUID().uuidString.lowercased(),
            queuedAt: Self.timestamp()
        )
        var line = try JSONEncoder().encode(item)
        line.append(0x0A)
        try appendPendingStoreLine(line, to: path)
    }

    func flushPendingStores() -> [FlushedPendingStore] {
        Self.pendingStoreFileLock.lock()
        defer { Self.pendingStoreFileLock.unlock() }

        let path = pendingStorePath()
        guard FileManager.default.fileExists(atPath: path.path) else { return [] }
        guard let lines = readPendingStoreLines(at: path) else {
            NSLog("[BrainBar] Failed to read pending stores queue at %@", path.path)
            return []
        }
        guard !lines.isEmpty else { return [] }

        let decoder = JSONDecoder()
        var flushed: [FlushedPendingStore] = []
        var remaining: [Data] = []

        for (lineIndex, line) in lines.enumerated() {
            let item: PendingStoreItem
            do {
                item = try decoder.decode(PendingStoreItem.self, from: line)
            } catch {
                remaining.append(line)
                continue
            }

            let queueID = Self.pendingStoreQueueID(for: item, lineIndex: lineIndex)
            let replayLine = Self.pendingStoreReplayLine(for: item, queueID: queueID) ?? line
            do {
                if try hasStoredQueuedItem(queueID: queueID) {
                    continue
                }
            } catch {
                NSLog("[BrainBar] Failed pending store dedupe lookup for %@: %@", queueID, String(describing: error))
                remaining.append(replayLine)
                continue
            }

            do {
                let stored = try store(
                    content: item.content,
                    tags: item.tags,
                    importance: item.importance,
                    source: item.source,
                    queueID: queueID,
                    refreshStatistics: false
                )
                flushed.append(
                    FlushedPendingStore(
                        storedChunk: stored,
                        content: item.content,
                        tags: item.tags,
                        importance: item.importance
                    )
                )
            } catch {
                NSLog("[BrainBar] Failed to flush pending store item: %@", String(describing: error))
                remaining.append(replayLine)
            }
        }

        rewritePendingStoreFile(path: path, remainingLines: remaining)
        if !flushed.isEmpty {
            refreshSearchStatisticsBestEffort()
        }
        return flushed
    }

    func searchCandidates(
        query: String,
        limit: Int,
        project: String? = nil,
        tag: String? = nil,
        importanceMin: Double? = nil,
        subscriberID: String? = nil,
        unreadOnly: Bool = false
    ) throws -> [SearchQueryCandidate] {
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

        let orderByClause = unreadOnly ? "c.rowid ASC" : "f.rank"
        let sql = """
            SELECT c.rowid, c.id, c.preview_text, f.rank, c.created_at, c.project, c.importance
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

        var results: [SearchQueryCandidate] = []
        var maxRowID: Int64 = 0
        while sqlite3_step(stmt) == SQLITE_ROW {
            let rowID = sqlite3_column_int64(stmt, 0)
            maxRowID = max(maxRowID, rowID)
            let rawRank = sqlite3_column_double(stmt, 3)
            let score = max(0, -rawRank)
            results.append(
                SearchQueryCandidate(
                    id: columnText(stmt, 1) ?? "",
                    previewText: columnText(stmt, 2) ?? "",
                    lexicalScore: score,
                    date: columnText(stmt, 4) ?? "",
                    project: columnText(stmt, 5) ?? "",
                    importance: Int(sqlite3_column_int(stmt, 6))
                )
            )
        }

        if unreadOnly, let subscriberID, maxRowID > 0 {
            try markDelivered(agentID: subscriberID, seq: maxRowID)
        }

        return results
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
                enrichmentRatePerMinute: 0,
                databaseSizeBytes: databaseSizeBytes(),
                recentActivityBuckets: [],
                recentEnrichmentBuckets: []
            )
        }

        let counts = try dashboardCounts()
        let chunkCount = counts.chunkCount
        let enrichedChunkCount = counts.enrichedChunkCount
        let pendingEnrichmentCount = max(0, chunkCount - enrichedChunkCount)
        let enrichmentPercent = chunkCount == 0 ? 0 : (Double(enrichedChunkCount) / Double(chunkCount)) * 100
        let enrichmentRateWindowMinutes = 1
        let enrichmentRatePerMinute = try recentEnrichmentRatePerMinute(windowMinutes: enrichmentRateWindowMinutes)
        let recentActivityBuckets = try recentActivityBuckets(
            activityWindowMinutes: activityWindowMinutes,
            bucketCount: bucketCount
        )
        let recentEnrichmentBuckets = try recentEnrichmentBuckets(
            activityWindowMinutes: activityWindowMinutes,
            bucketCount: bucketCount
        )

        return DashboardStats(
            chunkCount: chunkCount,
            enrichedChunkCount: enrichedChunkCount,
            pendingEnrichmentCount: pendingEnrichmentCount,
            enrichmentPercent: enrichmentPercent,
            enrichmentRatePerMinute: enrichmentRatePerMinute,
            databaseSizeBytes: databaseSizeBytes(),
            recentActivityBuckets: recentActivityBuckets,
            recentEnrichmentBuckets: recentEnrichmentBuckets
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

    private func recentEnrichmentBuckets(activityWindowMinutes: Int, bucketCount: Int) throws -> [Int] {
        guard activityWindowMinutes > 0 else { return Array(repeating: 0, count: bucketCount) }
        guard let db else { throw DBError.notOpen }

        let bucketWidthSeconds = max(1, Double(activityWindowMinutes * 60) / Double(bucketCount))
        let windowStart = Date().addingTimeInterval(Double(-activityWindowMinutes * 60))

        var stmt: OpaquePointer?
        let sql = """
            SELECT datetime(enriched_at)
            FROM chunks
            WHERE enriched_at IS NOT NULL
              AND TRIM(enriched_at) != ''
              AND datetime(enriched_at) >= datetime('now', ?)
            ORDER BY datetime(enriched_at) ASC
        """
        let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }
        bindText("-\(activityWindowMinutes) minutes", to: stmt, index: 1)

        var buckets = Array(repeating: 0, count: bucketCount)
        let formatter = Self.sqliteDateFormatter

        while sqlite3_step(stmt) == SQLITE_ROW {
            guard let enrichedAtText = columnText(stmt, 0),
                  let enrichedAt = formatter.date(from: enrichedAtText) else {
                continue
            }

            let offset = enrichedAt.timeIntervalSince(windowStart)
            if offset < 0 { continue }
            if offset > Double(activityWindowMinutes * 60) { continue }

            let rawIndex = Int(offset / bucketWidthSeconds)
            let clampedIndex = min(max(rawIndex, 0), bucketCount - 1)
            buckets[clampedIndex] += 1
        }

        return buckets
    }

    private func recentEnrichmentRatePerMinute(windowMinutes: Int) throws -> Double {
        guard windowMinutes > 0 else { return 0 }
        guard let db else { throw DBError.notOpen }

        var stmt: OpaquePointer?
        let sql = """
            SELECT COUNT(*)
            FROM chunks
            WHERE enriched_at IS NOT NULL
              AND TRIM(enriched_at) != ''
              AND datetime(enriched_at) >= datetime('now', ?)
        """
        let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }
        bindText("-\(windowMinutes) minutes", to: stmt, index: 1)

        guard sqlite3_step(stmt) == SQLITE_ROW else { throw DBError.noResult }
        let count = Double(sqlite3_column_int(stmt, 0))
        return count / Double(windowMinutes)
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
        // AIDEV-NOTE: READWRITE is needed for save/search. The async init in BrainBarApp.swift
        // ensures this runs on a background queue so it doesn't block the menu item.
        let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX
        let rc = sqlite3_open_v2(path, &handle, flags, nil)
        guard rc == SQLITE_OK, let handle else { throw DBError.open(path, rc) }
        NSLog("[BrainBar] Database opened READWRITE")
        return handle
    }

    private func configureConnection(_ handle: OpaquePointer?) throws {
        guard let handle else { throw DBError.notOpen }
        // AIDEV-NOTE: busy_timeout FIRST — 30s because watch agent holds locks during enrichment.
        try executeOnHandle(handle, sql: "PRAGMA busy_timeout = 30000")
        // AIDEV-NOTE: Skip journal_mode=WAL if already set — the PRAGMA itself needs a write lock
        // which blocks indefinitely when the watch agent is active. WAL is already set by Python.
        let currentMode = queryPragma(handle, name: "journal_mode")
        if currentMode?.lowercased() != "wal" {
            try executeOnHandle(handle, sql: "PRAGMA journal_mode = WAL")
        } else {
            NSLog("[BrainBar] journal_mode already WAL — skipping PRAGMA")
        }
        try executeOnHandle(handle, sql: "PRAGMA cache_size = -64000")
        try executeOnHandle(handle, sql: "PRAGMA synchronous = NORMAL")
        try executeOnHandle(handle, sql: "PRAGMA foreign_keys = ON")
    }

    private func queryPragma(_ handle: OpaquePointer, name: String) -> String? {
        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(handle, "PRAGMA \(name)", -1, &stmt, nil)
        guard rc == SQLITE_OK, let stmt else { return nil }
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
        guard let cStr = sqlite3_column_text(stmt, 0) else { return nil }
        return String(cString: cStr)
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
            return "\"\(cleaned)\"*"
        }
        guard !tokens.isEmpty else { return "\"\"" }
        // Prefix search keeps typeahead fast while the reranker narrows the
        // bounded candidate set later in the search pipeline.
        return tokens.joined(separator: " ")
    }

    private func ensureChunkColumns() throws {
        guard let db else { throw DBError.notOpen }
        let existingColumns = try tableColumns(name: "chunks", on: db)
        if !existingColumns.contains("preview_text") {
            try execute("ALTER TABLE chunks ADD COLUMN preview_text TEXT")
        }
    }

    private func ensurePreviewTextTriggers() throws {
        try execute("DROP TRIGGER IF EXISTS chunks_preview_text_insert")
        try execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_preview_text_insert
            AFTER INSERT ON chunks
            WHEN new.preview_text IS NULL OR trim(new.preview_text) = ''
            BEGIN
                UPDATE chunks
                SET preview_text = \(Self.previewExpression)
                WHERE rowid = new.rowid;
            END
        """)

        try execute("DROP TRIGGER IF EXISTS chunks_preview_text_update")
        try execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_preview_text_update
            AFTER UPDATE OF content, summary ON chunks
            BEGIN
                UPDATE chunks
                SET preview_text = \(Self.previewExpression)
                WHERE rowid = new.rowid;
            END
        """)
    }

    private func rebuildFTSTableIfNeeded() throws {
        let needsRebuild = try ftsTableNeedsRebuild()
        if needsRebuild {
            try execute("DROP TRIGGER IF EXISTS chunks_fts_insert")
            try execute("DROP TRIGGER IF EXISTS chunks_fts_delete")
            try execute("DROP TRIGGER IF EXISTS chunks_fts_update")
            try execute("DROP TABLE IF EXISTS chunks_fts")
        }

        try execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                \(Self.ftsColumns),
                \(Self.ftsOptions)
            )
        """)
        try execute("DELETE FROM chunks_fts")
        try execute("""
            INSERT INTO chunks_fts(content, summary, tags, resolved_query, chunk_id)
            SELECT content, summary, tags, NULL, id FROM chunks
        """)

        try execute("DROP TRIGGER IF EXISTS chunks_fts_insert")
        try execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(content, summary, tags, resolved_query, chunk_id)
                VALUES (new.content, new.summary, new.tags, NULL, new.id);
            END
        """)

        try execute("DROP TRIGGER IF EXISTS chunks_fts_delete")
        try execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
                DELETE FROM chunks_fts WHERE chunk_id = old.id;
            END
        """)

        try execute("DROP TRIGGER IF EXISTS chunks_fts_update")
        try execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON chunks BEGIN
                DELETE FROM chunks_fts WHERE chunk_id = old.id;
                INSERT INTO chunks_fts(content, summary, tags, resolved_query, chunk_id)
                VALUES (new.content, new.summary, new.tags, NULL, new.id);
            END
        """)
    }

    private func ftsTableNeedsRebuild() throws -> Bool {
        guard let db else { throw DBError.notOpen }
        guard let sql = try sqliteMasterSQL(name: "chunks_fts", on: db) else { return true }
        let normalizedSQL = sql.lowercased()
        return !normalizedSQL.contains("prefix='2 3 4'") ||
            !normalizedSQL.contains("tokenize='unicode61 remove_diacritics 2'") ||
            !normalizedSQL.contains("summary") ||
            !normalizedSQL.contains("resolved_query")
    }

    private func backfillPreviewText() throws {
        try execute("""
            UPDATE chunks
            SET preview_text = \(Self.previewExpression)
            WHERE preview_text IS NULL OR trim(preview_text) = ''
        """)
    }

    private func refreshSearchStatistics() throws {
        try execute("ANALYZE chunks")
        try execute("ANALYZE chunks_fts")
    }

    private func refreshSearchStatisticsBestEffort() {
        do {
            try refreshSearchStatistics()
        } catch {
            NSLog("[BrainBar] Non-fatal search statistics refresh failure: %@", String(describing: error))
        }
    }

    private func ensurePendingStoreQueueIndex() throws {
        try execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_brainbar_queue_id
            ON chunks(json_extract(metadata, '$.brainbar_queue_id'))
            WHERE json_extract(metadata, '$.brainbar_queue_id') IS NOT NULL
        """)
    }

    private func pendingStorePath() -> URL {
        if let override = ProcessInfo.processInfo.environment["BRAINBAR_PENDING_STORES_PATH"],
           !override.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return URL(fileURLWithPath: override)
        }
        return URL(fileURLWithPath: path).deletingLastPathComponent().appendingPathComponent("pending-stores.jsonl")
    }

    private func appendPendingStoreLine(_ line: Data, to path: URL) throws {
        let fd = open(path.path, O_WRONLY | O_CREAT | O_APPEND, 0o644)
        guard fd >= 0 else {
            let message = String(cString: strerror(errno))
            throw DBError.exec(SQLITE_CANTOPEN, "failed to open pending store queue for append: \(message)")
        }
        defer { Darwin.close(fd) }

        try line.withUnsafeBytes { rawBuffer in
            guard var baseAddress = rawBuffer.baseAddress else { return }
            var remaining = rawBuffer.count

            while remaining > 0 {
                let written = Darwin.write(fd, baseAddress, remaining)
                if written < 0 {
                    if errno == EINTR {
                        continue
                    }
                    let message = String(cString: strerror(errno))
                    throw DBError.exec(SQLITE_IOERR, "failed to append pending store queue line: \(message)")
                }

                remaining -= written
                baseAddress = baseAddress.advanced(by: written)
            }
        }
    }

    private func readPendingStoreLines(at path: URL) -> [Data]? {
        guard let data = try? Data(contentsOf: path) else {
            return nil
        }

        var lines: [Data] = []
        var start = data.startIndex

        func appendLine(_ range: Range<Data.Index>) {
            guard !range.isEmpty else { return }
            var line = Data(data[range])
            if line.last == 0x0D {
                line.removeLast()
            }
            guard !line.isEmpty else { return }
            lines.append(line)
        }

        for index in data.indices where data[index] == 0x0A {
            appendLine(start..<index)
            start = data.index(after: index)
        }

        if start < data.endIndex {
            appendLine(start..<data.endIndex)
        }

        return lines
    }

    private static func storeMetadataJSON(queueID: String?) -> String {
        guard let queueID = normalizedQueueID(queueID) else { return "{}" }
        let payload = ["brainbar_queue_id": queueID]
        guard let data = try? JSONEncoder().encode(payload),
              let text = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return text
    }

    private static func normalizedQueueID(_ queueID: String?) -> String? {
        guard let queueID = queueID?.trimmingCharacters(in: .whitespacesAndNewlines),
              !queueID.isEmpty else {
            return nil
        }
        return queueID
    }

    private static func pendingStoreQueueID(for item: PendingStoreItem, lineIndex: Int) -> String {
        if let queueID = normalizedQueueID(item.queueID) {
            return queueID
        }
        return deterministicPendingStoreQueueID(
            content: item.content,
            tags: item.tags,
            importance: item.importance,
            source: item.source,
            lineIndex: lineIndex
        )
    }

    private static func pendingStoreReplayLine(for item: PendingStoreItem, queueID: String) -> Data? {
        let replayItem = PendingStoreItem(
            content: item.content,
            tags: item.tags,
            importance: item.importance,
            source: item.source,
            queueID: queueID,
            queuedAt: item.queuedAt
        )
        guard var data = try? JSONEncoder().encode(replayItem) else {
            return nil
        }
        data.append(0x0A)
        return data
    }

    private static func deterministicPendingStoreQueueID(
        content: String,
        tags: [String],
        importance: Int,
        source: String,
        lineIndex: Int
    ) -> String {
        var hash: UInt64 = 0xcbf29ce484222325

        func mix(_ byte: UInt8) {
            hash ^= UInt64(byte)
            hash &*= 0x100000001b3
        }

        func mix(_ data: Data) {
            for byte in data {
                mix(byte)
            }
        }

        func mix(length: Int) {
            var value = UInt64(length).littleEndian
            withUnsafeBytes(of: &value) { bytes in
                mix(Data(bytes))
            }
        }

        func mix(_ string: String) {
            let data = Data(string.utf8)
            mix(length: data.count)
            mix(data)
        }

        mix(content)
        mix(length: tags.count)
        for tag in tags {
            mix(tag)
        }
        var encodedImportance = Int64(importance).littleEndian
        withUnsafeBytes(of: &encodedImportance) { bytes in
            mix(Data(bytes))
        }
        mix(source)
        var encodedLineIndex = Int64(lineIndex).littleEndian
        withUnsafeBytes(of: &encodedLineIndex) { bytes in
            mix(Data(bytes))
        }

        return String(format: "brainbar-pending-%016llx", hash)
    }

    private static func isRetryableQueueErrorCode(_ rc: Int32) -> Bool {
        let primaryCode = rc & 0xFF
        return primaryCode == SQLITE_BUSY || primaryCode == SQLITE_LOCKED
    }

    private func hasStoredQueuedItem(queueID: String) throws -> Bool {
        guard let db else { throw DBError.notOpen }
        var stmt: OpaquePointer?
        let sql = """
            SELECT 1
            FROM chunks
            WHERE json_extract(metadata, '$.brainbar_queue_id') = ?
            LIMIT 1
        """
        let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }
        bindText(queueID, to: stmt, index: 1)
        return sqlite3_step(stmt) == SQLITE_ROW
    }

    private func rewritePendingStoreFile(path: URL, remainingLines: [Data]) {
        guard !remainingLines.isEmpty else {
            try? FileManager.default.removeItem(at: path)
            return
        }
        do {
            var data = Data()
            for line in remainingLines {
                data.append(line)
                data.append(0x0A)
            }
            try data.write(to: path, options: .atomic)
        } catch {
            NSLog("[BrainBar] Failed to rewrite pending stores queue: %@", String(describing: error))
        }
    }

    private func tableColumns(name: String, on db: OpaquePointer) throws -> Set<String> {
        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(db, "PRAGMA table_info(\(name))", -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }

        var columns: Set<String> = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            if let columnName = columnText(stmt, 1) {
                columns.insert(columnName)
            }
        }
        return columns
    }

    private func sqliteMasterSQL(name: String, on db: OpaquePointer) throws -> String? {
        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(db, "SELECT sql FROM sqlite_master WHERE name = ?", -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }

        bindText(name, to: stmt, index: 1)
        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
        return columnText(stmt, 0)
    }

    private static func previewText(summary: String?, content: String?) -> String {
        let preferred = (summary?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false) ? summary! : (content ?? "")
        let flattened = preferred
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\r", with: " ")
            .replacingOccurrences(of: "\t", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return String(flattened.prefix(220))
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
        guard db != nil else { throw DBError.notOpen }
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
        var beforeContext: [[String: Any]] = []
        var afterContext: [[String: Any]] = []

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
            beforeContext = beforeChunks.reversed()
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
                afterContext.append([
                    "chunk_id": columnText(afterStmt, 0) as Any,
                    "content": columnText(afterStmt, 1) as Any,
                    "content_type": columnText(afterStmt, 2) as Any,
                    "importance": sqlite3_column_double(afterStmt, 3),
                    "created_at": columnText(afterStmt, 4) as Any,
                    "summary": columnText(afterStmt, 5) as Any
                ])
            }
        }

        return [
            "target": target,
            "before_context": beforeContext,
            "after_context": afterContext,
            "context": beforeContext + afterContext
        ]
    }

    func expandedConversation(id: String, before: Int = 3, after: Int = 3) throws -> ExpandedConversation {
        let payload = try expandChunk(id: id, before: before, after: after)
        guard let targetPayload = payload["target"] as? [String: Any] else {
            throw DBError.noResult
        }

        let target = ConversationChunk(
            chunkID: targetPayload["chunk_id"] as? String ?? "",
            content: targetPayload["content"] as? String ?? "",
            contentType: targetPayload["content_type"] as? String ?? "",
            importance: targetPayload["importance"] as? Double ?? 0,
            createdAt: targetPayload["created_at"] as? String ?? "",
            summary: targetPayload["summary"] as? String ?? "",
            isTarget: true
        )

        let beforeContext = ((payload["before_context"] as? [[String: Any]]) ?? []).map { item in
            ConversationChunk(
                chunkID: item["chunk_id"] as? String ?? "",
                content: item["content"] as? String ?? "",
                contentType: item["content_type"] as? String ?? "",
                importance: item["importance"] as? Double ?? 0,
                createdAt: item["created_at"] as? String ?? "",
                summary: item["summary"] as? String ?? "",
                isTarget: false
            )
        }
        let afterContext = ((payload["after_context"] as? [[String: Any]]) ?? []).map { item in
            ConversationChunk(
                chunkID: item["chunk_id"] as? String ?? "",
                content: item["content"] as? String ?? "",
                contentType: item["content_type"] as? String ?? "",
                importance: item["importance"] as? Double ?? 0,
                createdAt: item["created_at"] as? String ?? "",
                summary: item["summary"] as? String ?? "",
                isTarget: false
            )
        }

        return ExpandedConversation(target: target, entries: beforeContext + [target] + afterContext)
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

        // Get typed relations for found entity (excludes co_occurs_with noise)
        if let entityId, result != nil {
            let relSQL = """
                SELECT r.relation_type, e.name, 'outgoing' AS direction
                FROM kg_relations_typed r
                LEFT JOIN kg_entities e ON e.id = r.target_id
                WHERE r.source_id = ?
                UNION ALL
                SELECT r.relation_type, e.name, 'incoming' AS direction
                FROM kg_relations_typed r
                LEFT JOIN kg_entities e ON e.id = r.source_id
                WHERE r.target_id = ?
                ORDER BY 1
                LIMIT 20
            """
            var relStmt: OpaquePointer?
            if sqlite3_prepare_v2(db, relSQL, -1, &relStmt, nil) == SQLITE_OK {
                defer { sqlite3_finalize(relStmt) }
                bindText(entityId, to: relStmt, index: 1)
                bindText(entityId, to: relStmt, index: 2)
                var relations: [[String: Any]] = []
                while sqlite3_step(relStmt) == SQLITE_ROW {
                    let targetName = columnText(relStmt, 1) ?? ""
                    let direction = columnText(relStmt, 2) ?? "outgoing"
                    relations.append([
                        "relation_type": columnText(relStmt, 0) as Any,
                        "target_name": targetName as Any,
                        "direction": direction as Any
                    ])
                }
                result?["relations"] = relations
            }
        }

        return result
    }

    // MARK: - KG fact lookup for search augmentation

    struct KGFact {
        let relatedEntity: String
        let relationType: String
        let entityType: String
        let direction: String  // "outgoing" or "incoming"
    }

    /// Pure SQL KG fact lookup — no embeddings needed.
    /// Returns typed relations for an entity, excluding co_occurs_with.
    func lookupEntityFacts(entityName: String, limit: Int = 20) throws -> [KGFact] {
        guard let db else { throw DBError.notOpen }

        let sql = """
            SELECT e2.name, r.relation_type, e2.entity_type, 'outgoing' AS direction
            FROM kg_relations_typed r
            JOIN kg_entities e1 ON r.source_id = e1.id
            JOIN kg_entities e2 ON r.target_id = e2.id
            WHERE LOWER(e1.name) = LOWER(?1)
            UNION ALL
            SELECT e1.name, r.relation_type, e1.entity_type, 'incoming'
            FROM kg_relations_typed r
            JOIN kg_entities e1 ON r.source_id = e1.id
            JOIN kg_entities e2 ON r.target_id = e2.id
            WHERE LOWER(e2.name) = LOWER(?1)
            ORDER BY 2
            LIMIT ?2
        """

        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DBError.prepare(sqlite3_errcode(db))
        }
        defer { sqlite3_finalize(stmt) }
        bindText(entityName, to: stmt, index: 1)
        sqlite3_bind_int(stmt, 2, Int32(limit))

        var facts: [KGFact] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            facts.append(KGFact(
                relatedEntity: columnText(stmt, 0) ?? "",
                relationType: columnText(stmt, 1) ?? "",
                entityType: columnText(stmt, 2) ?? "",
                direction: columnText(stmt, 3) ?? "outgoing"
            ))
        }
        return facts
    }

    /// Expose the raw db pointer for EntityCache loading.
    var dbHandle: OpaquePointer? { db }

    // MARK: - Knowledge Graph bulk queries

    struct KGEntityRow: Equatable {
        let id: String
        let name: String
        let entityType: String
        let description: String?
        let importance: Double
    }

    struct KGRelationRow: Equatable {
        let id: String
        let sourceId: String
        let targetId: String
        let relationType: String
    }

    struct KGChunkRow: Equatable {
        let chunkID: String
        let snippet: String
        let importance: Int
        let relevance: Double
    }

    func fetchKGEntities(limit: Int = 500) throws -> [KGEntityRow] {
        guard let db else { throw DBError.notOpen }
        // Only return entities that participate in at least one semantic relation.
        // Without this, the graph is mostly disconnected noise.
        let sql = """
            SELECT e.id, e.name, e.entity_type, e.description,
                   COALESCE(CAST(json_extract(e.metadata, '$.importance') AS REAL), 5.0) AS importance
            FROM kg_entities e
            WHERE EXISTS (
                SELECT 1 FROM kg_relations r
                WHERE r.relation_type != 'co_occurs_with'
                AND (r.source_id = e.id OR r.target_id = e.id)
            )
            ORDER BY importance DESC
            LIMIT ?
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DBError.prepare(sqlite3_errcode(db))
        }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_int(stmt, 1, Int32(limit))

        var rows: [KGEntityRow] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            rows.append(KGEntityRow(
                id: columnText(stmt, 0) ?? "",
                name: columnText(stmt, 1) ?? "",
                entityType: columnText(stmt, 2) ?? "",
                description: columnText(stmt, 3),
                importance: sqlite3_column_double(stmt, 4)
            ))
        }
        return rows
    }

    func fetchKGRelations() throws -> [KGRelationRow] {
        guard let db else { throw DBError.notOpen }
        // Exclude co_occurs_with — these are auto-generated from text proximity
        // and produce noisy, often incorrect edges in the graph view.
        let sql = "SELECT id, source_id, target_id, relation_type FROM kg_relations WHERE relation_type != 'co_occurs_with'"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DBError.prepare(sqlite3_errcode(db))
        }
        defer { sqlite3_finalize(stmt) }

        var rows: [KGRelationRow] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            rows.append(KGRelationRow(
                id: columnText(stmt, 0) ?? "",
                sourceId: columnText(stmt, 1) ?? "",
                targetId: columnText(stmt, 2) ?? "",
                relationType: columnText(stmt, 3) ?? ""
            ))
        }
        return rows
    }

    func linkEntityChunk(entityId: String, chunkId: String, relevance: Double = 1.0) throws {
        guard let db else { throw DBError.notOpen }
        let sql = "INSERT OR REPLACE INTO kg_entity_chunks (entity_id, chunk_id, relevance) VALUES (?, ?, ?)"
        try runWriteStatement(on: db, sql: sql, retries: 3) { stmt in
            bindText(entityId, to: stmt, index: 1)
            bindText(chunkId, to: stmt, index: 2)
            sqlite3_bind_double(stmt, 3, relevance)
        }
    }

    func fetchEntityChunks(entityId: String, limit: Int = 20) throws -> [KGChunkRow] {
        guard let db else { throw DBError.notOpen }
        let sql = """
            SELECT c.id, COALESCE(NULLIF(c.summary, ''), substr(c.content, 1, 200)) AS snippet,
                   c.importance, ec.relevance
            FROM kg_entity_chunks ec
            JOIN chunks c ON c.id = ec.chunk_id
            WHERE ec.entity_id = ?
            ORDER BY ec.relevance DESC
            LIMIT ?
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DBError.prepare(sqlite3_errcode(db))
        }
        defer { sqlite3_finalize(stmt) }
        bindText(entityId, to: stmt, index: 1)
        sqlite3_bind_int(stmt, 2, Int32(limit))

        var rows: [KGChunkRow] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            rows.append(KGChunkRow(
                chunkID: columnText(stmt, 0) ?? "",
                snippet: columnText(stmt, 1) ?? "",
                importance: Int(sqlite3_column_int(stmt, 2)),
                relevance: sqlite3_column_double(stmt, 3)
            ))
        }
        return rows
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

    // MARK: - Injection events

    @discardableResult
    func recordInjectionEvent(sessionID: String, query: String, chunkIDs: [String], tokenCount: Int, timestamp: String? = nil) -> InjectionEvent {
        guard let db else {
            return InjectionEvent(id: 0, sessionID: sessionID, timestamp: "", query: query, chunkIDs: chunkIDs, tokenCount: tokenCount)
        }
        let chunkJSON = (try? JSONSerialization.data(withJSONObject: chunkIDs)).flatMap { String(data: $0, encoding: .utf8) } ?? "[]"
        let ts = timestamp ?? Self.sqliteDateFormatter.string(from: Date())
        let sql = "INSERT INTO injection_events (session_id, timestamp, query, chunk_ids, token_count) VALUES (?, ?, ?, ?, ?)"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            return InjectionEvent(id: 0, sessionID: sessionID, timestamp: ts, query: query, chunkIDs: chunkIDs, tokenCount: tokenCount)
        }
        defer { sqlite3_finalize(stmt) }
        bindText(sessionID, to: stmt, index: 1)
        bindText(ts, to: stmt, index: 2)
        bindText(query, to: stmt, index: 3)
        bindText(chunkJSON, to: stmt, index: 4)
        sqlite3_bind_int(stmt, 5, Int32(tokenCount))
        let stepRC = sqlite3_step(stmt)
        let rowID = sqlite3_last_insert_rowid(db)
        if stepRC == SQLITE_DONE {
            Self.postDashboardChangeNotification()
        }
        return InjectionEvent(id: rowID, sessionID: sessionID, timestamp: ts, query: query, chunkIDs: chunkIDs, tokenCount: tokenCount)
    }

    // MARK: - Injection event listing

    func listInjectionEvents(sessionID: String? = nil, limit: Int = 20) throws -> [InjectionEvent] {
        guard let db else { throw DBError.notOpen }
        var sql = "SELECT id, session_id, timestamp, query, chunk_ids, token_count FROM injection_events"
        if sessionID != nil { sql += " WHERE session_id = ?" }
        sql += " ORDER BY timestamp DESC LIMIT ?"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DBError.prepare(sqlite3_errcode(db))
        }
        defer { sqlite3_finalize(stmt) }
        var idx: Int32 = 1
        if let sessionID {
            bindText(sessionID, to: stmt, index: idx)
            idx += 1
        }
        sqlite3_bind_int(stmt, idx, Int32(limit))

        var events: [InjectionEvent] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            let row: [String: Any] = [
                "id": Int64(sqlite3_column_int64(stmt, 0)),
                "session_id": columnText(stmt, 1) as Any,
                "timestamp": columnText(stmt, 2) as Any,
                "query": columnText(stmt, 3) as Any,
                "chunk_ids": columnText(stmt, 4) as Any,
                "token_count": Int(sqlite3_column_int(stmt, 5))
            ]
            if let event = try? InjectionEvent(row: row) {
                events.append(event)
            }
        }
        return events
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
