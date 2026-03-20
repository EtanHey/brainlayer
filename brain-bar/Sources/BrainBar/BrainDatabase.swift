// BrainDatabase.swift — SQLite database layer for BrainBar.
//
// Wraps SQLite3 C API directly (no external dependencies).
// Configures: WAL mode, FTS5, busy_timeout=5000, cache_size=-64000, synchronous=NORMAL.
// Single-writer: only BrainBar writes. Concurrent reads are safe (WAL).
//
// IMPORTANT: Schema matches the production BrainLayer DB (312K chunks).
// Column names: id (not chunk_id), conversation_id (not session_id),
// source_file (NOT NULL), metadata (JSON), etc.

import Foundation
import SQLite3

final class BrainDatabase: @unchecked Sendable {
    struct SubscriberRecord: Sendable {
        let agentID: String
        let tags: [String]
        let lastSeen: String?
        let lastDeliveredAt: String?
        let lastConnectedAt: String?
        let disconnectedAt: String?
    }

    private var db: OpaquePointer?
    private var metadataDB: OpaquePointer?
    private let path: String
    private let metadataPath: String
    /// Whether the database opened successfully.
    private(set) var isOpen: Bool = false

    init(path: String) {
        self.path = path
        self.metadataPath = Self.makeMetadataPath(for: path)
        openAndConfigure()
    }

    private func openAndConfigure() {
        do {
            db = try openConnection(path: path)
            metadataDB = try openConnection(path: metadataPath)
        } catch {
            NSLog("[BrainBar] Failed to open database(s) at %@ / %@: %@", path, metadataPath, String(describing: error))
            return
        }
        isOpen = true

        do {
            try configureConnection(db)
            try configureConnection(metadataDB)
            try attachMetadataDatabase()
        } catch {
            NSLog("[BrainBar] Failed to configure database(s) at %@ / %@: %@", path, metadataPath, String(describing: error))
        }

        do {
            try ensureSchema()
        } catch {
            NSLog("[BrainBar] Failed to ensure schema at %@: %@", path, String(describing: error))
        }
    }

    private func ensureSchema() throws {
        // If chunks table already exists (production DB), don't touch schema.
        if (try? tableExists("chunks")) != true {
            // New/test DB — create production-compatible schema
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

            // FTS5 — matches production: content, summary, tags, resolved_query, chunk_id UNINDEXED
            try execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    content, summary, tags, resolved_query, chunk_id UNINDEXED
                )
            """)

            // FTS sync triggers — match production (no explicit rowid; JOIN uses chunk_id)
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

        try ensureAuxiliarySchema()
    }

    private func ensureAuxiliarySchema() throws {
        if (try? auxiliaryTableExists("agent_subscriptions")) == true,
           (try? auxiliaryTableExists("agent_reads")) == true {
            return
        }

        // Lightweight BrainBar-specific durability for subscriptions and read receipts.
        try execute("""
            CREATE TABLE IF NOT EXISTS agent_subscriptions (
                agent_id TEXT PRIMARY KEY,
                tags TEXT NOT NULL DEFAULT '[]',
                last_seen TEXT,
                last_delivered_at TEXT,
                last_connected_at TEXT,
                disconnected_at TEXT
            )
        """, on: metadataDB)

        try execute("""
            CREATE TABLE IF NOT EXISTS agent_reads (
                agent_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                read_at TEXT NOT NULL,
                PRIMARY KEY (agent_id, chunk_id)
            )
        """, on: metadataDB)

        try execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_reads_agent_id
            ON agent_reads(agent_id, read_at DESC)
        """, on: metadataDB)

        try execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_reads_chunk_id
            ON agent_reads(chunk_id)
        """, on: metadataDB)
    }

    func close() {
        if let metadataDB {
            sqlite3_close(metadataDB)
            self.metadataDB = nil
        }
        if let db {
            sqlite3_close(db)
            self.db = nil
        }
    }

    // MARK: - PRAGMA queries

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

    // MARK: - Table existence

    func tableExists(_ name: String) throws -> Bool {
        try tableExists(name, on: db)
    }

    func auxiliaryTableExists(_ name: String) throws -> Bool {
        try tableExists(name, on: metadataDB)
    }

    private func tableExists(_ name: String, on handle: OpaquePointer?) throws -> Bool {
        guard let handle else { throw DBError.notOpen }
        var stmt: OpaquePointer?
        let sql = "SELECT count(*) FROM sqlite_master WHERE type IN ('table','view') AND name = ?"
        let rc = sqlite3_prepare_v2(handle, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }

        sqlite3_bind_text(stmt, 1, name, -1, unsafeBitCast(-1, to: sqlite3_destructor_type.self))
        guard sqlite3_step(stmt) == SQLITE_ROW else { return false }
        return sqlite3_column_int(stmt, 0) > 0
    }

    // MARK: - Insert chunk (production schema)

    func insertChunk(id: String, content: String, sessionId: String, project: String, contentType: String, importance: Int, tags: String = "[]") throws {
        guard let db else { throw DBError.notOpen }
        let sql = """
            INSERT OR REPLACE INTO chunks (id, content, metadata, source_file, project, content_type, importance, conversation_id, char_count, tags, summary)
            VALUES (?, ?, '{}', 'brainbar', ?, ?, ?, ?, ?, ?, '')
        """
        try runWriteStatement(on: db, sql: sql, retries: 3) { stmt in
            let TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
            sqlite3_bind_text(stmt, 1, id, -1, TRANSIENT)
            sqlite3_bind_text(stmt, 2, content, -1, TRANSIENT)
            sqlite3_bind_text(stmt, 3, project, -1, TRANSIENT)
            sqlite3_bind_text(stmt, 4, contentType, -1, TRANSIENT)
            sqlite3_bind_int(stmt, 5, Int32(importance))
            sqlite3_bind_text(stmt, 6, sessionId, -1, TRANSIENT)
            sqlite3_bind_int(stmt, 7, Int32(content.count))
            sqlite3_bind_text(stmt, 8, tags, -1, TRANSIENT)
        }
    }

    // MARK: - FTS5 Search (production schema)

    func search(query: String, limit: Int, project: String? = nil, tag: String? = nil, importanceMin: Double? = nil, subscriberID: String? = nil, unreadOnly: Bool = false) throws -> [[String: Any]] {
        guard let db else { throw DBError.notOpen }
        if unreadOnly {
            try ensureAuxiliarySchema()
        }

        let sanitized = sanitizeFTS5Query(query)

        // Build WHERE clause dynamically with optional filters
        var conditions = ["chunks_fts MATCH ?"]
        if project != nil { conditions.append("c.project = ?") }
        if tag != nil { conditions.append("c.tags LIKE ?") }
        if importanceMin != nil { conditions.append("c.importance >= ?") }
        if unreadOnly { conditions.append("r.chunk_id IS NULL") }

        let whereClause = conditions.joined(separator: " AND ")
        let joinClause = unreadOnly ? "LEFT JOIN brainbar_meta.agent_reads r ON r.chunk_id = c.id AND r.agent_id = ?" : ""
        let sql = """
            SELECT c.id, c.content, c.project, c.content_type, c.importance,
                   c.created_at, c.summary, c.tags, c.conversation_id
            FROM chunks_fts f
            JOIN chunks c ON c.id = f.chunk_id
            \(joinClause)
            WHERE \(whereClause)
            ORDER BY rank
            LIMIT ?
        """

        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }

        let TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
        var paramIdx: Int32 = 1
        if unreadOnly {
            bindOptionalText(subscriberID, to: stmt, index: paramIdx)
            paramIdx += 1
        }
        sqlite3_bind_text(stmt, paramIdx, sanitized, -1, TRANSIENT); paramIdx += 1
        if let project {
            sqlite3_bind_text(stmt, paramIdx, project, -1, TRANSIENT); paramIdx += 1
        }
        if let tag {
            // Tags stored as JSON array string, e.g. '["bug-fix","auth"]'. Use LIKE for containment.
            let pattern = "%\"\(tag)\"%"
            sqlite3_bind_text(stmt, paramIdx, pattern, -1, TRANSIENT); paramIdx += 1
        }
        if let importanceMin {
            sqlite3_bind_double(stmt, paramIdx, importanceMin); paramIdx += 1
        }
        sqlite3_bind_int(stmt, paramIdx, Int32(limit))

        var results: [[String: Any]] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            var row: [String: Any] = [:]
            row["chunk_id"] = columnText(stmt, 0)
            row["content"] = columnText(stmt, 1)
            row["project"] = columnText(stmt, 2)
            row["content_type"] = columnText(stmt, 3)
            row["importance"] = sqlite3_column_double(stmt, 4)
            row["created_at"] = columnText(stmt, 5)
            row["summary"] = columnText(stmt, 6)
            row["tags"] = columnText(stmt, 7)
            row["session_id"] = columnText(stmt, 8)
            results.append(row)
        }

        return results
    }


    // MARK: - Store (brain_store, production schema)

    func store(content: String, tags: [String], importance: Int, source: String) throws -> String {
        let id = "brainbar-\(UUID().uuidString.lowercased().prefix(12))"
        let tagsJSON: String
        if let data = try? JSONSerialization.data(withJSONObject: tags),
           let str = String(data: data, encoding: .utf8) {
            tagsJSON = str
        } else {
            tagsJSON = "[]"
        }

        guard let db else { throw DBError.notOpen }
        let sql = """
            INSERT INTO chunks (id, content, metadata, source_file, tags, importance, source, content_type, char_count)
            VALUES (?, ?, '{}', 'brainbar-store', ?, ?, ?, 'user_message', ?)
        """
        try runWriteStatement(on: db, sql: sql, retries: 3) { stmt in
            let TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
            sqlite3_bind_text(stmt, 1, id, -1, TRANSIENT)
            sqlite3_bind_text(stmt, 2, content, -1, TRANSIENT)
            sqlite3_bind_text(stmt, 3, tagsJSON, -1, TRANSIENT)
            sqlite3_bind_int(stmt, 4, Int32(importance))
            sqlite3_bind_text(stmt, 5, source, -1, TRANSIENT)
            sqlite3_bind_int(stmt, 6, Int32(content.count))
        }

        return id
    }

    func upsertSubscription(agentID: String, tags: [String]) throws -> SubscriberRecord {
        try ensureAuxiliarySchema()
        let existing = try subscription(agentID: agentID)
        let mergedTags = Array(Set((existing?.tags ?? []) + tags)).sorted()
        let tagsJSON = try encodeJSON(mergedTags)
        let now = Self.timestamp()

        guard let metadataDB else { throw DBError.notOpen }
        var stmt: OpaquePointer?
        let sql = """
            INSERT INTO agent_subscriptions (agent_id, tags, last_connected_at, disconnected_at)
            VALUES (?, ?, ?, NULL)
            ON CONFLICT(agent_id) DO UPDATE SET
                tags = excluded.tags,
                last_connected_at = excluded.last_connected_at,
                disconnected_at = NULL
        """
        let rc = sqlite3_prepare_v2(metadataDB, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }

        let TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
        sqlite3_bind_text(stmt, 1, agentID, -1, TRANSIENT)
        sqlite3_bind_text(stmt, 2, tagsJSON, -1, TRANSIENT)
        sqlite3_bind_text(stmt, 3, now, -1, TRANSIENT)

        let stepRC = sqlite3_step(stmt)
        guard stepRC == SQLITE_DONE else { throw DBError.step(stepRC) }

        return try subscription(agentID: agentID) ?? SubscriberRecord(
            agentID: agentID,
            tags: mergedTags,
            lastSeen: nil,
            lastDeliveredAt: nil,
            lastConnectedAt: now,
            disconnectedAt: nil
        )
    }

    func removeSubscription(agentID: String, tags: [String]?) throws -> SubscriberRecord {
        try ensureAuxiliarySchema()
        let existing = try subscription(agentID: agentID) ?? SubscriberRecord(
            agentID: agentID,
            tags: [],
            lastSeen: nil,
            lastDeliveredAt: nil,
            lastConnectedAt: nil,
            disconnectedAt: nil
        )
        let updatedTags: [String]
        if let tags, !tags.isEmpty {
            let removalSet = Set(tags)
            updatedTags = existing.tags.filter { !removalSet.contains($0) }.sorted()
        } else {
            updatedTags = []
        }

        let tagsJSON = try encodeJSON(updatedTags)
        guard let metadataDB else { throw DBError.notOpen }
        var stmt: OpaquePointer?
        let sql = """
            INSERT INTO agent_subscriptions (agent_id, tags)
            VALUES (?, ?)
            ON CONFLICT(agent_id) DO UPDATE SET tags = excluded.tags
        """
        let rc = sqlite3_prepare_v2(metadataDB, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }

        let TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
        sqlite3_bind_text(stmt, 1, agentID, -1, TRANSIENT)
        sqlite3_bind_text(stmt, 2, tagsJSON, -1, TRANSIENT)

        let stepRC = sqlite3_step(stmt)
        guard stepRC == SQLITE_DONE else { throw DBError.step(stepRC) }

        return try subscription(agentID: agentID) ?? SubscriberRecord(
            agentID: agentID,
            tags: updatedTags,
            lastSeen: existing.lastSeen,
            lastDeliveredAt: existing.lastDeliveredAt,
            lastConnectedAt: existing.lastConnectedAt,
            disconnectedAt: existing.disconnectedAt
        )
    }

    func subscription(agentID: String) throws -> SubscriberRecord? {
        try ensureAuxiliarySchema()
        guard let metadataDB else { throw DBError.notOpen }
        var stmt: OpaquePointer?
        let sql = """
            SELECT agent_id, tags, last_seen, last_delivered_at, last_connected_at, disconnected_at
            FROM agent_subscriptions
            WHERE agent_id = ?
        """
        let rc = sqlite3_prepare_v2(metadataDB, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }

        let TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
        sqlite3_bind_text(stmt, 1, agentID, -1, TRANSIENT)

        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
        return SubscriberRecord(
            agentID: columnText(stmt, 0) ?? agentID,
            tags: decodeJSONArray(columnText(stmt, 1)),
            lastSeen: columnText(stmt, 2),
            lastDeliveredAt: columnText(stmt, 3),
            lastConnectedAt: columnText(stmt, 4),
            disconnectedAt: columnText(stmt, 5)
        )
    }

    func markSubscriberDisconnected(agentID: String) throws {
        try ensureAuxiliarySchema()
        try updateSubscriptionTimestamps(
            agentID: agentID,
            lastSeen: nil,
            lastDeliveredAt: nil,
            disconnectedAt: Self.timestamp()
        )
    }

    func markChunkRead(agentID: String, chunkID: String) throws {
        try ensureAuxiliarySchema()
        let now = Self.timestamp()
        guard let metadataDB else { throw DBError.notOpen }
        var stmt: OpaquePointer?
        let sql = """
            INSERT INTO agent_reads (agent_id, chunk_id, read_at)
            VALUES (?, ?, ?)
            ON CONFLICT(agent_id, chunk_id) DO UPDATE SET read_at = excluded.read_at
        """
        let rc = sqlite3_prepare_v2(metadataDB, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }

        let TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
        sqlite3_bind_text(stmt, 1, agentID, -1, TRANSIENT)
        sqlite3_bind_text(stmt, 2, chunkID, -1, TRANSIENT)
        sqlite3_bind_text(stmt, 3, now, -1, TRANSIENT)

        let stepRC = sqlite3_step(stmt)
        guard stepRC == SQLITE_DONE else { throw DBError.step(stepRC) }

        try updateSubscriptionTimestamps(
            agentID: agentID,
            lastSeen: now,
            lastDeliveredAt: now,
            disconnectedAt: nil
        )
    }

    func unreadCount(agentID: String, tags: [String]? = nil) throws -> Int {
        try ensureAuxiliarySchema()
        guard let db else { throw DBError.notOpen }

        let tagClause: String
        if let tags, !tags.isEmpty {
            let orTerms = Array(repeating: "c.tags LIKE ?", count: tags.count).joined(separator: " OR ")
            tagClause = " AND (\(orTerms))"
        } else {
            tagClause = ""
        }

        let sql = """
            SELECT COUNT(*)
            FROM chunks c
            LEFT JOIN brainbar_meta.agent_reads r
              ON r.chunk_id = c.id AND r.agent_id = ?
            WHERE r.chunk_id IS NULL\(tagClause)
        """

        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }

        let TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
        var paramIdx: Int32 = 1
        sqlite3_bind_text(stmt, paramIdx, agentID, -1, TRANSIENT)
        paramIdx += 1
        if let tags {
            for tag in tags where !tag.isEmpty {
                let pattern = "%\"\(tag)\"%"
                sqlite3_bind_text(stmt, paramIdx, pattern, -1, TRANSIENT)
                paramIdx += 1
            }
        }

        guard sqlite3_step(stmt) == SQLITE_ROW else { throw DBError.noResult }
        return Int(sqlite3_column_int(stmt, 0))
    }

    // MARK: - Helpers

    func exec(_ sql: String) {
        do {
            try execute(sql)
        } catch {
            NSLog("[BrainBar] SQL error: %@", String(describing: error))
        }
    }

    func metadataExec(_ sql: String) {
        do {
            try execute(sql, on: metadataDB)
        } catch {
            NSLog("[BrainBar] Metadata SQL error: %@", String(describing: error))
        }
    }

    private func execute(_ sql: String, on handle: OpaquePointer? = nil, retries: Int = 0, retryDelayMillis: UInt32 = 250) throws {
        guard let handle = handle ?? db else { throw DBError.notOpen }
        var attempts = 0

        while true {
            var errMsg: UnsafeMutablePointer<CChar>?
            let rc = sqlite3_exec(handle, sql, nil, nil, &errMsg)
            if rc == SQLITE_OK {
                sqlite3_free(errMsg)
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

    private func runWriteStatement(on handle: OpaquePointer?, sql: String, retries: Int = 0, retryDelayMillis: UInt32 = 250, bind: (OpaquePointer) -> Void) throws {
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

    private func columnText(_ stmt: OpaquePointer?, _ col: Int32) -> String? {
        guard let cStr = sqlite3_column_text(stmt, col) else { return nil }
        return String(cString: cStr)
    }

    private func openConnection(path: String) throws -> OpaquePointer {
        var handle: OpaquePointer?
        let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX
        let rc = sqlite3_open_v2(path, &handle, flags, nil)
        guard rc == SQLITE_OK, let handle else { throw DBError.open(path, rc) }
        return handle
    }

    private func configureConnection(_ handle: OpaquePointer?) throws {
        try execute("PRAGMA journal_mode = WAL", on: handle)
        try execute("PRAGMA busy_timeout = 5000", on: handle)
        try execute("PRAGMA cache_size = -64000", on: handle)
        try execute("PRAGMA synchronous = NORMAL", on: handle)
    }

    private func attachMetadataDatabase() throws {
        let escapedPath = metadataPath.replacingOccurrences(of: "'", with: "''")
        try execute("ATTACH DATABASE '\(escapedPath)' AS brainbar_meta", on: db)
    }

    private func encodeJSON(_ array: [String]) throws -> String {
        let data = try JSONSerialization.data(withJSONObject: array)
        return String(data: data, encoding: .utf8) ?? "[]"
    }

    private func decodeJSONArray(_ text: String?) -> [String] {
        guard let text,
              let data = text.data(using: .utf8),
              let array = try? JSONSerialization.jsonObject(with: data) as? [String] else {
            return []
        }
        return array
    }

    private func updateSubscriptionTimestamps(agentID: String, lastSeen: String?, lastDeliveredAt: String?, disconnectedAt: String?) throws {
        guard let metadataDB else { throw DBError.notOpen }
        var stmt: OpaquePointer?
        let sql = """
            INSERT INTO agent_subscriptions (agent_id, tags, last_seen, last_delivered_at, disconnected_at)
            VALUES (?, '[]', ?, ?, ?)
            ON CONFLICT(agent_id) DO UPDATE SET
                last_seen = COALESCE(excluded.last_seen, agent_subscriptions.last_seen),
                last_delivered_at = COALESCE(excluded.last_delivered_at, agent_subscriptions.last_delivered_at),
                disconnected_at = excluded.disconnected_at
        """
        let rc = sqlite3_prepare_v2(metadataDB, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }

        sqlite3_bind_text(stmt, 1, agentID, -1, unsafeBitCast(-1, to: sqlite3_destructor_type.self))
        bindOptionalText(lastSeen, to: stmt, index: 2)
        bindOptionalText(lastDeliveredAt, to: stmt, index: 3)
        bindOptionalText(disconnectedAt, to: stmt, index: 4)

        let stepRC = sqlite3_step(stmt)
        guard stepRC == SQLITE_DONE else { throw DBError.step(stepRC) }
    }

    private func bindOptionalText(_ value: String?, to stmt: OpaquePointer?, index: Int32) {
        let TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
        if let value {
            sqlite3_bind_text(stmt, index, value, -1, TRANSIENT)
        } else {
            sqlite3_bind_null(stmt, index)
        }
    }

    private static func timestamp() -> String {
        ISO8601DateFormatter().string(from: Date())
    }

    private static func makeMetadataPath(for primaryPath: String) -> String {
        let url = URL(fileURLWithPath: primaryPath)
        let directory = url.deletingLastPathComponent()
        let stem = url.deletingPathExtension().lastPathComponent
        let baseName = stem.isEmpty ? url.lastPathComponent : stem
        return directory.appendingPathComponent("\(baseName).brainbar-meta.db").path
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
        return tokens.joined(separator: " OR ")
    }

    // MARK: - Errors

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
